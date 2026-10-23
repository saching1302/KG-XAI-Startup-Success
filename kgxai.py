"""
KG-XAI: a knowledge-guided, explainable model for predicting startup success
from multi-source entrepreneurial data.

One file, end to end (python kgxai.py):

  Study 1  Kaggle "Startup Success Prediction" (922 U.S. startups, lifetime
           snapshot features) - the widely used benchmark.
  Study 2  Open Crunchbase snapshot (4 linked tables: companies, rounds,
           investments, acquisitions; 7,454 startups worldwide). Leakage-safe:
           every feature uses only information available 2 years after the
           first funding round; time-split test on later cohorts.

  E1  accuracy + knowledge consistency, 10 models, 5x5 repeated CV with
      nested hyper-parameter tuning, corrected resampled t-test;
      Study 2 also trains on 2005-09 cohorts and tests on 2010-12 cohorts.
  E2  wrong-knowledge stress test: flip k = 0..3 rules and check that
      in-fold refinement detects them while naive knowledge injection fails.
  E3  rule-threshold sensitivity (x0.5 ... x1.5).

Outputs (./results): Table 1, Table 2, Figures 1-2, supplementary tables and
figures, results.json.

Usage:  python kgxai.py                 # full run, ~10-20 min on 4 CPUs
        python kgxai.py --quick         # smoke test, ~2 min
"""
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
OUT = "results"
S1_PATH = "data/startup_data.csv"
S1_URL = ("https://raw.githubusercontent.com/sumitjhaleriya/"
          "Startup-Success-Prediction-using-Machine-Learning/main/startup%20data.csv")
S2_CACHE = "data/crunchbase_s2_features.csv"
S2_RAW = "data/crunchbase_raw"
S2_URL = "https://raw.githubusercontent.com/notpeter/crunchbase-data/master/{}.csv"
WINDOW_DAYS = 730          # prediction point = first funding + 2 years
TRAIN_UNTIL = 2009         # time split: train cohorts <= 2009, test 2010-2012


# =========================================================================== #
# 1. Datasets and knowledge bases                                              #
# =========================================================================== #
@dataclass
class Rule:
    id: str
    name: str
    theory: str
    sign: int
    cond: Callable          # cond(X, s) -> bool Series; s scales numeric thresholds
    links: tuple = ()       # raw features whose direction the same theory fixes


@dataclass
class Dataset:
    key: str
    name: str
    X: pd.DataFrame
    y: np.ndarray
    year: np.ndarray
    views: dict
    rules: list
    cont: set = field(default_factory=set)
    couple: dict = field(default_factory=dict)

    @property
    def base(self):
        return [f for fs in self.views.values() for f in fs]

    @property
    def view_of(self):
        return {f: v for v, fs in self.views.items() for f in fs}

    @property
    def signed(self):
        return {f: r.sign for r in self.rules for f in r.links}

    @property
    def rule_ids(self):
        return [r.id for r in self.rules]

    def subset(self, mask):
        return Dataset(self.key, self.name, self.X[mask].reset_index(drop=True), self.y[mask],
                       self.year[mask], self.views, self.rules, self.cont, self.couple)


def study1() -> Dataset:
    if not os.path.exists(S1_PATH):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(S1_URL, S1_PATH)
    d = pd.read_csv(S1_PATH).drop_duplicates("id").reset_index(drop=True)
    X = pd.DataFrame(index=d.index)
    X["log_funding"] = np.log1p(d.funding_total_usd)
    for c in ["funding_rounds", "has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC",
              "has_roundD", "relationships", "avg_participants", "is_top500", "milestones",
              "age_first_funding_year", "age_last_funding_year", "is_CA", "is_NY", "is_MA", "is_TX",
              "is_otherstate", "is_software", "is_web", "is_mobile", "is_enterprise",
              "is_advertising", "is_gamesvideo", "is_ecommerce", "is_biotech", "is_othercategory"]:
        X[c] = d[c].astype(float)
    X["age_first_milestone_year"] = d.age_first_milestone_year.fillna(0.0)
    X["age_last_milestone_year"] = d.age_last_milestone_year.fillna(0.0)
    X["funding_span"] = d.age_last_funding_year - d.age_first_funding_year
    X["founded_year"] = pd.to_datetime(d.founded_at, format="%m/%d/%Y").dt.year.astype(float)
    views = {
        "Funding": ["log_funding", "funding_rounds", "has_VC", "has_angel", "has_roundA",
                    "has_roundB", "has_roundC", "has_roundD"],
        "Network": ["relationships", "avg_participants", "is_top500"],
        "Milestone": ["milestones", "age_first_milestone_year", "age_last_milestone_year"],
        "Temporal": ["age_first_funding_year", "age_last_funding_year", "funding_span", "founded_year"],
        "Ecosystem": ["is_CA", "is_NY", "is_MA", "is_TX", "is_otherstate"],
        "Sector": ["is_software", "is_web", "is_mobile", "is_enterprise", "is_advertising",
                   "is_gamesvideo", "is_ecommerce", "is_biotech", "is_othercategory"],
    }
    rules = [
        Rule("R1", "VC certification", "Signalling / VC certification", +1,
             lambda X, s: X.has_VC == 1, ("has_VC", "has_roundA")),
        Rule("R2", "Staged-financing progression", "Staged VC financing", +1,
             lambda X, s: (X.has_roundB + X.has_roundC + X.has_roundD) > 0,
             ("funding_rounds", "has_roundB", "has_roundC", "has_roundD")),
        Rule("R3", "Resource munificence", "Resource-based view", +1,
             lambda X, s: X.log_funding >= np.log1p(10e6 * s), ("log_funding",)),
        Rule("R4", "Investor syndication", "VC syndication", +1,
             lambda X, s: X.avg_participants >= 3 * s, ("avg_participants",)),
        Rule("R5", "Network embeddedness", "Inter-organisational endorsement", +1,
             lambda X, s: X.relationships >= 5 * s, ("relationships",)),
        Rule("R6", "Legitimacy milestones", "Legitimacy-based growth", +1,
             lambda X, s: X.milestones >= 2 * s, ("milestones",)),
        Rule("R7", "Ecosystem hub", "Clusters and entrepreneurship", +1,
             lambda X, s: (X.is_CA + X.is_MA + X.is_NY) > 0),
        Rule("R8", "Funding stagnation", "Liability of newness", -1,
             lambda X, s: (X.funding_rounds == 1) & (X.milestones == 0)),
        Rule("R9", "Angel-only financing", "Weak certification", -1,
             lambda X, s: (X.has_angel == 1) & (X.has_VC == 0) & (X.has_roundA == 0)),
    ]
    inc_rounds = [("inc", "funding_rounds")]
    return Dataset("S1", "Study 1 (Kaggle, 922 U.S. startups)", X[[f for v in views.values() for f in v]],
                   (d.status == "acquired").astype(int).to_numpy(), X.founded_year.to_numpy(), views, rules,
                   cont={"log_funding", "avg_participants"},
                   couple={f: inc_rounds for f in ["has_roundA", "has_roundB", "has_roundC", "has_roundD"]})


S2_HUBS = {"hub_sf": "SF Bay Area", "hub_nyc": "New York City", "hub_boston": "Boston",
           "hub_la": "Los Angeles", "hub_london": "London", "hub_seattle": "Seattle"}
S2_SECTORS = {"sec_software": "Software", "sec_biotech": "Biotechnology", "sec_mobile": "Mobile",
              "sec_web": "Curated Web", "sec_ecommerce": "E-Commerce", "sec_social": "Social Media",
              "sec_advertising": "Advertising", "sec_enterprise": "Enterprise Software",
              "sec_games": "Games", "sec_health": "Health Care"}


def build_s2_features(raw: str) -> pd.DataFrame:
    """Join the four Crunchbase tables into one leakage-safe row per startup.
    Prediction point t0 = first funding + 2 years; only rounds, investments
    and investor exits dated before t0 are used. Outcome = acquired or IPO
    (success) vs closed, observed by the Dec-2015 snapshot."""
    for f in ("companies", "rounds", "investments", "acquisitions"):
        p = f"{raw}/{f}.csv"
        if not os.path.exists(p):
            os.makedirs(raw, exist_ok=True)
            print(f"downloading {f}.csv ...")
            urllib.request.urlretrieve(S2_URL.format(f), p)
    c = pd.read_csv(f"{raw}/companies.csv")
    r = pd.read_csv(f"{raw}/rounds.csv")
    inv = pd.read_csv(f"{raw}/investments.csv")
    acq = pd.read_csv(f"{raw}/acquisitions.csv")
    for df, col in [(c, "first_funding_at"), (c, "founded_at"), (r, "funded_at"),
                    (inv, "funded_at"), (acq, "acquired_at")]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    acq = acq[(acq.acquired_at >= "1990-01-01") & (acq.acquired_at <= "2015-12-31")]
    exit_at = acq.groupby("company_permalink").acquired_at.min()

    c = c[c.status.isin(["acquired", "ipo", "closed"])]
    c = c[(c.first_funding_at >= "2005-01-01") & (c.first_funding_at <= "2012-12-31")].copy()
    c["t0"] = c.first_funding_at + pd.Timedelta(days=WINDOW_DAYS)
    c["exit_at"] = c.permalink.map(exit_at)
    early = (c.status == "acquired") & (c.exit_at.isna() | (c.exit_at <= c.t0))
    c = c[~early].set_index("permalink")
    t0, ff = c.t0, c.first_funding_at

    # --- funding view: rounds table, inside the window ------------------------
    rw = r[r.company_permalink.isin(c.index)].copy()
    rw = rw[rw.funded_at <= rw.company_permalink.map(t0)]
    g = rw.groupby("company_permalink")
    F = pd.DataFrame(index=c.index)
    F["n_rounds_w"] = g.size()
    F["log_raised_w"] = np.log1p(g.raised_amount_usd.sum(min_count=1))
    first = rw[rw.funded_at <= rw.company_permalink.map(ff)]
    F["log_first_round"] = np.log1p(first.groupby("company_permalink").raised_amount_usd.sum(min_count=1))
    for name, typ in [("has_seed", "seed"), ("has_angel", "angel"), ("has_venture", "venture")]:
        F[name] = rw.assign(v=rw.funding_round_type.eq(typ)).groupby("company_permalink").v.max()
    F["n_venture_rounds"] = rw.funding_round_type.eq("venture").groupby(rw.company_permalink).sum()
    F["has_seriesB_plus"] = (rw.funding_round_type.eq("venture")
                             & rw.funding_round_code.isin(list("BCDEFGH"))).groupby(rw.company_permalink).max()
    F["has_debt_grant"] = rw.funding_round_type.isin(["debt_financing", "grant"]).groupby(rw.company_permalink).max()
    second = rw.sort_values("funded_at").groupby("company_permalink").funded_at.apply(
        lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)
    F["months_to_second"] = ((second - ff.reindex(second.index)).dt.days / 30.44).reindex(c.index)

    # --- network + investor track record: investments + acquisitions tables ---
    iw = inv[inv.company_permalink.isin(c.index)].copy()
    iw = iw[iw.funded_at <= iw.company_permalink.map(t0)].dropna(subset=["investor_permalink"])
    gi = iw.groupby("company_permalink")
    F["n_investors_w"] = gi.investor_permalink.nunique()
    F["avg_investors_per_round"] = gi.size() / gi.funding_round_permalink.nunique()
    deals = inv.dropna(subset=["investor_permalink", "funded_at"])
    dates = {k: np.sort(v.to_numpy()) for k, v in deals.groupby("investor_permalink").funded_at}
    port = deals.groupby("investor_permalink").company_permalink.unique()
    exits = {k: np.sort(exit_at.reindex(v).dropna().to_numpy()) for k, v in port.items()}
    pairs = iw[["company_permalink", "investor_permalink"]].drop_duplicates()
    T = pairs.company_permalink.map(t0).to_numpy()
    own = pairs.company_permalink.map(c.exit_at).to_numpy()
    exp_, ex_ = [], []
    for inv_id, t, o in zip(pairs.investor_permalink.to_numpy(), T, own):
        exp_.append(np.searchsorted(dates[inv_id], t))
        e = exits.get(inv_id, np.array([], dtype="datetime64[ns]"))
        n = np.searchsorted(e, t)
        ex_.append(n - int(not pd.isna(o) and o < t))
    pairs = pairs.assign(exp=exp_, ex=ex_)
    F["max_investor_experience"] = np.log1p(pairs.groupby("company_permalink").exp.max())
    F["max_investor_exits"] = np.log1p(pairs.groupby("company_permalink").ex.max())

    # --- temporal, ecosystem, sector: companies table --------------------------
    F["first_funding_year"] = ff.dt.year
    age = (ff - c.founded_at).dt.days / 365.25
    F["founded_known"] = age.notna().astype(float)
    F["age_at_first_funding"] = age.clip(-1, 30).fillna(-1)
    F["is_USA"] = c.country_code.eq("USA").astype(float)
    for k, v in S2_HUBS.items():
        F[k] = c.region.eq(v).astype(float)
    cats = c.category_list.fillna("")
    for k, v in S2_SECTORS.items():
        F[k] = cats.str.split("|").apply(lambda xs, v=v: float(v in xs))

    # undisclosed amounts and unlisted investors are MISSING, not zero: a zero
    # would wrongly rank "undisclosed" below every disclosed round
    F["amount_disclosed"] = F.log_raised_w.notna().astype(float)
    F["investors_listed"] = F.n_investors_w.fillna(0).gt(0).astype(float)
    keep_nan = ["log_raised_w", "log_first_round", "avg_investors_per_round",
                "max_investor_experience", "max_investor_exits"]
    fill = {c: 0.0 for c in F.columns if c not in keep_nan}
    F = F.fillna({**fill, "months_to_second": 25.0}).astype(float)
    F["success"] = c.status.ne("closed").astype(int)
    F["status"] = c.status
    return F.reset_index()


def study2(raw: str = S2_RAW) -> Dataset:
    if not os.path.exists(S2_CACHE):
        os.makedirs("data", exist_ok=True)
        build_s2_features(raw).to_csv(S2_CACHE, index=False)
    d = pd.read_csv(S2_CACHE)
    views = {
        "Funding": ["n_rounds_w", "log_raised_w", "log_first_round", "amount_disclosed", "has_seed",
                    "has_angel", "has_venture", "n_venture_rounds", "has_seriesB_plus", "has_debt_grant"],
        "Network": ["n_investors_w", "investors_listed", "avg_investors_per_round", "max_investor_experience"],
        "Track record": ["max_investor_exits"],
        "Temporal": ["first_funding_year", "age_at_first_funding", "founded_known", "months_to_second"],
        "Ecosystem": ["is_USA", *S2_HUBS],
        "Sector": list(S2_SECTORS),
    }
    hubs = list(S2_HUBS)
    rules = [
        Rule("R1", "VC certification", "Signalling / VC certification", +1,
             lambda X, s: X.has_venture == 1, ("has_venture", "n_venture_rounds")),
        Rule("R2", "Staged-financing progression", "Staged VC financing", +1,
             lambda X, s: X.n_rounds_w >= 2 * s, ("n_rounds_w", "has_seriesB_plus")),
        Rule("R3", "Resource munificence", "Resource-based view", +1,
             lambda X, s: X.log_raised_w >= np.log1p(10e6 * s), ("log_raised_w", "log_first_round")),
        Rule("R4", "Investor syndication", "VC syndication", +1,
             lambda X, s: X.avg_investors_per_round >= 3 * s, ("avg_investors_per_round", "n_investors_w")),
        Rule("R5", "Prominent-investor endorsement", "Inter-organisational endorsement", +1,
             lambda X, s: X.max_investor_experience >= np.log1p(20 * s), ("max_investor_experience",)),
        Rule("R6", "Investor track record", "Legitimacy-based growth", +1,
             lambda X, s: X.max_investor_exits >= np.log1p(3 * s), ("max_investor_exits",)),
        Rule("R7", "Ecosystem hub", "Clusters and entrepreneurship", +1,
             lambda X, s: X[hubs].sum(axis=1) > 0),
        Rule("R8", "Funding stagnation", "Liability of newness", -1,
             lambda X, s: (X.n_rounds_w == 1) & (X.log_raised_w < np.log1p(1e6))),
        Rule("R9", "Angel/seed-only financing", "Weak certification", -1,
             lambda X, s: ((X.has_angel + X.has_seed) > 0) & (X.has_venture == 0)),
    ]
    X = d[[f for v in views.values() for f in v]]
    couple = {"has_venture": [("inc", "n_venture_rounds"), ("inc", "n_rounds_w")],
              "n_venture_rounds": [("one", "has_venture"), ("inc", "n_rounds_w")],
              "has_seriesB_plus": [("one", "has_venture"), ("inc", "n_venture_rounds"), ("inc", "n_rounds_w")]}
    return Dataset("S2", "Study 2 (open Crunchbase, 4 linked tables)", X, d.success.to_numpy(),
                   d.first_funding_year.to_numpy(), views, rules,
                   cont={"log_raised_w", "log_first_round", "avg_investors_per_round",
                         "max_investor_experience", "max_investor_exits"}, couple=couple)


# =========================================================================== #
# 2. Knowledge encoding and refinement                                         #
# =========================================================================== #
def theory_signs(ds: Dataset) -> dict:
    return {**ds.signed, **{r.id: r.sign for r in ds.rules}}


def add_knowledge(ds, X, signs, scale=1.0):
    """Rule activations r_j(x) and the signed knowledge score s_K(x)."""
    K = X.copy()
    for r in ds.rules:
        K[r.id] = r.cond(X, scale).astype(float)
    K["kb_score"] = sum(signs[r.id] * K[r.id] for r in ds.rules)
    return K


def refine(ds, X, y, given, scale=1.0):
    """Knowledge validation on the training fold: keep a stated direction only if
    the training data agree with it; release (0) contradicted knowledge."""
    K = add_knowledge(ds, X, given, scale)
    out = dict(given)
    for r in ds.rules:
        on = K[r.id].to_numpy() == 1
        gap = y[on].mean() - y[~on].mean() if 0 < on.sum() < len(y) else 0.0
        out[r.id] = given[r.id] if np.sign(gap) == given[r.id] else 0
    for f in ds.signed:
        rho = stats.spearmanr(X[f], y, nan_policy="omit")[0]
        out[f] = given[f] if np.sign(rho) == given[f] else 0
    return out


# =========================================================================== #
# 3. Models                                                                    #
# =========================================================================== #
GBM = dict(n_estimators=300, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=1)
GRID = {
    "lr": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "rf": [{"min_samples_leaf": m} for m in (1, 5, 20)],
    "hgb": [{"learning_rate": a, "max_leaf_nodes": b} for a in (0.05, 0.1) for b in (15, 31)],
    "gbm": [{"num_leaves": b, "learning_rate": a} for b in (7, 15, 31) for a in (0.03, 0.1)],
    "gam": [{"num_leaves": b, "learning_rate": a} for b in (3, 7) for a in (0.03, 0.1)],
}
DEFAULT = {"lr": GRID["lr"][2], "rf": GRID["rf"][1], "hgb": GRID["hgb"][0],
           "gbm": {"num_leaves": 15, "learning_rate": 0.03}, "gam": GRID["gam"][2]}


@dataclass
class Spec:
    name: str
    kind: str          # lr | rf | hgb | gbm | gam
    space: str         # base | rules | full
    knowledge: str     # none | impose | refine
    constrained: bool = False

    @property
    def tree(self):
        return self.kind in ("gbm", "gam")


PROPOSED = "KG-XAI (proposed)"
NAIVE = "KG-XAI w/o refinement"
SPECS = [
    Spec("Logistic regression", "lr", "base", "none"),
    Spec("Random forest", "rf", "base", "none"),
    Spec("Hist. gradient boosting", "hgb", "base", "none"),
    Spec("LightGBM", "gbm", "base", "none"),
    Spec("Boosted GAM (EBM-style)", "gam", "base", "none"),
    Spec("Knowledge base only", "lr", "rules", "impose"),
    Spec(NAIVE, "gbm", "full", "impose", True),
    Spec("KG-XAI w/o constraints", "gbm", "full", "refine", False),
    Spec("KG-XAI w/o rule features", "gbm", "base", "refine", True),
    Spec(PROPOSED, "gbm", "full", "refine", True),
]


def make_estimator(kind, hp, cols, signs, constrained):
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=hp["C"], max_iter=3000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1, **hp)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, random_state=SEED, **hp)
    extra = {}
    if constrained:
        sign = {**signs, "kb_score": +1}
        extra = dict(monotone_constraints=[sign.get(c, 0) for c in cols],
                     monotone_constraints_method="advanced")
    if kind == "gam":   # one feature per tree = additive shape functions (EBM idea)
        extra["interaction_constraints"] = [[i] for i in range(len(cols))]
    return lgb.LGBMClassifier(**GBM, **hp, **extra)


class Fitted:
    def __init__(self, ds, spec, X, y, given=None, tune=True, scale=1.0):
        self.ds, self.spec, self.scale = ds, spec, scale
        given = given or theory_signs(ds)
        self.signs = refine(ds, X, y, given, scale) if spec.knowledge == "refine" else given
        F = self.features(X)
        self.cols = list(F.columns)
        self.hp = self._tune(F, y) if tune else DEFAULT[spec.kind]
        self.est = make_estimator(spec.kind, self.hp, self.cols, self.signs, spec.constrained).fit(F, y)

    def features(self, X):
        K = add_knowledge(self.ds, X, self.signs, self.scale)
        rules = self.ds.rule_ids + ["kb_score"]
        return {"base": K[self.ds.base], "rules": K[rules], "full": K}[self.spec.space]

    def _tune(self, F, y):
        cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
        best, best_hp = -1, None
        for hp in GRID[self.spec.kind]:
            s = []
            for a, b in cv.split(F, y):
                m = make_estimator(self.spec.kind, hp, self.cols, self.signs, self.spec.constrained)
                m.fit(F.iloc[a], y[a])
                s.append(roc_auc_score(y[b], m.predict_proba(F.iloc[b])[:, 1]))
            if np.mean(s) > best:
                best, best_hp = np.mean(s), hp
        return best_hp

    def proba(self, X):
        return self.est.predict_proba(self.features(X)[self.cols])[:, 1]

    def shap(self, X):  # exact TreeSHAP (Lundberg et al., 2020) inside LightGBM
        return self.est.predict(self.features(X)[self.cols], pred_contrib=True)[:, :-1]


# =========================================================================== #
# 4. Metrics                                                                   #
# =========================================================================== #
def precision_at(y, p, q=0.10):
    k = max(1, int(math.ceil(q * len(y))))
    return float(y[np.argsort(-p)[:k]].mean())


def violation_rate(ds, model, X):
    """Directional violation rate: share of (startup, theory-signed feature)
    probes where a theory-consistent improvement LOWERS predicted success.
    Returns (any decrease, decrease >= 1 percentage point)."""
    p0, bad, mat, n = model.proba(X), 0, 0, 0
    for f in ds.signed:
        Xp = X.copy()
        if set(np.unique(X[f])) <= {0.0, 1.0}:
            mask = (X[f] == 0).to_numpy()
            Xp[f] = 1.0
        else:
            mask = X[f].notna().to_numpy()
            Xp[f] = X[f] + (X[f].std() if f in ds.cont else 1.0)
        for how, g in ds.couple.get(f, []):
            Xp[g] = np.where(mask, 1.0, X[g]) if how == "one" else X[g] + mask
        if mask.sum() == 0:
            continue
        d = model.proba(Xp) - p0
        bad += int((d < -1e-6)[mask].sum())
        mat += int((d < -0.01)[mask].sum())
        n += int(mask.sum())
    return bad / n, mat / n


def evaluate(ds, m, Xte, yte, with_shap=False):
    p = m.proba(Xte)
    v, v1 = violation_rate(ds, m, Xte)
    rec = {"auc": roc_auc_score(yte, p), "brier": brier_score_loss(yte, p),
           "p10": precision_at(yte, p), "f1": f1_score(yte, p >= 0.5),
           "acc": accuracy_score(yte, p >= 0.5), "dvr": v, "dvr1": v1}
    if with_shap and m.spec.tree:
        S = m.shap(Xte)
        rec["imp"] = dict(zip(m.cols, np.abs(S).mean(0)))
        rec["S"], rec["cols"] = S, m.cols
    return rec


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV."""
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1) * (1 / len(d) + n_test / n_train)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt(var)
    return float(t), float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def stability(imps):
    """Mean pairwise Spearman rho of global |SHAP| rankings across folds."""
    keys = list(imps[0])
    v = [[i[k] for k in keys] for i in imps]
    return float(np.mean([stats.spearmanr(a, b)[0] for a, b in combinations(v, 2)]))


# =========================================================================== #
# 5. Experiments                                                               #
# =========================================================================== #
def _fold(ds, specs, tr, te, tune, given=None, scale=1.0):
    Xtr, Xte, ytr, yte = ds.X.iloc[tr], ds.X.iloc[te], ds.y[tr], ds.y[te]
    out = {}
    for sp in specs:
        m = Fitted(ds, sp, Xtr, ytr, given=given, tune=tune, scale=scale)
        rec = evaluate(ds, m, Xte, yte, with_shap=True)
        rec.update(signs=m.signs, hp=m.hp, te=te)
        out[sp.name] = rec
    return out


def cv_runs(ds, specs, repeats, jobs, tune=True, given=None, scale=1.0, seed=SEED):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    return Parallel(n_jobs=jobs)(delayed(_fold)(ds, specs, tr, te, tune, given, scale)
                                 for tr, te in cv.split(ds.X, ds.y))


def e1_main(ds, repeats, jobs):
    folds = cv_runs(ds, SPECS, repeats, jobs)
    n = len(ds.y)
    res = {}
    ref = [f[PROPOSED]["auc"] for f in folds]
    for sp in SPECS:
        rs = [f[sp.name] for f in folds]
        r = {k: [x[k] for x in rs] for k in ("auc", "brier", "p10", "f1", "acc", "dvr", "dvr1")}
        r["p_auc"] = None if sp.name == PROPOSED else corrected_ttest(ref, r["auc"], n - n // 5, n // 5)[1]
        r["stability"] = stability([x["imp"] for x in rs]) if sp.tree else None
        r["hp"] = [x["hp"] for x in rs]
        res[sp.name] = r
    kept = pd.DataFrame([f[PROPOSED]["signs"] for f in folds])
    S = np.zeros((n, len(folds[0][PROPOSED]["cols"])))
    for f in folds[:5]:                         # first repeat = one OOF value per startup
        S[f[PROPOSED]["te"]] = f[PROPOSED]["S"]
    return res, kept, pd.DataFrame(S, columns=folds[0][PROPOSED]["cols"])


def e2_stress(ds, jobs, draws=10, seed=SEED):
    """Flip k stated rule directions (and their linked features) and compare
    in-fold refinement with naive knowledge injection, judged against the
    theory (true) directions. Only rules the full data confirm are flipped."""
    theory = theory_signs(ds)
    K = add_knowledge(ds, ds.X, theory)
    conf = [r for r in ds.rules if r.links and np.sign(
        ds.y[K[r.id] == 1].mean() - ds.y[K[r.id] == 0].mean()) == r.sign]
    rng = np.random.default_rng(seed)
    rows = []
    specs = [s for s in SPECS if s.name in (PROPOSED, NAIVE)]
    for k in range(0, 4):
        combos = list(combinations([r.id for r in conf], k))
        pick = combos if len(combos) <= draws else [combos[i] for i in rng.choice(len(combos), draws, replace=False)]
        for flip in pick:
            given = dict(theory)
            for r in ds.rules:
                if r.id in flip:
                    for key in (r.id, *r.links):
                        given[key] = -theory[key]
            folds = cv_runs(ds, specs, 1, jobs, tune=False, given=given, seed=seed + 7)
            comp = [key for r in ds.rules if r.id in flip for key in (r.id, *r.links)]
            good = [key for r in conf if r.id not in flip for key in (r.id, *r.links)]
            for name in (PROPOSED, NAIVE):
                fs = [f[name] for f in folds]
                rows.append({"dataset": ds.key, "k": k, "flipped": "+".join(flip) or "-", "model": name,
                             "auc": np.mean([f["auc"] for f in fs]), "dvr": np.mean([f["dvr"] for f in fs]),
                             "dvr1": np.mean([f["dvr1"] for f in fs]),
                             "recovery": np.mean([np.mean([f["signs"][c] == 0 for c in comp]) for f in fs])
                             if comp else np.nan,
                             "false_release": np.mean([np.mean([f["signs"][c] == 0 for c in good]) for f in fs])
                             if good else np.nan})
    return pd.DataFrame(rows), [r.id for r in conf]


def e3_sensitivity(ds, jobs, scales=(0.5, 0.75, 1.0, 1.25, 1.5)):
    rows = []
    spec = [s for s in SPECS if s.name == PROPOSED]
    for s in scales:
        folds = cv_runs(ds, spec, 1, jobs, tune=False, scale=s, seed=SEED + 11)
        fs = [f[PROPOSED] for f in folds]
        rows.append({"dataset": ds.key, "scale": s, "auc": np.mean([f["auc"] for f in fs]),
                     "dvr": np.mean([f["dvr"] for f in fs]),
                     "rules_retained": np.mean([sum(f["signs"][r] != 0 for r in ds.rule_ids) for f in fs])})
    return pd.DataFrame(rows)


# =========================================================================== #
# 6. Tables                                                                    #
# =========================================================================== #
def ms(v, d=3):
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"


def mark(p):
    return "" if p is None else ("‡" if p < 0.01 else "†" if p < 0.05 else "")


def table1(main, ts, out):
    rows, full = [], []
    for sp in SPECS:
        a, b, t = main["S1"][sp.name], main["S2"][sp.name], ts[sp.name]
        rows.append({
            "Model": sp.name,
            "S1 AUC": ms(a["auc"]) + mark(a["p_auc"]),
            "S1 DVR (%)": f"{100 * np.mean(a['dvr']):.1f}",
            "S2 AUC": ms(b["auc"]) + mark(b["p_auc"]),
            "S2 P@10%": f"{np.mean(b['p10']):.3f}",
            "S2 Brier": f"{np.mean(b['brier']):.3f}",
            "S2 DVR (%)": f"{100 * np.mean(b['dvr']):.1f}",
            "S2 time-split AUC": f"{t['auc']:.3f}" + t.get("mark", ""),
            "S2 SHAP stability": "–" if b["stability"] is None else f"{b['stability']:.3f}",
        })
        for key, r in (("S1", a), ("S2", b)):
            full.append({"dataset": key, "model": sp.name,
                         **{m: np.mean(r[m]) for m in ("auc", "brier", "p10", "f1", "acc", "dvr", "dvr1")},
                         **{f"{m}_sd": np.std(r[m], ddof=1) for m in ("auc", "brier", "p10")},
                         "p_auc_vs_proposed": r["p_auc"], "stability": r["stability"]})
        full.append({"dataset": "S2-time", "model": sp.name,
                     **{m: t[m] for m in ("auc", "brier", "p10", "dvr", "dvr1")},
                     "p_auc_vs_proposed": t.get("p")})
    T = pd.DataFrame(rows)
    T.to_csv(f"{out}/table1_performance.csv", index=False)
    pd.DataFrame(full).to_csv(f"{out}/table1_full_metrics.csv", index=False)
    with open(f"{out}/table1_performance.md", "w") as fh:
        fh.write(T.to_markdown(index=False))
    return T


def table2(stress, out):
    g = stress.groupby(["dataset", "k", "model"])
    agg = g.agg(auc=("auc", "mean"), dvr=("dvr", "mean"), recovery=("recovery", "mean"),
                false_release=("false_release", "mean"), draws=("auc", "size")).reset_index()
    rows = []
    for (dset, k), sub in agg.groupby(["dataset", "k"]):
        p, nv = sub[sub.model == PROPOSED].iloc[0], sub[sub.model == NAIVE].iloc[0]
        rows.append({"Study": dset, "Flipped rules (k)": int(k), "Draws": int(p.draws),
                     "AUC refined": f"{p.auc:.3f}", "AUC naive": f"{nv.auc:.3f}",
                     "DVR refined (%)": f"{100 * p.dvr:.1f}", "DVR naive (%)": f"{100 * nv.dvr:.1f}",
                     "Flipped knowledge released (%)": "–" if k == 0 else f"{100 * p.recovery:.0f}",
                     "Correct knowledge released (%)": f"{100 * p.false_release:.1f}"})
    T = pd.DataFrame(rows)
    T.to_csv(f"{out}/table2_knowledge_robustness.csv", index=False)
    stress.to_csv(f"{out}/table2_stress_test_draws.csv", index=False)
    with open(f"{out}/table2_knowledge_robustness.md", "w") as fh:
        fh.write(T.to_markdown(index=False))
    return T


def table_s1(ds, kept, S, out):
    K = add_knowledge(ds, ds.X, theory_signs(ds))
    y, rows = ds.y, []
    for r in ds.rules:
        on = K[r.id].to_numpy() == 1
        rows.append({"Study": ds.key, "Rule": r.id, "Knowledge rule": r.name, "Theory": r.theory,
                     "Expected": "+" if r.sign > 0 else "−", "Support (%)": round(100 * on.mean(), 1),
                     "Success if fired (%)": round(100 * y[on].mean(), 1),
                     "Success if not (%)": round(100 * y[~on].mean(), 1),
                     "Lift": round(y[on].mean() / y.mean(), 2),
                     "Retained in folds (%)": round(100 * (kept[r.id] != 0).mean()),
                     "Mean SHAP when fired": round(float(S[r.id][on].mean()), 3)})
    return pd.DataFrame(rows)


# =========================================================================== #
# 7. Figures                                                                   #
# =========================================================================== #
INK, QUIET, EDGE, ACCENT, WARM = "#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#eb6834"
LIGHT = "#86b6ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": EDGE,
                     "axes.labelcolor": INK, "xtick.color": QUIET, "ytick.color": QUIET})


def figure1(n1, n2, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    ax.set_xlim(0, 100), ax.set_ylim(0, 50), ax.axis("off")

    def box(x, y, w, h, title, lines, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=(0.165, 0.47, 0.84, 0.10) if accent else "white",
                                    ec=ACCENT if accent else EDGE, lw=1.4 if accent else 0.9))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=6.5,
                fontweight="bold", color=INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 7.0 - 3.3 * i, ln, ha="center", va="top", fontsize=6.2, color=QUIET)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=EDGE, lw=0.9))

    box(0.5, 6, 18, 38, "Multi-source data", ["funding rounds", "investor network",
                                               "investor track record", "milestones, timing",
                                               "ecosystem, sector", "", f"S1: {n1:,} startups",
                                               f"S2: {n2:,} startups"])
    box(25, 27, 22, 17, "Knowledge base", ["9 theory-grounded rules", "signalling, syndication,",
                                           "legitimacy, clusters"])
    box(25, 6, 22, 17, "Knowledge refinement", ["rules kept only if the", "training fold agrees;",
                                                "s_K(x) = Σ d_j r_j(x)"])
    box(53.5, 6, 21, 38, "KG-XAI learner", ["gradient-boosted trees", "on x ⊕ r(x) ⊕ s_K(x)", "",
                                            "monotone constraints", "follow the refined", "direction d_j"],
        accent=True)
    box(80.5, 27, 19, 17, "Prediction", ["P(exit | x)", "decision support for", "investors, accelerators"])
    box(80.5, 6, 19, 17, "Explanation", ["exact TreeSHAP", "by view and by rule,", "consistency audit"])
    arrow(19.3, 35.5, 24.6, 35.5), arrow(19.3, 14.5, 24.6, 14.5)
    arrow(36, 26.6, 36, 23.4), arrow(47.4, 14.5, 53.1, 14.5), arrow(47.4, 35.5, 53.1, 35.5)
    arrow(74.9, 35.5, 80.1, 35.5), arrow(74.9, 14.5, 80.1, 14.5)
    fig.savefig(f"{out}/fig1_framework.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def view_importance(ds, S):
    vi = {v: np.abs(S[[c for c in S.columns if ds.view_of.get(c) == v]]).sum(1).mean() for v in ds.views}
    vi["Knowledge rules"] = np.abs(S[ds.rule_ids + ["kb_score"]]).sum(1).mean()
    return pd.Series(vi).sort_values()


def figure2(ds2, S2, stress, out):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300, gridspec_kw={"width_ratios": [1.15, 1, 1]})
    vi = view_importance(ds2, S2)
    a.barh(vi.index, vi.values, color=[ACCENT if v == "Knowledge rules" else LIGHT for v in vi.index], height=0.6)
    for i, v in enumerate(vi.values):
        a.text(v + vi.max() * 0.02, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK)
    a.set_xlim(0, vi.max() * 1.25)
    a.tick_params(axis="y", labelsize=6.8)
    a.set_xlabel("Mean |SHAP| per startup (log-odds)")
    a.set_title("(a) S2: contribution by source", fontsize=7.5, loc="left", color=INK)

    agg = stress.groupby(["dataset", "k", "model"]).agg(auc=("auc", "mean"), dvr=("dvr", "mean")).reset_index()
    for dset, col in (("S1", WARM), ("S2", ACCENT)):
        for name, ls, lab in ((PROPOSED, "-", "refined"), (NAIVE, "--", "naive")):
            s = agg[(agg.dataset == dset) & (agg.model == name)]
            b.plot(s.k, s.auc, ls=ls, color=col, lw=1.6, marker="o", ms=3.5, label=f"{dset} {lab}")
            c.plot(s.k, 100 * s.dvr, ls=ls, color=col, lw=1.6, marker="o", ms=3.5, label=f"{dset} {lab}")
    for axx, t, yl in ((b, "(b) Accuracy under wrong rules", "AUC (5-fold CV)"),
                       (c, "(c) Theory violations", "DVR vs true theory (%)")):
        axx.set_xticks([0, 1, 2, 3])
        axx.set_xlabel("Rules given the wrong direction (k)")
        axx.set_ylabel(yl)
        axx.set_title(t, fontsize=7.5, loc="left", color=INK)
        axx.grid(axis="y", color="#e6e5e0", lw=0.6)
    c.legend(fontsize=6, frameon=False, loc="upper left")
    for axx in (a, b, c):
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{out}/fig2_results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return vi


LABEL = {
    "log_funding": "Total funding (log)", "funding_rounds": "Funding rounds", "relationships": "Relationships",
    "milestones": "Milestones", "avg_participants": "Avg. investors per round", "is_top500": "Top-500 flag",
    "age_first_milestone_year": "Age at first milestone", "age_last_milestone_year": "Age at last milestone",
    "age_first_funding_year": "Age at first funding", "age_last_funding_year": "Age at last funding",
    "funding_span": "Funding span (yrs)", "founded_year": "Founding year",
    "log_raised_w": "Raised in window (log)", "log_first_round": "First round size (log)",
    "n_rounds_w": "Rounds in window", "n_investors_w": "Investors in window",
    "avg_investors_per_round": "Avg. investors per round", "max_investor_experience": "Lead investor experience",
    "max_investor_exits": "Lead investor prior exits", "age_at_first_funding": "Age at first funding",
    "first_funding_year": "First funding year", "months_to_second": "Months to second round",
    "is_USA": "U.S.-based", "sec_biotech": "Biotech sector", "has_venture": "Venture round",
    "kb_score": "Knowledge score s_K"}


def beeswarm(ds, S, title, path):
    K = add_knowledge(ds, ds.X, theory_signs(ds))[S.columns]
    top = np.abs(S).mean().sort_values(ascending=False).head(12).index[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
    cmap = LinearSegmentedColormap.from_list("bo", [ACCENT, "#d9d8d3", WARM])
    cmap.set_bad("#bdbcb6")
    rules = {r.id: f"{r.id} {r.name}" for r in ds.rules}
    rng = np.random.default_rng(SEED)
    for i, f in enumerate(top):
        v = K[f].to_numpy()
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        col = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        ax.scatter(S[f], i + rng.uniform(-0.28, 0.28, len(v)), c=col, cmap=cmap, s=2.5, lw=0, alpha=0.8)
    ax.axvline(0, color=EDGE, lw=0.6)
    ax.set_yticks(range(len(top)), [LABEL.get(f, rules.get(f, f)) for f in top], fontsize=6.5)
    ax.set_xlabel("SHAP value (log-odds of success)")
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================================== #
# 8. Main                                                                      #
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="1 repeat, 2 draws: smoke test")
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.draws = 1, 2
    os.makedirs(OUT, exist_ok=True)

    ds = {"S1": study1(), "S2": study2()}
    for d in ds.values():
        print(f"{d.name}: n={len(d.y)}, features={d.X.shape[1]}, success={d.y.mean():.3f}")

    main_res, kept, oof = {}, {}, {}
    for k, d in ds.items():
        print(f"[E1] {k}: {args.repeats}x5 CV with nested tuning")
        main_res[k], kept[k], oof[k] = e1_main(d, args.repeats, args.jobs)

    print("[E1] S2 time split")
    d2 = ds["S2"]
    tr, te = np.where(d2.year <= TRAIN_UNTIL)[0], np.where(d2.year > TRAIN_UNTIL)[0]
    fits = Parallel(n_jobs=args.jobs)(delayed(Fitted)(d2, sp, d2.X.iloc[tr], d2.y[tr]) for sp in SPECS)
    yte, Xte = d2.y[te], d2.X.iloc[te]
    preds = {sp.name: m.proba(Xte) for sp, m in zip(SPECS, fits)}
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(te), len(te)) for _ in range(2000)]
    ts = {}
    for sp, m in zip(SPECS, fits):
        rec = evaluate(d2, m, Xte, yte)
        if sp.name != PROPOSED:
            diffs = np.array([roc_auc_score(yte[i], preds[PROPOSED][i]) - roc_auc_score(yte[i], preds[sp.name][i])
                              for i in boots])
            # two-sided paired bootstrap p; identical predictions give p = 1
            rec["p"] = 1.0 if np.all(diffs == 0) else float(min(1.0, 2 * min((diffs <= 0).mean(),
                                                                              (diffs >= 0).mean())))
            rec["ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
            rec["mark"] = mark(rec["p"])
        rec["hp"] = m.hp
        ts[sp.name] = rec

    print("[E2] wrong-knowledge stress test")
    stress, confirmed = [], {}
    for k, d in ds.items():
        s, confirmed[k] = e2_stress(d, args.jobs, draws=args.draws)
        stress.append(s)
    stress = pd.concat(stress, ignore_index=True)
    print("[E3] threshold sensitivity")
    sens = pd.concat([e3_sensitivity(d, args.jobs) for d in ds.values()], ignore_index=True)
    sens.to_csv(f"{OUT}/tableS2_threshold_sensitivity.csv", index=False)

    print("[E1b] S1 without lifetime-accumulation features (leakage check)")
    leak = ["age_last_milestone_year", "age_last_funding_year", "funding_span"]
    d1 = ds["S1"]
    v1r = {v: [f for f in fs if f not in leak] for v, fs in d1.views.items()}
    d1r = Dataset("S1r", "S1 reduced", d1.X[[f for fs in v1r.values() for f in fs]], d1.y, d1.year, v1r,
                  d1.rules, d1.cont, d1.couple)
    folds = cv_runs(d1r, [s for s in SPECS if s.name in ("LightGBM", PROPOSED)], args.repeats, args.jobs)
    s1r = {n: float(np.mean([f[n]["auc"] for f in folds])) for n in ("LightGBM", PROPOSED)}

    T1 = table1(main_res, ts, OUT)
    T2 = table2(stress, OUT)
    TS1 = pd.concat([table_s1(ds[k], kept[k], oof[k], OUT) for k in ds], ignore_index=True)
    TS1.to_csv(f"{OUT}/tableS1_rule_validation.csv", index=False)
    with open(f"{OUT}/tableS1_rule_validation.md", "w") as fh:
        fh.write(TS1.to_markdown(index=False))
    figure1(len(ds["S1"].y), len(ds["S2"].y), OUT)
    vi2 = figure2(ds["S2"], oof["S2"], stress, OUT)
    beeswarm(ds["S2"], oof["S2"], "S2: top-12 inputs, out-of-fold TreeSHAP", f"{OUT}/figS1_shap_s2.png")
    beeswarm(ds["S1"], oof["S1"], "S1: top-12 inputs, out-of-fold TreeSHAP", f"{OUT}/figS2_shap_s1.png")

    js = {
        "data": {k: {"n": int(len(d.y)), "features": int(d.X.shape[1]), "success_rate": float(d.y.mean())}
                 for k, d in ds.items()},
        "s2_time_split": {"train_n": int(len(tr)), "test_n": int(len(te)),
                          "train_cohorts": f"<= {TRAIN_UNTIL}", "test_cohorts": f"> {TRAIN_UNTIL}"},
        "cv": f"5-fold x {args.repeats} repeats, nested 3-fold tuning",
        "main": {k: {n: {m: [float(np.mean(v[m])), float(np.std(v[m], ddof=1))]
                         for m in ("auc", "brier", "p10", "f1", "acc", "dvr", "dvr1")}
                     | {"p_auc": v["p_auc"], "stability": v["stability"]}
                     for n, v in r.items()} for k, r in main_res.items()},
        "time_split": {n: {m: v.get(m) for m in ("auc", "brier", "p10", "dvr", "dvr1", "p", "ci", "hp")}
                       for n, v in ts.items()},
        "stress_confirmed_rules": confirmed,
        "sensitivity": sens.to_dict(orient="records"),
        "s1_reduced_auc": s1r,
        "view_importance_s2": vi2.round(4).to_dict(),
        "view_importance_s1": view_importance(ds["S1"], oof["S1"]).round(4).to_dict(),
        "top_features": {k: np.abs(oof[k]).mean().sort_values(ascending=False).head(12).round(4).to_dict()
                         for k in ds},
        "refined_retention": {k: {c: float((kept[k][c] != 0).mean()) for c in kept[k].columns} for k in ds},
    }
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(js, fh, indent=2, default=float)
    pd.set_option("display.width", 250)
    print(T1.to_string(index=False))
    print(T2.to_string(index=False))
    print(sens.to_string(index=False))
    print("S1 reduced AUC:", s1r)


if __name__ == "__main__":
    main()
