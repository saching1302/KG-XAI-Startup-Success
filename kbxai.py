"""
KB-XAI: an explainable, knowledge-based model for predicting startup success
from multi-source entrepreneurial data.

One script reproduces every number, table and figure in the paper:

    python kbxai.py --data data --out results

Pipeline
  1. Multi-source data   companies + funding rounds + investments + acquisitions
                         (Crunchbase open export, Dec 2015, CC BY-NC 4.0).
  2. Leakage-safe design only information from the first W months after a
                         startup's first funding is used; startups that exit
                         inside the window are removed; train/test split is
                         temporal (cohorts 2000-08 -> 2009-12).
  3. Knowledge base      entrepreneurial-finance theory encoded as rules
                         (feature, expected direction, construct, source).
  4. Knowledge injection rules become monotone constraints of a LightGBM model.
  5. Explanation         exact TreeSHAP attributions (Lundberg et al., 2020).
  6. Knowledge audit     monotonicity-violation rate (MVR) and knowledge-alignment
                         score (KAS) test whether explanations agree with theory.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_URL = "https://raw.githubusercontent.com/notpeter/crunchbase-data/master/{}.csv"
TABLES = ("companies", "rounds", "investments", "acquisitions")
TRAIN_COHORTS, TEST_COHORTS = (2000, 2008), (2009, 2012)
HUBS = {"SF Bay Area", "New York City", "Boston", "Los Angeles", "Seattle",
        "London", "Austin", "Tel Aviv"}
SECTORS = {  # keyword groups over Crunchbase category lists
    "sec_biohealth": ("biotech", "health", "medic", "pharma", "life science"),
    "sec_software": ("software", "saas", "analytics", "cloud", "security", "data"),
    "sec_internet": ("e-commerce", "web", "social", "advertising", "search",
                     "mobile", "apps", "games", "media"),
    "sec_hardware": ("hardware", "semiconductor", "electronics", "devices"),
    "sec_cleantech": ("clean", "energy", "solar", "environment"),
    "sec_finance": ("financ", "fintech", "payments", "banking", "insurance"),
}

# ---------------------------------------------------------------- knowledge base
# Each rule: feature it governs, expected direction (+1 = success probability
# must not fall as the feature rises), theoretical construct and source.
KNOWLEDGE_BASE = [
    dict(id="R1", feature="investor_exits_pct", direction=1,
         construct="Investor track record (certification)",
         source="Hsu (2004); Nahata (2008)"),
    dict(id="R2", feature="investor_experience_pct", direction=1,
         construct="Investor experience and networks",
         source="Hochberg et al. (2007)"),
    dict(id="R3", feature="n_investors", direction=1,
         construct="Syndication",
         source="Lerner (1994)"),
    dict(id="R4", feature="series_stage", direction=1,
         construct="Staged-financing milestones",
         source="Gompers (1995)"),
    dict(id="R5", feature="log_total_raised", direction=1,
         construct="Financial resource endowment",
         source="Cooper et al. (1994)"),
    dict(id="R6", feature="has_venture", direction=1,
         construct="Venture-capital certification",
         source="Megginson & Weiss (1991)"),
    dict(id="R7", feature="in_hub", direction=1,
         construct="Entrepreneurial-cluster location",
         source="Chen et al. (2010)"),
]
RULES = {r["feature"]: r for r in KNOWLEDGE_BASE}

COMPANY_FEATS = ["age_at_first_funding", "in_hub", "is_usa", "n_sectors",
                 *SECTORS]
ROUND_FEATS = ["n_rounds", "log_total_raised", "log_max_round", "series_stage",
               "months_to_round2", "has_seed", "has_angel", "has_venture",
               "has_debt", "has_grant", "has_crowd", "has_convertible",
               "has_private_equity"]
INVESTOR_FEATS = ["n_investors", "investor_experience_pct",
                  "investor_exits_pct"]
ALL_FEATS = COMPANY_FEATS + ROUND_FEATS + INVESTOR_FEATS
SINGLE_SOURCE_FEATS = COMPANY_FEATS + ROUND_FEATS

LABELS = {  # short labels used in figures
    "investor_exits_pct": "Investor prior exits (pctl)",
    "investor_experience_pct": "Investor prior deals (pctl)",
    "n_investors": "Number of investors",
    "series_stage": "Series stage reached",
    "log_total_raised": "Total raised (log USD)",
    "log_max_round": "Largest round (log USD)",
    "has_venture": "Venture round",
    "in_hub": "Located in hub",
    "n_rounds": "Number of rounds",
    "age_at_first_funding": "Age at first funding (yrs)",
    "months_to_round2": "Months to 2nd round",
    "has_seed": "Seed round", "has_angel": "Angel round",
    "has_debt": "Debt financing", "has_grant": "Grant",
    "has_crowd": "Crowdfunding", "has_convertible": "Convertible note",
    "has_private_equity": "Private equity", "is_usa": "USA",
    "n_sectors": "Number of sector tags",
    "sec_biohealth": "Bio/health sector", "sec_software": "Software sector",
    "sec_internet": "Internet/media sector", "sec_hardware": "Hardware sector",
    "sec_cleantech": "Cleantech sector", "sec_finance": "Finance sector",
}


# ---------------------------------------------------------------- data
def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the four Crunchbase tables, downloading them on first use."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in TABLES:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            print(f"downloading {name}.csv ...")
            urllib.request.urlretrieve(DATA_URL.format(name), path)
        tables[name] = pd.read_csv(path, low_memory=False)
    return tables


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def investor_track_record(inv: pd.DataFrame, acq: pd.DataFrame) -> pd.DataFrame:
    """For every investment, the investor's prior deals and prior portfolio
    exits strictly before the deal date (no look-ahead), as within-year
    percentiles."""
    inv = inv.dropna(subset=["investor_permalink", "funded_at"]).copy()
    inv["prior_deals"] = (inv.groupby("investor_permalink")["funded_at"]
                          .rank(method="min") - 1)
    first_inv = (inv.groupby(["investor_permalink", "company_permalink"])
                 ["funded_at"].min().rename("first_inv").reset_index())
    exits = first_inv.merge(acq, on="company_permalink")
    exits = exits[exits.acquired_at > exits.first_inv]
    exits = exits.sort_values("acquired_at")
    exits["cum_exits"] = exits.groupby("investor_permalink").cumcount() + 1
    deals = inv.sort_values("funded_at")
    deals = pd.merge_asof(
        deals, exits[["investor_permalink", "acquired_at", "cum_exits"]],
        left_on="funded_at", right_on="acquired_at", by="investor_permalink",
        allow_exact_matches=False)
    deals["prior_exits"] = deals["cum_exits"].fillna(0)
    # The export starts in the 2000s, so raw counts drift upward over time
    # (left-censored history). Rank each deal against all deals of its year.
    year = deals.funded_at.dt.year
    for col in ("prior_deals", "prior_exits"):
        deals[col] = deals.groupby(year)[col].rank(pct=True)
    return deals


def build_dataset(t: dict[str, pd.DataFrame], window: int = 24) -> pd.DataFrame:
    """One row per startup with features observed in the first `window`
    months after first funding and the eventual outcome as label."""
    comp, rnd = t["companies"].copy(), t["rounds"].copy()
    inv, acq = t["investments"].copy(), t["acquisitions"].copy()
    rnd["funded_at"], inv["funded_at"] = _dt(rnd.funded_at), _dt(inv.funded_at)
    acq["acquired_at"] = _dt(acq.acquired_at)
    acq = (acq.dropna(subset=["acquired_at"])
           .groupby("company_permalink", as_index=False)["acquired_at"].min())

    comp = comp[comp.status.isin(["acquired", "ipo", "closed"])]
    comp = comp.rename(columns={"permalink": "company_permalink"})
    rnd = rnd.dropna(subset=["funded_at"])
    t0 = rnd.groupby("company_permalink")["funded_at"].min().rename("t0")
    df = comp.merge(t0, on="company_permalink")
    df = df[df.t0.dt.year.between(TRAIN_COHORTS[0], TEST_COHORTS[1])]
    df["t1"] = df.t0 + pd.DateOffset(months=window)
    df["y"] = (df.status != "closed").astype(int)

    # leakage control: drop startups whose exit is visible inside the window
    df = df.merge(acq, on="company_permalink", how="left")
    df = df[~(df.acquired_at < df.t1)]
    in_win = lambda x: x[(x.funded_at >= x.t0) & (x.funded_at < x.t1)]
    r = in_win(rnd.merge(df[["company_permalink", "t0", "t1"]],
                         on="company_permalink"))
    post_ipo = r[r.funding_round_type.str.startswith("post_ipo", na=False)]
    df = df[~df.company_permalink.isin(post_ipo.company_permalink)]
    r = r[~r.funding_round_type.isin(
        ["post_ipo_equity", "post_ipo_debt", "secondary_market"])]

    # company source
    df["age_at_first_funding"] = ((df.t0 - _dt(df.founded_at)).dt.days / 365.25
                                  ).where(lambda a: a.between(-1, 50))
    df["in_hub"] = df.region.isin(HUBS).astype(int)
    df["is_usa"] = (df.country_code == "USA").astype(int)
    cats = df.category_list.fillna("").str.lower()
    df["n_sectors"] = cats.str.count(r"\|") + (cats != "").astype(int)
    for name, keys in SECTORS.items():
        df[name] = cats.apply(lambda c: int(any(k in c for k in keys)))

    # funding-round source
    r = r.assign(amt=r.raised_amount_usd,  # NaN = amount undisclosed
                 stage=r.funding_round_code.map({"A": 1, "B": 2}).fillna(
                     r.funding_round_code.notna() * 3))
    g = r.groupby("company_permalink")
    types = r.funding_round_type
    flags = {"has_seed": types == "seed", "has_angel": types == "angel",
             "has_venture": types == "venture",
             "has_debt": types == "debt_financing", "has_grant": types == "grant",
             "has_crowd": types.str.contains("crowdfunding", na=False),
             "has_convertible": types == "convertible_note",
             "has_private_equity": types == "private_equity"}
    second = g.funded_at.apply(lambda s: np.sort(s.values)[1]
                               if s.nunique() > 1 else pd.NaT)
    feats = pd.DataFrame({
        "n_rounds": g.size(),
        # undisclosed amounts stay missing rather than being read as $0
        "log_total_raised": np.log1p(g.amt.sum(min_count=1)),
        "log_max_round": np.log1p(g.amt.max()),
        "series_stage": g.stage.max(),
        "_second": pd.to_datetime(second),
        **{k: v.groupby(r.company_permalink).max().astype(int)
           for k, v in flags.items()}})
    df = df.merge(feats, left_on="company_permalink", right_index=True)
    df["months_to_round2"] = (df._second - df.t0).dt.days / 30.44

    # investor-network source (track record measured before each deal)
    deals = investor_track_record(inv, acq)
    d = in_win(deals.merge(df[["company_permalink", "t0", "t1"]],
                           on="company_permalink"))
    gi = d.groupby("company_permalink")
    inv_feats = pd.DataFrame({
        "n_investors": gi.investor_permalink.nunique(),
        "investor_experience_pct": gi.prior_deals.max(),
        "investor_exits_pct": gi.prior_exits.max()})
    df = df.merge(inv_feats, left_on="company_permalink", right_index=True,
                  how="left")
    df[INVESTOR_FEATS] = df[INVESTOR_FEATS].fillna(0)
    df["cohort"] = df.t0.dt.year
    return df[["company_permalink", "name", "cohort", "status", "y", *ALL_FEATS]
              ].reset_index(drop=True)


# ---------------------------------------------------------------- models
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                  verbose=-1)


def make_models() -> dict[str, tuple[object, list[str]]]:
    """Name -> (estimator, feature list). The last entry is KB-XAI."""
    mono = [RULES[f]["direction"] if f in RULES else 0 for f in ALL_FEATS]
    return {
        "Logistic regression": (make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(), LogisticRegression(max_iter=5000)), ALL_FEATS),
        "Random forest": (make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   n_jobs=-1, random_state=SEED)), ALL_FEATS),
        "LightGBM (single-source)": (lgb.LGBMClassifier(**LGB_PARAMS),
                                     SINGLE_SOURCE_FEATS),
        "LightGBM (multi-source)": (lgb.LGBMClassifier(**LGB_PARAMS), ALL_FEATS),
        "KB-XAI (proposed)": (lgb.LGBMClassifier(
            **LGB_PARAMS, monotone_constraints=mono,
            monotone_constraints_method="intermediate"), ALL_FEATS),
    }


def bootstrap_auc(y, p, q=None, n=1000, seed=SEED):
    """95% CI of AUC(p); if q given, CI of the paired difference AUC(p)-AUC(q)."""
    rng, y = np.random.default_rng(seed), np.asarray(y)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a = roc_auc_score(y[i], p[i])
        stats.append(a - roc_auc_score(y[i], q[i]) if q is not None else a)
    return np.percentile(stats, [2.5, 97.5])


def monotonicity_violations(model, X: pd.DataFrame, X_train: pd.DataFrame):
    """Share of test startups whose predicted success probability FALLS when
    a knowledge-governed feature is improved (binary 0->1, else +1 IQR)."""
    base = model.predict_proba(X)[:, 1]
    out = {}
    for f, rule in RULES.items():
        if f not in X:
            continue
        Xp = X.copy()
        binary = set(X_train[f].dropna().unique()) <= {0, 1}
        if binary:
            mask = X[f].values == 0
            Xp.loc[mask, f] = 1
        else:
            q75, q25 = np.nanpercentile(X_train[f], [75, 25])
            step = (q75 - q25) or np.nanstd(X_train[f])
            mask = X[f].notna().values
            Xp[f] = X[f] + rule["direction"] * step
        delta = (model.predict_proba(Xp)[:, 1] - base) * rule["direction"]
        out[f] = (float((delta[mask] < -1e-9).mean()), int(mask.sum()))
    return out


def tree_shap(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP values (log-odds scale) from LightGBM's native routine."""
    phi = model.predict(X, pred_contrib=True)
    return pd.DataFrame(phi[:, :-1], columns=X.columns, index=X.index)


def knowledge_alignment(X: pd.DataFrame, phi: pd.DataFrame) -> dict[str, float]:
    """KAS: direction-signed Spearman correlation between a feature and its
    SHAP value; +1 = explanations fully agree with the knowledge rule."""
    return {f: float(r["direction"] * spearmanr(X[f], phi[f],
                                                nan_policy="omit").statistic)
            for f, r in RULES.items() if X[f].nunique() > 1}


def explain_case(model, x: pd.DataFrame, k: int = 5) -> str:
    """Rule-grounded natural-language explanation for one startup."""
    p = model.predict_proba(x)[0, 1]
    phi = tree_shap(model, x).iloc[0]
    lines = [f"Predicted success probability: {p:.2f}"]
    for f in phi.abs().sort_values(ascending=False).index[:k]:
        sign = "raises" if phi[f] > 0 else "lowers"
        rule = RULES.get(f)
        why = (f" [{rule['id']}: {rule['construct']}; {rule['source']}]"
               if rule else " [data-driven; no knowledge rule]")
        lines.append(f"  {LABELS[f]} = {x[f].iloc[0]:.2f} {sign} the odds "
                     f"({phi[f]:+.2f} log-odds){why}")
    return "\n".join(lines)


# ---------------------------------------------------------------- figures
COL = {"ink": "#1f2937", "muted": "#6b7280", "kb": "#0f766e",
       "base": "#b45309", "box": "#f3f4f6", "accent": "#e0f2f1"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def figure_framework(path: Path, n: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.set_xlim(0, 104), ax.set_ylim(0, 46), ax.axis("off")
    W, H, X = 22.5, 19, (1, 27.5, 54, 80.5)

    def box(x, y, title, body, kb=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), W, H, boxstyle="round,pad=0.3,rounding_size=1.2",
            fc=COL["accent"] if kb else COL["box"],
            ec=COL["kb"] if kb else COL["muted"], lw=0.9))
        ax.text(x + W / 2, y + H - 2.0, title, ha="center", va="top",
                fontsize=7.1, weight="bold", color=COL["ink"])
        ax.text(x + W / 2, y + H - 6.2, body, ha="center", va="top",
                fontsize=6.3, color=COL["ink"], linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9,
                                     color=COL["muted"]))

    top, bot = 25, 1
    box(X[0], top, "Multi-source data",
        f"Companies\nFunding rounds\nInvestments ({n['investments']:,})\n"
        f"Acquisitions ({n['acquisitions']:,})")
    box(X[1], top, "Leakage-safe features",
        f"First {n['window']} months after\nfirst funding only\n"
        f"{len(ALL_FEATS)} features, 3 sources\ntemporal train/test split")
    box(X[2], top, "Knowledge injection",
        f"{len(KNOWLEDGE_BASE)} theory rules \u2192\nmonotone constraints\n"
        "on gradient-boosted\ntrees (LightGBM)", kb=True)
    box(X[3], top, "Explanation",
        "Exact TreeSHAP\nattributions\n(global and local)", kb=True)
    box(X[1], bot, "Knowledge base",
        "Signalling, syndication,\nstaging, resources,\nclusters (R1\u2013R7)",
        kb=True)
    box(X[2], bot, "Knowledge audit",
        "Monotonicity-violation\nrate (MVR)\nKnowledge-alignment\nscore (KAS)",
        kb=True)
    box(X[3], bot, "Decision support",
        "Success probability +\nrule-grounded rationale\nfor investors, founders")
    for i in range(3):
        arrow(X[i] + W + 0.6, top + H / 2, X[i + 1] - 0.6, top + H / 2)
    for i in (1, 2):
        arrow(X[i] + W + 0.6, bot + H / 2, X[i + 1] - 0.6, bot + H / 2)
    arrow(X[1] + W / 2, bot + H + 0.5, X[2] + 4, top - 0.5)   # rules -> model
    arrow(X[3] + W / 2, top - 0.5, X[2] + W - 4, bot + H + 0.5)  # SHAP -> audit
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_explanations(path: Path, X, phi_kb, phi_base, focus: str) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    top = phi_kb.abs().mean().sort_values().index[-10:]
    rng = np.random.default_rng(SEED)
    for i, f in enumerate(top):
        v = X[f].rank(pct=True).fillna(0.5).values
        a.scatter(phi_kb[f], i + rng.uniform(-0.28, 0.28, len(v)), c=v,
                  cmap="viridis", s=2.5, alpha=0.55, lw=0, rasterized=True)
    a.axvline(0, color=COL["muted"], lw=0.6)
    a.set_yticks(range(len(top)))
    a.set_yticklabels([("* " if f in RULES else "") + LABELS[f] for f in top],
                      fontsize=7.2)
    a.set_xlabel("SHAP value (log-odds of success)")
    a.set_title("(a) KB-XAI global explanation", fontsize=9, loc="left")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    cb = fig.colorbar(sm, ax=a, fraction=0.04, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=7)
    cb.set_label("Feature value", fontsize=7)

    x = X[focus]
    jit = rng.uniform(-0.06, 0.06, len(x)) if x.nunique() < 12 else 0
    b.scatter(x + jit, phi_base[focus], s=3, alpha=0.35, color=COL["base"],
              lw=0, label="LightGBM (multi-source)", rasterized=True)
    b.scatter(x + jit, phi_kb[focus], s=3, alpha=0.35, color=COL["kb"], lw=0,
              label="KB-XAI (proposed)", rasterized=True)
    b.axhline(0, color=COL["muted"], lw=0.6)
    b.set_xlabel(LABELS[focus])
    b.set_ylabel("SHAP value")
    b.set_title(f"(b) {RULES[focus]['id']}: {RULES[focus]['construct']}",
                fontsize=9, loc="left")
    b.legend(frameon=False, fontsize=7, markerscale=3, loc="upper left")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- experiment
def run(data_dir: Path, out: Path, window: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tables = load_tables(data_dir)
    df = build_dataset(tables, window)
    tr = df[df.cohort.between(*TRAIN_COHORTS)]
    te = df[df.cohort.between(*TEST_COHORTS)]
    summary = {"n_total": len(df), "n_train": len(tr), "n_test": len(te),
               "success_rate_train": round(tr.y.mean(), 4),
               "success_rate_test": round(te.y.mean(), 4),
               "status_counts": df.status.value_counts().to_dict(),
               "window_months": window, "n_features": len(ALL_FEATS),
               "source_rows": {k: len(v) for k, v in tables.items()}}
    print(json.dumps(summary, indent=2))

    rows, fitted, preds, mvr_all = [], {}, {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, (model, feats) in make_models().items():
        cv_auc = []
        for a, b in cv.split(tr[feats], tr.y):
            m = make_models()[name][0].fit(tr[feats].iloc[a], tr.y.iloc[a])
            cv_auc.append(roc_auc_score(tr.y.iloc[b],
                                        m.predict_proba(tr[feats].iloc[b])[:, 1]))
        model.fit(tr[feats], tr.y)
        p = model.predict_proba(te[feats])[:, 1]
        lo, hi = bootstrap_auc(te.y.values, p)
        mvr = monotonicity_violations(model, te[feats], tr[feats])
        n_tests = sum(n for _, n in mvr.values())
        pooled = sum(v * n for v, n in mvr.values()) / n_tests
        fitted[name], preds[name], mvr_all[name] = model, p, mvr
        rows.append({"Model": name, "Sources": "3" if feats is ALL_FEATS else "2",
                     "CV AUC (train)": f"{np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}",
                     "Test AUC": round(roc_auc_score(te.y, p), 3),
                     "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                     "PR-AUC": round(average_precision_score(te.y, p), 3),
                     "F1": round(f1_score(te.y, p >= 0.5), 3),
                     "Balanced acc.": round(balanced_accuracy_score(te.y, p >= 0.5), 3),
                     "Brier": round(brier_score_loss(te.y, p), 3),
                     "MVR (%)": round(100 * pooled, 2)})
        print(rows[-1])
    table1 = pd.DataFrame(rows)
    table1.to_csv(out / "table1_performance.csv", index=False)

    kb, base = fitted["KB-XAI (proposed)"], fitted["LightGBM (multi-source)"]
    d_lo, d_hi = bootstrap_auc(te.y.values, preds["KB-XAI (proposed)"],
                               preds["LightGBM (multi-source)"])
    d_lo2, d_hi2 = bootstrap_auc(te.y.values, preds["LightGBM (multi-source)"],
                                 preds["LightGBM (single-source)"])
    Xte = te[ALL_FEATS]
    phi_kb, phi_base = tree_shap(kb, Xte), tree_shap(base, Xte)
    kas_kb, kas_base = knowledge_alignment(Xte, phi_kb), knowledge_alignment(Xte, phi_base)
    # rule ablation: AUC gained on the test cohort by dropping one constraint
    auc_kb = roc_auc_score(te.y, preds["KB-XAI (proposed)"])
    rule_cost = {}
    for f in RULES:
        mono = [RULES[g]["direction"] if g in RULES and g != f else 0
                for g in ALL_FEATS]
        m = lgb.LGBMClassifier(**LGB_PARAMS, monotone_constraints=mono,
                               monotone_constraints_method="intermediate")
        m.fit(tr[ALL_FEATS], tr.y)
        rule_cost[f] = roc_auc_score(te.y, m.predict_proba(Xte)[:, 1]) - auc_kb
    imp = phi_kb.abs().mean()
    rank = imp.rank(ascending=False).astype(int)
    table2 = pd.DataFrame([{
        "Rule": r["id"], "Construct": r["construct"], "Source": r["source"],
        "Feature": LABELS[r["feature"]], "Expected": "+" if r["direction"] > 0 else "-",
        "Mean |SHAP| (KB-XAI)": round(imp[r["feature"]], 3),
        "Rank (of %d)" % len(ALL_FEATS): rank[r["feature"]],
        "MVR % (LightGBM)": round(100 * mvr_all["LightGBM (multi-source)"][r["feature"]][0], 2),
        "MVR % (KB-XAI)": round(100 * mvr_all["KB-XAI (proposed)"][r["feature"]][0], 2),
        "KAS (LightGBM)": round(kas_base.get(r["feature"], np.nan), 3),
        "KAS (KB-XAI)": round(kas_kb.get(r["feature"], np.nan), 3),
        "AUC cost of rule": round(rule_cost[r["feature"]], 4),
    } for r in KNOWLEDGE_BASE])
    table2.to_csv(out / "table2_knowledge_audit.csv", index=False)
    print(table2.to_string())

    focus = "log_total_raised"  # most important knowledge-governed feature
    figure_framework(out / "fig1_framework.png",
                     {"window": window, "investments": len(tables["investments"]),
                      "acquisitions": len(tables["acquisitions"])})
    figure_explanations(out / "fig2_explanations.png", Xte, phi_kb, phi_base, focus)

    # one rule-grounded local explanation (the most confident correct success)
    p_kb = preds["KB-XAI (proposed)"]
    idx = int(np.argmax(np.where(te.y.values == 1, p_kb, -1)))
    case = (f"Startup: {te.name.iloc[idx]} (cohort {te.cohort.iloc[idx]}, "
            f"outcome: {te.status.iloc[idx]})\n" + explain_case(kb, Xte.iloc[[idx]]))
    (out / "case_explanation.txt").write_text(case + "\n")
    print(case)

    summary.update({
        "auc_diff_kbxai_minus_lgbm_multi": round(roc_auc_score(te.y, p_kb) - roc_auc_score(te.y, preds["LightGBM (multi-source)"]), 4),
        "auc_diff_kbxai_minus_lgbm_multi_95ci": [round(d_lo, 4), round(d_hi, 4)],
        "auc_diff_multi_minus_single_source": round(roc_auc_score(te.y, preds["LightGBM (multi-source)"]) - roc_auc_score(te.y, preds["LightGBM (single-source)"]), 4),
        "auc_diff_multi_minus_single_source_95ci": [round(d_lo2, 4), round(d_hi2, 4)],
        "mean_kas_kbxai": round(np.mean(list(kas_kb.values())), 4),
        "mean_kas_lgbm_multi": round(np.mean(list(kas_base.values())), 4),
        "fig2b_focus_feature": focus,
        "top10_features_kbxai": [LABELS[f] for f in imp.sort_values(ascending=False).index[:10]],
        "lr_coefficients_rule_features": {
            f: round(float(c), 4) for f, c in zip(
                ALL_FEATS, fitted["Logistic regression"][-1].coef_[0][:len(ALL_FEATS)])
            if f in RULES},
    })
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--window", type=int, default=24,
                    help="observation window in months after first funding")
    args = ap.parse_args()
    run(args.data, args.out, args.window)
