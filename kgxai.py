"""
KG-XAI: a knowledge-guided, explainable model for predicting startup success
from multi-source entrepreneurial data.

One file, end to end:
  1. load the Crunchbase-derived startup dataset and build six source views
  2. encode a literature-grounded knowledge base (9 rules) as rule activations
  3. train a knowledge-guided gradient-boosted model (rule features + monotone
     constraints that follow the direction each rule/theory predicts)
  4. explain every prediction with exact TreeSHAP, aggregated to source views
     and to knowledge rules
  5. evaluate against data-only and knowledge-only baselines with 5x5 repeated
     stratified CV, a corrected resampled t-test, and two explanation-quality
     metrics (directional violation rate, attribution stability)
  6. write Table 1, Table 2, Figure 1, Figure 2 and results.json to ./results

Usage:  python kgxai.py            (downloads the data on first run)
        python kgxai.py --data path/to/startup_data.csv --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import warnings
from itertools import combinations

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
DATA_URL = ("https://raw.githubusercontent.com/sumitjhaleriya/"
            "Startup-Success-Prediction-using-Machine-Learning/main/startup%20data.csv")
OUT = "results"

# --------------------------------------------------------------------------- #
# 1. Multi-source views                                                        #
# --------------------------------------------------------------------------- #
VIEWS = {
    "Funding":   ["log_funding", "funding_rounds", "has_VC", "has_angel", "has_roundA",
                  "has_roundB", "has_roundC", "has_roundD"],
    "Network":   ["relationships", "avg_participants", "is_top500"],
    "Milestone": ["milestones", "age_first_milestone_year", "age_last_milestone_year"],
    "Temporal":  ["age_first_funding_year", "age_last_funding_year", "funding_span", "founded_year"],
    "Ecosystem": ["is_CA", "is_NY", "is_MA", "is_TX", "is_otherstate"],
    "Sector":    ["is_software", "is_web", "is_mobile", "is_enterprise", "is_advertising",
                  "is_gamesvideo", "is_ecommerce", "is_biotech", "is_othercategory"],
}
BASE = [f for v in VIEWS.values() for f in v]
VIEW_OF = {f: v for v, fs in VIEWS.items() for f in fs}

# Theory-signed raw features: entrepreneurship theory predicts more => more success.
SIGNED = {f: +1 for f in ["log_funding", "funding_rounds", "has_VC", "has_roundA", "has_roundB",
                          "has_roundC", "has_roundD", "avg_participants", "relationships",
                          "milestones"]}

# --------------------------------------------------------------------------- #
# 2. Knowledge base: expert rules from entrepreneurship / venture finance      #
# --------------------------------------------------------------------------- #
KB = [  # id, name, view, theory, direction, condition
    ("R1", "VC certification", "Funding", "Signalling / VC certification", +1,
     lambda d: d.has_VC == 1),
    ("R2", "Staged-financing progression", "Funding", "Staged VC financing", +1,
     lambda d: (d.has_roundB + d.has_roundC + d.has_roundD) > 0),
    ("R3", "Resource munificence", "Funding", "Resource-based view", +1,
     lambda d: d.log_funding >= np.log1p(10e6)),
    ("R4", "Investor syndication", "Network", "VC syndication", +1,
     lambda d: d.avg_participants >= 3),
    ("R5", "Network embeddedness", "Network", "Inter-organisational endorsement", +1,
     lambda d: d.relationships >= 5),
    ("R6", "Legitimacy milestones", "Milestone", "Legitimacy-based growth", +1,
     lambda d: d.milestones >= 2),
    ("R7", "Ecosystem hub", "Ecosystem", "Clusters and entrepreneurship", +1,
     lambda d: (d.is_CA + d.is_MA + d.is_NY) > 0),
    ("R8", "Funding stagnation", "Temporal", "Liability of newness", -1,
     lambda d: (d.funding_rounds == 1) & (d.milestones == 0)),
    ("R9", "Angel-only financing", "Funding", "Weak certification", -1,
     lambda d: (d.has_angel == 1) & (d.has_VC == 0) & (d.has_roundA == 0)),
]
RULES = [r[0] for r in KB]


def load(path: str | None) -> pd.DataFrame:
    path = path or os.path.join("data", "startup_data.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading data -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def build_views(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw Crunchbase export -> six source views. Leakage columns (closed_at,
    labels, ids, names) are never used."""
    d = raw.drop_duplicates("id").reset_index(drop=True)
    X = pd.DataFrame(index=d.index)
    X["log_funding"] = np.log1p(d.funding_total_usd)
    for c in ["funding_rounds", "has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC",
              "has_roundD", "relationships", "avg_participants", "is_top500", "milestones",
              "age_first_funding_year", "age_last_funding_year", "is_CA", "is_NY", "is_MA",
              "is_TX", "is_otherstate", "is_software", "is_web", "is_mobile", "is_enterprise",
              "is_advertising", "is_gamesvideo", "is_ecommerce", "is_biotech", "is_othercategory"]:
        X[c] = d[c].astype(float)
    X["age_first_milestone_year"] = d.age_first_milestone_year.fillna(0.0)
    X["age_last_milestone_year"] = d.age_last_milestone_year.fillna(0.0)
    X["funding_span"] = d.age_last_funding_year - d.age_first_funding_year
    X["founded_year"] = pd.to_datetime(d.founded_at, format="%m/%d/%Y").dt.year.astype(float)
    y = (d.status == "acquired").astype(int).to_numpy()
    return X[BASE], y


def add_knowledge(X: pd.DataFrame, signs: dict | None = None) -> pd.DataFrame:
    """Append rule activations r_j(x) and the signed knowledge score
    s_K(x) = sum_j d_j r_j(x), with d_j the (possibly refined) rule direction."""
    signs = signs or {r[0]: r[4] for r in KB}
    K = X.copy()
    for rid, _, _, _, _, cond in KB:
        K[rid] = cond(X).astype(float)
    K["kb_score"] = sum(signs[rid] * K[rid] for rid in RULES)
    return K


def refine(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Knowledge validation and refinement (training fold only).
    A rule or theory-signed feature keeps its direction only if the training
    data agree with it (sign of the success-rate gap for rules, sign of the
    Spearman correlation for features); contradicted knowledge is released to
    0 = unconstrained, so wrong expert knowledge cannot bias the learner."""
    K = add_knowledge(X)
    d = {}
    for rid, *_, sign, _ in KB:
        on = K[rid].to_numpy() == 1
        gap = y[on].mean() - y[~on].mean() if 0 < on.sum() < len(y) else 0.0
        d[rid] = sign if np.sign(gap) == sign else 0
    for f, sign in SIGNED.items():
        rho = stats.spearmanr(X[f], y)[0]
        d[f] = sign if np.sign(rho) == sign else 0
    return d


# --------------------------------------------------------------------------- #
# 3. Models                                                                    #
# --------------------------------------------------------------------------- #
GBM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=SEED, verbose=-1)


def theory_signs() -> dict:
    return {**SIGNED, **{r[0]: r[4] for r in KB}}


def monotone_vector(cols: list[str], signs: dict) -> list[int]:
    sign = {**signs, "kb_score": +1}
    return [sign.get(c, 0) for c in cols]


class Model:
    """Uniform wrapper: name, feature space, estimator factory."""

    def __init__(self, name, space, make, tree=False, refined=False):
        self.name, self.space, self.make, self.tree, self.refined = name, space, make, tree, refined

    def features(self, X: pd.DataFrame) -> pd.DataFrame:
        K = add_knowledge(X, self.signs)
        return {"base": K[BASE], "rules": K[RULES + ["kb_score"]], "full": K}[self.space]

    def fit(self, X, y):
        self.signs = refine(X, y) if self.refined else theory_signs()
        F = self.features(X)
        self.cols = list(F.columns)
        self.est = self.make(self.cols, self.signs).fit(F, y)
        return self

    def proba(self, X):
        return self.est.predict_proba(self.features(X)[self.cols])[:, 1]

    def shap(self, X):  # exact TreeSHAP (Lundberg et al., 2020) via LightGBM
        return self.est.predict(self.features(X)[self.cols], pred_contrib=True)[:, :-1]


def models() -> list[Model]:
    lr = lambda c, s: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    rf = lambda c, s: RandomForestClassifier(n_estimators=500, min_samples_leaf=3, n_jobs=-1,
                                             random_state=SEED)
    gbm = lambda c, s: lgb.LGBMClassifier(**GBM)
    kg = lambda c, s: lgb.LGBMClassifier(**GBM, monotone_constraints=monotone_vector(c, s),
                                         monotone_constraints_method="advanced")
    return [
        Model("Logistic regression (data only)", "base", lr),
        Model("Random forest (data only)", "base", rf),
        Model("LightGBM (data only)", "base", gbm, tree=True),
        Model("Knowledge base only (rule scorer)", "rules", lr),
        Model("KG-XAI w/o knowledge refinement", "full", kg, tree=True),
        Model("KG-XAI w/o monotone constraints", "full", gbm, tree=True, refined=True),
        Model("KG-XAI w/o rule features", "base", kg, tree=True, refined=True),
        Model("KG-XAI (proposed)", "full", kg, tree=True, refined=True),
    ]


# --------------------------------------------------------------------------- #
# 4. Explanation-quality metrics                                               #
# --------------------------------------------------------------------------- #
def violation_rate(model: Model, X: pd.DataFrame) -> float:
    """Directional violation rate (DVR): share of (startup, signed feature) pairs
    where a theory-consistent improvement (one more round/relationship/milestone,
    a 0->1 flag, +1 sd of log funding or syndicate size) LOWERS predicted success.
    Rule activations are recomputed from the perturbed profile."""
    p0, bad, mat, n = model.proba(X), 0, 0, 0
    for f in SIGNED:
        Xp = X.copy()
        if set(np.unique(X[f])) <= {0.0, 1.0}:
            mask = (X[f] == 0).to_numpy()
            Xp[f] = 1.0
        else:
            mask = np.ones(len(X), bool)
            Xp[f] = X[f] + (X[f].std() if f in ("log_funding", "avg_participants") else 1.0)
        if f in ("has_roundB", "has_roundC", "has_roundD"):
            Xp["funding_rounds"] = np.maximum(Xp.funding_rounds, X.funding_rounds + mask)
        if mask.sum() == 0:
            continue
        p1 = model.proba(Xp)
        bad += int(((p1 - p0) < -1e-6)[mask].sum())
        mat += int(((p1 - p0) < -0.01)[mask].sum())
        n += int(mask.sum())
    return bad / n, mat / n


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


# --------------------------------------------------------------------------- #
# 5. Experiment                                                                #
# --------------------------------------------------------------------------- #
def run(X, y, repeats):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    ms = models()
    rec = {m.name: {"auc": [], "f1": [], "acc": [], "brier": [], "dvr": [], "dvr1": [], "imp": [], "kept": []} for m in ms}
    oof_shap = {}  # proposed model, first repeat: one out-of-fold explanation per startup
    retained = []  # refined knowledge directions of the proposed model, per fold
    for k, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        for m in ms:
            m.fit(Xtr, ytr)
            p = m.proba(Xte)
            r = rec[m.name]
            r["auc"].append(roc_auc_score(yte, p))
            r["f1"].append(f1_score(yte, p >= 0.5))
            r["acc"].append(accuracy_score(yte, p >= 0.5))
            r["brier"].append(brier_score_loss(yte, p))
            v, v1 = violation_rate(m, Xte)
            r["dvr"].append(v), r["dvr1"].append(v1)
            r["kept"].append(sum(m.signs[f] != 0 for f in theory_signs()))
            if m.tree:
                S = m.shap(Xte)
                imp = pd.Series(np.abs(S).mean(0), index=m.cols)
                r["imp"].append(imp)
                if m.name == "KG-XAI (proposed)":
                    retained.append(m.signs)
                if m.name == "KG-XAI (proposed)" and k < 5:
                    oof_shap.update({i: s for i, s in zip(te, S)})
                    oof_cols = m.cols
        print(f"fold {k + 1}/{5 * repeats} done")
    S = np.vstack([oof_shap[i] for i in range(len(X))])
    return ms, rec, pd.DataFrame(S, columns=oof_cols), pd.DataFrame(retained)


def stability(imps: list[pd.Series]) -> float:
    """Mean pairwise Spearman rho of global |SHAP| rankings across CV folds."""
    return float(np.mean([stats.spearmanr(a, b)[0] for a, b in combinations(imps, 2)]))


def table1(rec, n, out):
    ref = rec["KG-XAI (proposed)"]["auc"]
    n_test = n // 5
    rows = []
    for name, r in rec.items():
        t, p = corrected_ttest(ref, r["auc"], n - n_test, n_test) if name != "KG-XAI (proposed)" else (np.nan, np.nan)
        ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
        rows.append({"Model": name, "AUC": ms(r["auc"]), "F1": ms(r["f1"]), "Accuracy": ms(r["acc"]),
                     "Brier": ms(r["brier"]), "DVR (%)": f"{100 * np.mean(r['dvr']):.1f}",
                     "DVR >= 1 pp (%)": f"{100 * np.mean(r['dvr1']):.1f}",
                     "SHAP stability (rho)": f"{stability(r['imp']):.3f}" if r["imp"] else "–",
                     "p (AUC vs proposed)": "–" if np.isnan(p) else f"{p:.3f}",
                     "_auc": np.mean(r["auc"]), "_dvr": np.mean(r["dvr"])})
    T = pd.DataFrame(rows)
    T.drop(columns=["_auc", "_dvr"]).to_csv(f"{out}/table1_performance.csv", index=False)
    with open(f"{out}/table1_performance.md", "w") as fh:
        fh.write(T.drop(columns=["_auc", "_dvr"]).to_markdown(index=False))
    return T


def table2(X, y, S, kept, out):
    K = add_knowledge(X)
    rows = []
    for rid, name, view, theory, sign, _ in KB:
        fired = K[rid].to_numpy() == 1
        rate_on, rate_off = y[fired].mean(), y[~fired].mean()
        lift = rate_on / y.mean()
        rows.append({"Rule": rid, "Knowledge rule": name, "Theory": theory, "View": view,
                     "Expected": "+" if sign > 0 else "−",
                     "Support (%)": round(100 * fired.mean(), 1),
                     "Success if fired (%)": round(100 * rate_on, 1),
                     "Success if not (%)": round(100 * rate_off, 1),
                     "Lift": round(lift, 2),
                     "Retained in folds (%)": round(100 * (kept[rid] != 0).mean(), 0),
                     "Mean SHAP when fired": round(float(S[rid][fired].mean()), 3)})
    T = pd.DataFrame(rows)
    T.to_csv(f"{out}/table2_knowledge_base.csv", index=False)
    with open(f"{out}/table2_knowledge_base.md", "w") as fh:
        fh.write(T.to_markdown(index=False))
    return T


# --------------------------------------------------------------------------- #
# 6. Figures                                                                   #
# --------------------------------------------------------------------------- #
INK, QUIET, EDGE, ACCENT, WARM = "#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": EDGE,
                     "axes.labelcolor": INK, "xtick.color": QUIET, "ytick.color": QUIET})


def figure1(n, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    ax.set_xlim(0, 100), ax.set_ylim(0, 50), ax.axis("off")

    def box(x, y, w, h, title, lines, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=(0.165, 0.47, 0.84, 0.10) if accent else "white",
                                    ec=ACCENT if accent else EDGE, lw=1.4 if accent else 0.9))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=6.5,
                fontweight="bold", color=INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 7.0 - 3.3 * i, ln, ha="center", va="top", fontsize=6.2,
                    color=QUIET)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=EDGE, lw=0.9))

    box(0.5, 6, 18, 38, "Multi-source data", ["Funding (8)", "Network (3)", "Milestone (3)",
                                               "Temporal (4)", "Ecosystem (5)", "Sector (9)",
                                               f"{n} startups"])
    box(25, 27, 22, 17, "Knowledge base", ["9 theory-grounded rules", "signalling, syndication,",
                                           "legitimacy, clusters"])
    box(25, 6, 22, 17, "Knowledge refinement", ["rules kept only if the", "training fold agrees;",
                                                "s_K(x) = Σ d_j r_j(x)"])
    box(53.5, 6, 21, 38, "KG-XAI learner", ["boosted trees on x ⊕ r(x) ⊕ s_K(x)", "",
                                                   "monotone constraints", "follow the refined", "direction d_j"],
        accent=True)
    box(80.5, 27, 19, 17, "Prediction", ["P(acquisition | x)", "decision support for", "investors, accelerators"])
    box(80.5, 6, 19, 17, "Explanation", ["exact TreeSHAP", "by view and by rule,", "consistency audit"])
    arrow(19.3, 35.5, 24.6, 35.5), arrow(19.3, 14.5, 24.6, 14.5)
    arrow(36, 26.6, 36, 23.4), arrow(47.4, 14.5, 53.1, 14.5), arrow(47.4, 35.5, 53.1, 35.5)
    arrow(74.9, 35.5, 80.1, 35.5), arrow(74.9, 14.5, 80.1, 14.5)
    fig.savefig(f"{out}/fig1_framework.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


NICE = {"log_funding": "Total funding (log)", "funding_rounds": "Funding rounds",
        "relationships": "Relationships", "milestones": "Milestones",
        "age_first_milestone_year": "Age at first milestone", "age_last_milestone_year": "Age at last milestone",
        "age_first_funding_year": "Age at first funding", "age_last_funding_year": "Age at last funding",
        "funding_span": "Funding span (yrs)", "founded_year": "Founding year",
        "avg_participants": "Avg. investors per round", "is_top500": "Top-500 flag", "kb_score": "Knowledge score s_K",
        **{r[0]: f"{r[0]} {r[1]}" for r in KB}}


def figure2(X, S, out):
    K = add_knowledge(X)[S.columns]
    view_imp = {v: np.abs(S[[c for c in S.columns if VIEW_OF.get(c) == v]]).sum(1).mean() for v in VIEWS}
    view_imp["Knowledge rules"] = np.abs(S[RULES + ["kb_score"]]).sum(1).mean()
    vi = pd.Series(view_imp).sort_values()
    top = np.abs(S).mean().sort_values(ascending=False).head(12).index[::-1]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=300, gridspec_kw={"width_ratios": [1, 1.6]})
    colors = [ACCENT if v == "Knowledge rules" else "#86b6ef" for v in vi.index]
    a.barh(vi.index, vi.values, color=colors, height=0.6)
    for i, v in enumerate(vi.values):
        a.text(v + vi.max() * 0.02, i, f"{v:.2f}", va="center", fontsize=7, color=INK)
    a.tick_params(axis="y", labelsize=7.5)
    a.set_xlabel("Mean |SHAP| per startup (log-odds)")
    a.set_title("(a) Contribution by source view", fontsize=8.5, loc="left", color=INK)
    a.set_xlim(0, vi.max() * 1.22)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    cmap = LinearSegmentedColormap.from_list("bo", [ACCENT, "#d9d8d3", WARM])
    rng = np.random.default_rng(SEED)
    for i, f in enumerate(top):
        v = K[f].to_numpy()
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        c = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        b.scatter(S[f], i + rng.uniform(-0.28, 0.28, len(v)), c=c, cmap=cmap, s=3.5, lw=0, alpha=0.85)
    b.axvline(0, color=EDGE, lw=0.6)
    b.set_yticks(range(len(top)), [NICE.get(f, f) for f in top], fontsize=7)
    b.set_xlabel("SHAP value (impact on log-odds of acquisition)")
    b.set_title("(b) Top-12 features, out-of-fold", fontsize=8.5, loc="left", color=INK)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=b, fraction=0.04, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=6.5)
    cb.set_label("Feature value", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_explanations.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return vi, np.abs(S).mean().sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    X, y = build_views(load(args.data))
    print(f"{len(X)} startups, {X.shape[1]} features in {len(VIEWS)} views, success rate {y.mean():.3f}")
    ms, rec, S, kept = run(X, y, args.repeats)
    T1, T2 = table1(rec, len(X), OUT), table2(X, y, S, kept, OUT)
    figure1(len(X), OUT)
    vi, gi = figure2(X, S, OUT)

    summary = {
        "n_startups": int(len(X)), "n_features": int(X.shape[1]), "success_rate": float(y.mean()),
        "cv": f"5-fold x {args.repeats} repeats",
        "models": {n: {k: [float(np.mean(v)), float(np.std(v, ddof=1))]
                       for k, v in r.items() if k not in ("imp",)} for n, r in rec.items()},
        "stability": {n: stability(r["imp"]) for n, r in rec.items() if r["imp"]},
        "signed_feature_retention": {f: float((kept[f] != 0).mean()) for f in SIGNED},
        "view_importance": vi.round(4).to_dict(),
        "top_features": gi.head(12).round(4).to_dict(),
    }
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(T1.drop(columns=["_auc", "_dvr"]).to_string(index=False))
    print(T2.to_string(index=False))


if __name__ == "__main__":
    main()
