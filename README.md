# KG-XAI: Knowledge-Guided Explainable AI for Startup Success Prediction

Code, data and results for the short communication
**"An Explainable AI Knowledge-Based Model for Predicting Startup Success Using Multi-Source Entrepreneurial Data"**
(submitted to *Knowledge-Based Systems*).

KG-XAI combines a knowledge base of nine entrepreneurship rules with gradient-boosted trees:

1. **Knowledge base.** Rules from signalling, staged financing, resource-based, syndication, endorsement, legitimacy, cluster and liability-of-newness theory.
2. **In-fold refinement.** A rule, and the raw features its theory signs, is kept only if the training fold agrees with its direction. Contradicted knowledge is released and flagged.
3. **Knowledge-guided learner.** LightGBM on raw features plus rule activations, with monotone constraints that follow the validated directions.
4. **Explanations and audit.** Exact TreeSHAP attributions by source view and by fired rule, plus a directional violation rate (DVR) that measures theory-violating what-if responses.

![Framework](results/fig1_framework.png)

## Reproduce

```bash
pip install -r requirements.txt
python kgxai.py             # full run: about 30 min on 2 CPU cores, faster with more
python kgxai.py --quick     # smoke test: 1 repeat, 2 stress draws
```

## Studies

| | Study 1 (benchmark) | Study 2 (leakage-safe) |
|---|---|---|
| Source | Kaggle *Startup Success Prediction* | Open Crunchbase snapshot (Dec 2015), 4 linked tables |
| Startups | 922 (U.S.) | 7,455 (worldwide), first funded 2005–2012 |
| Success | acquired (64.6%) | acquired or IPO (49.2%) |
| Features | 32 in 6 views (lifetime snapshot) | 36 in 6 views, only information dated before first funding + 2 years |
| Extra test | – | out-of-time: train 2005–09, test 2010–12 |

## Experiments

- **E1 – Accuracy and consistency.** 10 models (5 baselines including an EBM-style boosted additive model, a knowledge-only scorer, and 3 ablations); 5 × 5 repeated CV with nested 3-fold tuning; corrected resampled t-test; an out-of-time test with a paired bootstrap.
- **E2 – Wrong-knowledge stress test.** k = 1–3 confirmed rules are given the wrong direction, with up to 10 random draws each. Refinement is compared with naive knowledge injection.
- **E3 – Robustness.** Rule thresholds scaled ×0.5–×1.5, and Study 1 re-run without its lifetime-accumulation features.

## Key results

| Model | S1 AUC | S2 AUC (CV) | S2 AUC (2010–12 test) | S2 DVR (%) |
|---|---|---|---|---|
| LightGBM (best data-only) | 0.834 | 0.815 | 0.840 | 15.6 |
| Boosted GAM (EBM-style) | 0.831 | 0.806 | 0.825 | 17.4 |
| Knowledge base only | 0.778 | 0.765 | 0.796 | 0.8 |
| **KG-XAI (proposed)** | **0.827** | **0.814** | **0.837** | **0.1** |

KG-XAI is statistically indistinguishable from LightGBM in both studies and out of time (p = 0.15), while nearly eliminating theory-violating responses. In the stress test, refinement released **100%** of the wrongly-directed knowledge and **0%** of the correct knowledge, and its accuracy stayed flat while naive injection degraded. Full numbers are in `results/`.

## Outputs (`results/`)

| File | Content |
|---|---|
| `table1_performance.*` | Table 1: accuracy, calibration, P@10%, DVR, stability (both studies, out-of-time) |
| `table1_full_metrics.csv` | Every metric with SDs and p-values |
| `table2_knowledge_robustness.*` | Table 2: stress-test summary; `table2_stress_test_draws.csv` has every draw |
| `tableS1_rule_validation.*` | Supplementary Table S1: rule support, lift, retention, mean SHAP |
| `tableS2_threshold_sensitivity.csv` | Supplementary Table S2: threshold sensitivity |
| `fig1_framework.png`, `fig2_results.png` | Figures 1–2 |
| `figS1_shap_s2.png`, `figS2_shap_s1.png` | Supplementary SHAP plots |
| `results.json` | All metrics, chosen hyperparameters, refinement retention |

## Data

- `data/startup_data.csv` – the Kaggle *Startup Success Prediction* dataset by Manish KC (https://www.kaggle.com/datasets/manishkc06/startup-success-prediction). It is included unchanged (MD5 `817a0130e2d3f763af4632afed46654a`). The Kaggle page lists no explicit license, all rights remain with the publisher, and the file will be removed on request.
- `data/crunchbase_s2_features.csv` – derived Study 2 features. It was built by `kgxai.py` from the open Crunchbase export (© Crunchbase, CC BY-NC 4.0, mirrored at https://github.com/notpeter/crunchbase-data). Delete the file to rebuild it from the raw tables, which are downloaded automatically. It is for non-commercial use only.

## Files

`kgxai.py` is the whole pipeline: data → knowledge base → models → experiments → tables → figures. `data/` holds the inputs, and `results/` holds the generated outputs.

## License

The code is MIT-licensed. The data keep their original licenses (see above).
