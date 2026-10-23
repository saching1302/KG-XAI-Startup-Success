# KG-XAI: Knowledge-Guided Explainable AI for Startup Success Prediction

Code and results for the short communication
**"An Explainable AI Knowledge-Based Model for Predicting Startup Success Using Multi-Source Entrepreneurial Data"**
(submitted to *Knowledge-Based Systems*).

KG-XAI joins three parts into one decision-support model:

1. **A knowledge base.** Nine rules from entrepreneurship and venture-finance theory (signalling, staged financing, syndication, network endorsement, legitimacy, clusters, liability of newness).
2. **A knowledge-guided learner.** Gradient-boosted trees learn from the multi-source data plus the rule activations. Monotone constraints follow each rule's direction. Within each training fold, the knowledge is first validated and refined: a rule the data contradict is released rather than imposed.
3. **Exact explanations.** TreeSHAP attributions are reported per source view and per fired rule, along with a consistency audit (the directional violation rate).

![Framework](results/fig1_framework.png)

## Reproduce

```bash
pip install -r requirements.txt
python kgxai.py               # about 2 minutes on a laptop; uses data/startup_data.csv
```

All outputs are written to `results/`:

| File | Content |
|---|---|
| `table1_performance.md/.csv` | Table 1: accuracy, calibration and explanation quality of 8 models (5-fold × 5 repeated CV) |
| `table2_knowledge_base.md/.csv` | Table 2: the knowledge base with each rule's support, lift, retention after refinement and mean SHAP |
| `fig1_framework.png` | Figure 1: the KG-XAI framework |
| `fig2_explanations.png` | Figure 2: out-of-fold SHAP by source view and top features |
| `results.json` | Every metric (mean, sd) for every model |

## Key results (922 startups, 5 × 5 CV)

| Model | AUC | DVR (%) | DVR ≥ 1 pp (%) |
|---|---|---|---|
| LightGBM (data only) | 0.828 ± 0.039 | 38.1 | 20.0 |
| Knowledge base only | 0.777 ± 0.041 | 7.6 | 7.3 |
| **KG-XAI (proposed)** | **0.821 ± 0.037** | **6.1** | **4.1** |

KG-XAI matches the black-box accuracy: the AUC difference is not significant under the corrected resampled t-test (p = 0.31). It also cuts theory-violating "what-if" responses by about 6×. The refinement step finds that the *VC certification* rule (R1) is contradicted by these data and releases it.

## Data

`data/startup_data.csv` contains the public Crunchbase-derived *Startup Success Prediction* dataset (923 U.S. startups, 49 columns; target `status` = acquired / closed). It was published on Kaggle by Manish KC: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

The file is included unchanged (MD5 `817a0130e2d3f763af4632afed46654a`) so the results can be reproduced. All rights to the data remain with its original publisher, and the Kaggle page lists no explicit license; please cite the Kaggle source when you use it. The file will be removed on request from the rights holder. Leakage columns (`closed_at`, `labels`, identifiers) are never used by the model.

## Files

`kgxai.py` is the whole pipeline (data → knowledge base → model → evaluation → tables → figures). `data/` holds the input dataset, `requirements.txt` lists the dependencies, and `results/` holds the generated outputs.

## License

MIT
