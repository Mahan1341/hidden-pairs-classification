# Hidden Pairs: kernel fusion and rank ensembling

A tabular classification solution for **[Week 03: Hidden Pairs](https://www.kaggle.com/competitions/week-03-hidden-pairs)**, an Introduction to Machine Learning course competition at Innopolis University.

**CHIMIN team result: 0.94832 private ROC-AUC, rank 14 of 94 entries in the final leaderboard export.** The approach combines several representations of the same 512-dimensional input with a small, fixed SVM ensemble.

[Русское описание](README.ru.md) · [Final notebook](solution.ipynb) · [Runnable script](solution.py) · [Recorded results](reports/recorded_results.json)

## Task and data

Predict a binary target from **512 anonymous, nonnegative numerical features** named `f0000`–`f0511`.

| Split | Rows | Purpose |
|---|---:|---|
| Train | 2,520 | Fit preprocessing and models; historical cross-validation |
| Validation | 560 | Compare candidate methods during research |
| Test | 1,116 | Generate the competition submission |

Train and validation are balanced between the two classes. The final output contains `row_id` and `target`, preserving the test row order. Data and prediction CSVs are obtained separately from Kaggle and are excluded from this repository.

## Method

The main component, **D v2**, is one `SVC(C=2, kernel="precomputed")` trained on a weighted sum of five RBF kernels:

| Representation | Transformation | Kernel weight |
|---|---|---:|
| Raw | `StandardScaler` | 0.288 |
| Rank-Gauss | `QuantileTransformer(output_distribution="normal")` | 0.288 |
| PCA64 | Standardized input, 64 whitened principal components | 0.160 |
| Direction | Input divided by its L2 norm | 0.200 |
| Polar geometry | Class-specific direction/radius prototypes; 28 retained features | 0.064 |

The supervised geometry component fits 32 KMeans prototypes per class. Its training features are produced with **five-fold cross-fitting**, so a row's own label is excluded from the prototype model used to transform that row. A full-train geometry model transforms validation and test data.

Two complementary branches clip each original feature at its training 99.5th percentile, then apply either ordinary PCA128 or `log1p` followed by PCA128. Both use an RBF SVC with `C=3`; their gamma is `2 / (128 × variance of training PCA scores)`.

The three models' decision scores are converted into percentile ranks within the scored batch:

```python
prediction = 0.63 * rank(D_v2) + 0.27 * rank(PCA128) + 0.10 * rank(Log_PCA128)
```

Here `rank(s) = scipy.stats.rankdata(s, method="average") / len(s)`. These values are ranking scores, not calibrated class probabilities. Rank blending also means predictions depend on the set of rows scored together.

## Recorded results

| Model | Reported 5-fold CV ROC-AUC | Validation ROC-AUC |
|---|---:|---:|
| D v2 | 0.93658037 | 0.95025510 |
| PCA128 branch | 0.93037919 | 0.95061224 |
| Log-PCA128 branch | 0.93341522 | 0.95237245 |
| D v2 / PCA128, 70/30 | 0.93790942 | 0.95397321 |
| **Final 63/27/10 ensemble** | **0.93805430** | **0.95454082** |

The final submission achieved **0.94826 public** and **0.94832 private ROC-AUC**. The saved local run took **44.62 seconds overall**, including **2.75 seconds for test inference**. These are historical measurements on the original machine, not runtime promises for another environment.

Sources: the original final notebook's experiment table and saved stdout, the later defense notebook, and the CHIMIN row of the private leaderboard exported on **2026-09-30**. [Machine-readable provenance](reports/recorded_results.json) and [sanitized historical stdout](reports/original_run.txt) are included. Other teams' leaderboard rows are not included.

Validation was consulted repeatedly during research. Its score should therefore not be treated as performance on an untouched holdout, or as proof that the small CV improvements are statistically significant. The reported CV summary is retained as recorded; it is not relabeled as pooled OOF AUC.

## Run

Use a Python environment compatible with the pinned dependencies; Python 3.11 is a suitable setup choice. The original Python version was not recorded.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python solution.py
```

Before running, place the competition's `train.csv`, `validation.csv`, and `test.csv` **in the repository root**. The script intentionally checks the original dataset shapes and expected validation AUCs. It fits the fixed final model on train only and writes `final_submission.csv`. It does not rerun the historical outer five-fold experiment or tune parameters.

For the notebook, use a Jupyter environment with these dependencies and run all cells in `solution.ipynb`. Its executable code is identical to `solution.py`.

If the original `submission_D_PCA128_LogPCA128_63_27_10.csv` is also present, the code verifies prediction agreement within `1e-6`. Otherwise it explicitly reports that exact submission equivalence was not checked. The saved original run recorded a maximum absolute difference of `2.22044604925e-16`.

The implementation uses four numerical-library threads, random seed 42, and geometry cross-fitting seed 137. NumPy, SciPy, pandas and scikit-learn versions are pinned to the versions printed in the successful historical run. The complete model has not been retrained during repository preparation because the input data are not included.

