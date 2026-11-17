# %% [markdown]
#  # Predictive Maintenance: Machine Failure Classification
# 
# 
# 
#  > **Term project:** A Comparative Study of Feature Reduction Strategies and Machine Learning Classifiers for Industrial Fault Detection
# 
# 
# 
#  ## 1. Project Objective|
# 
# 
# 
#  This project investigates whether selecting a smaller set of machine sensors can produce a useful classifier for the `Machine failure` target in the AI4I 2020 predictive-maintenance dataset. The primary metric is positive-class failure F1, supported by failure recall, precision, balanced accuracy, missed failures, and false alarms.
# 
# 
# 
#  ## 2. Experiment Roadmap
# 
# 
# 
#  The notebook follows one fixed workflow from start to finish:
# 
# 
# 
#  1. **Load and explore the data.** Inspect columns, data types, missing values, duplicates, class balance, and simple relationships.
# 
#  2. **Prepare predictors and target.** Keep the 6 usable raw sensor variables, encode `Type`, and exclude identifiers and failure-cause columns that would leak the target.
# 
#  3. **Create fixed data splits.** Use stratified 70% training, 15% validation, and 15% test splits. The test split is hidden until the final step.
# 
#  4. **Configure experiment settings.** Set the number of selected features, selector models, PCA variance threshold, and cross-validation folds in one place.
# 
#  5. **Fit feature-selection methods on training data only.** Compare ANOVA, Pearson correlation, adapted Lasso, RFE, forward/backward selection, and PCA.
# 
#  6. **Screen model combinations.** Fit the baseline and selected-feature models on training data and compare them on validation data.
# 
#  7. **Choose three finalists & test Stacking.** Select the top three distinct model architectures using validation failure F1, recall, and precision, and test whether combining them into a Stacking Ensemble improves performance.
# 
#  8. **Fine-tune finalists.** Search focused hyperparameter grids using five-fold stratified cross-validation on the training split.
# 
#  9. **Choose the tuned winner & refit.** Compare the tuned finalists on validation performance, then refit the winning configuration on the combined training and validation data (85%).
# 
#  10. **Evaluate once on the test set.** Report the final unseen-data result, confusion matrix, ROC & AUC curve, missed failures, and false alarms.
# 
# 
# 
#  ## 3. Important Interpretation Rule
# 
# 
# 
#  The validation tables answer **which approach should be selected**. The final test table answers **how the already-selected approach performs on unseen data**. A test score must not be used to choose a different winner.

# %% [markdown]
#  ## 4. Load Data

# %%
# Reproducibility and common imports
import os
import random

RANDOM_STATE = 2026
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(RANDOM_STATE)


# %%
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "ai4i2020.csv"

# Load the latest version
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
    file_path,
)

print("First 5 records:")
display(df.head())


# %% [markdown]
#  ## 5. Explore
# 
# 
# 
#  Inspect the AI4I 2020 predictive-maintenance dataset structure, data types, missing values, duplicates, class balance, and physical sensor relationships.

# %%
display(df.columns)


# %%
df.info()


# %%
print(df["Type"])


# %%
display(df)


# %%
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate UDIs:", df["UDI"].duplicated().sum())


# %%
counts = df["Machine failure"].value_counts()
print(counts, "\n", (counts / len(df) * 100).round(2))

ax = sns.countplot(x="Machine failure", data=df)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.35, p.get_height() + 50))
plt.title("Class balance: Machine failure")
plt.show()


# %%
sns.set_theme(style="whitegrid")

# Original column names (used for EDA visualization only)
air, proc = "Air temperature [K]", "Process temperature [K]"
rpm, tq, wear = "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Temporary helper columns for plotting physical failure boundaries in EDA
eda_df = df.copy()
eda_df["Temp_diff"] = eda_df[proc] - eda_df[air]
eda_df["Power"] = eda_df[tq] * eda_df[rpm] * 2 * np.pi / 60
eda_df["Strain"] = eda_df[wear] * eda_df[tq]

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.flatten()

# 1) Count of each failure mode
mode_counts = eda_df[modes].sum().sort_values()
mode_counts.plot(kind="barh", ax=axes[0], color="steelblue")
for i, v in enumerate(mode_counts):
    axes[0].text(v + 1, i, int(v), va="center")
axes[0].set_title("Number of failures per mode")
axes[0].set_xlabel("Count")

# 2) TWF: tool wear (failures expected between 200 and 240 min)
sns.histplot(data=eda_df, x=wear, hue="TWF", bins=40, multiple="stack", ax=axes[1])
axes[1].axvline(200, color="red", ls="--")
axes[1].axvline(240, color="red", ls="--")
axes[1].set_title("TWF: tool wear (rule: 200-240 min)")

# 3) HDF: temperature difference < 8.6 K and speed < 1380 rpm
sns.scatterplot(data=eda_df, x=rpm, y="Temp_diff", hue="HDF", s=15, ax=axes[2])
axes[2].axhline(8.6, color="red", ls="--")
axes[2].axvline(1380, color="red", ls="--")
axes[2].set_title("HDF: temp difference vs speed")

# 4) PWF: power < 3500 W or > 9000 W
sns.scatterplot(data=eda_df, x=tq, y=rpm, hue="PWF", s=15, ax=axes[3])
axes[3].set_title("PWF: torque vs speed")
axes[3].set_ylim(eda_df[rpm].min(), eda_df[rpm].max())
t = np.linspace(max(eda_df[tq].min(), 1), eda_df[tq].max(), 300)
for P in (3500, 9000):
    axes[3].plot(t, P * 60 / (2 * np.pi * t), color="red", ls="--")

# 5) OSF: strain above 11000 (L), 12000 (M), 13000 (H)
sns.scatterplot(data=eda_df, x=wear, y=tq, hue="OSF", s=15, ax=axes[4])
w = np.linspace(max(eda_df[wear].min(), 1), eda_df[wear].max(), 300)
for lim in (11000, 12000, 13000):
    axes[4].plot(w, lim / w, ls="--", label=f"{lim}")
axes[4].set_ylim(eda_df[tq].min(), eda_df[tq].max())
axes[4].set_title("OSF: tool wear vs torque (dashed = strain limits)")

# 6) RNF: random by design, so no rule region is expected
sns.scatterplot(data=eda_df, x=wear, y=tq, hue="RNF", s=15, ax=axes[5])
axes[5].set_title("RNF: random failures (no pattern expected)")

plt.tight_layout()
plt.show()


# %%
print(np.unique(df["Machine failure"]))


# %% [markdown]
#  ## 6. Pre-processing and Data Splitting
# 
# 
# 
#  Select the 6 usable raw predictor columns, encode machine `Type` (`L=0, M=1, H=2`), define the target, and create fixed stratified training (70%), validation (15%), and test (15%) splits. Identifiers (`UDI`, `Product ID`) and failure-cause columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded so they cannot leak the target label.

# %%
failure_flags = ["TWF", "HDF", "PWF", "OSF", "RNF"]
failure_without_flags = (
    (df["Machine failure"] == 1)
    & df[failure_flags].eq(0).all(axis=1)
).sum()
rnf_without_failure = (
    (df["RNF"] == 1)
    & (df["Machine failure"] == 0)
).sum()

print(
    "Machine failure == 1 with all five failure flags == 0:",
    failure_without_flags,
)
print("RNF == 1 with Machine failure == 0:", rnf_without_failure)

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

x = df[features].copy()
y = df["Machine failure"]
print("Predictor shape:", x.shape)
print("Target shape:   ", y.shape)


# %%
# Encode ordinal quality type from string to integer
x["Type"] = x["Type"].map({"L": 0, "M": 1, "H": 2})
display(x.head())


# %%
print("Unique Type values:", x["Type"].unique())
print(x.dtypes)
print("Missing values:\n", x.isna().sum())


# %%
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Keep the test set untouched until the final evaluation.
x_train, x_temp, y_train, y_temp = train_test_split(
    x,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"Training samples:   {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples:       {len(x_test)}")


# %%
# Check class balance in the development splits. The test labels remain unused until final evaluation.
display(
    pd.DataFrame({
        "Training": y_train.value_counts(),
        "Validation": y_val.value_counts(),
    }).fillna(0).astype(int)
)


# %% [markdown]
#  ## 7. Master Experiment Configuration
# 
# 
# 
#  Change the variables in the cell below and re-run the notebook to test different feature counts, RFE/SFS wrapper selectors, PCA variance thresholds, or finalist selection rules.

# %%
# ── Master Experiment Configuration ──────────────────────────────────
# Note: RANDOM_STATE = 2026 is already set in Section 4 and shared across the notebook.
# Note: All configurations below are used to control the behavior of the experiment.
# # ─────────────────────────────────────────────────────────────────────
# # ─────────────────────────────────────────────────────────────────────

NUMBER_OF_FEATURES = 4           # Features kept by ANOVA / Correlation / Lasso / RFE / SFS (out of 6)

RFE_SELECTOR_MODEL = "Logistic Regression"
# Options: "Logistic Regression", "Linear SVC", "Decision Tree", "Random Forest"

SFS_SELECTOR_MODEL = "Decision Tree"
# Options: "Logistic Regression", "Decision Tree", "Random Forest"
# (Tip: "Decision Tree" or "Logistic Regression" runs much faster than 200-tree "Random Forest")

PCA_VARIANCE = 0.95               # Cumulative training variance PCA must retain
CV_FOLDS     = 5                  # Folds for GridSearchCV and SFS

UNIQUE_FINALIST_MODELS = False   # True  -> Top 3 finalists must be 3 distinct model architectures
                                    # False -> Allow same model architecture with different feature sets
                                    
#──────────────────────────────────────────────────────────────────────

# NUMBER_OF_FEATURES = 3
# RFE_SELECTOR_MODEL = "Random Forest"
# SFS_SELECTOR_MODEL = "Decision Tree"
# PCA_VARIANCE = 0.95
# CV_FOLDS = 5
# UNIQUE_FINALIST_MODELS = True

# Consistent sort order for ALL validation comparison tables
PRIMARY_SORT_METRICS = [
    "Validation failure F1",
    "Validation failure recall",
    "Validation failure precision",
    "Validation accuracy",
]


# %% [markdown]
#  ## 8. Modelling and Evaluation Setup
# 
# 
# 
#  The baseline classifiers compared across every feature-selection strategy are:
# 
# 
# 
#  - **Support Vector Classifier (SVC)** (using RBF kernel for baseline screening; Linear, Poly, and RBF tested in tuning)
# 
#  - **Logistic Regression**
# 
#  - **Decision Tree** (Gini impurity)
# 
#  - **Random Forest** (Bagging ensemble)
# 
#  - **Gradient Boosting Classifier** (Boosting ensemble)
# 
#  - **Neural Network MLP** (Multilayer Perceptron)
# 
# 
# 
#  *(Note: **Stacking Ensemble** is evaluated dynamically in Section 15 by combining the Top 3 screening models).*
# 
# 
# 
#  ### 8.1 Experimental Rules & Metrics
# 
#  - **No feature selection** (all 6 sensors) is the reference baseline.
# 
#  - Filter, Embedded, and Wrapper methods retain `NUMBER_OF_FEATURES` sensors so the screening comparison is controlled.
# 
#  - All feature selectors and scalers are fitted on **training data only** (`x_train`).
# 
#  - Because machine failures are rare (3.39%), accuracy is misleading. All tables are ranked by `PRIMARY_SORT_METRICS`:
# 
#    1. **Validation failure F1** (balance between precision and recall)
# 
#    2. **Validation failure recall** (minimizing missed failures / false negatives)
# 
#    3. **Validation failure precision** (minimizing false alarms / false positives)
# 
#    4. **Validation accuracy** (tie-breaker)

# %%
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

selection_models = {
    "SVC": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    "MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=1000, random_state=RANDOM_STATE),
    ),
}

def evaluate_selected_features(selection_name, selected_features):
    """Fit all screening models on x_train[selected_features] and evaluate on x_val."""
    selected_features = [col for col in x_train.columns if col in selected_features] 
    rows = []
    for model_name, model in selection_models.items():
        fitted_model = clone(model)
        fitted_model.fit(x_train[selected_features], y_train)
        predictions = fitted_model.predict(x_val[selected_features])
        report = classification_report(
            y_val, predictions, output_dict=True, zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()
        rows.append({
            "Feature selection": selection_name,
            "Features": ", ".join(selected_features),
            "Model": model_name,
            "Validation accuracy": report["accuracy"],
            "Validation balanced accuracy": balanced_accuracy_score(y_val, predictions),
            "Validation failure precision": report["1"]["precision"],
            "Validation failure recall": report["1"]["recall"],
            "Validation failure F1": report["1"]["f1-score"],
            "Validation missed failures": fn,
            "Validation false alarms": fp,
        })
    return pd.DataFrame(rows)


# %% [markdown]
#  ## 9. No Feature Selection: Baseline
# 
# 
# 
#  Evaluate all 6 models using all 6 original sensor predictors. This serves as our benchmark before removing any features.

# %%
baseline_results = evaluate_selected_features(
    "No feature selection", x_train.columns.tolist()
)
display(baseline_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))


# %% [markdown]
#  ## 10. Filter Methods
# 
# 
# 
#  Filter methods score each sensor independently based on statistical tests against the target before any classifier is trained.
# 
# 
# 
#  ### 10.1 ANOVA F-Test Filter Selection
# 
#  ANOVA (`f_classif`) measures whether the mean of each sensor differs significantly between normal machines (`0`) and failed machines (`1`), keeping the top `NUMBER_OF_FEATURES` sensors.

# %%
anova_filter = SelectKBest(f_classif, k=NUMBER_OF_FEATURES)
anova_filter.fit(x_train, y_train)
anova_features = x_train.columns[anova_filter.get_support()].tolist()

print(f"ANOVA-selected features ({NUMBER_OF_FEATURES}):", anova_features)
anova_results = evaluate_selected_features("ANOVA", anova_features)
display(anova_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))


# %% [markdown]
#  ### 10.2 Pearson Correlation Selection
# 
#  Pearson's $R$ measures the absolute linear correlation between each sensor and the binary `Machine failure` target on the training split.

# %%
correlation_scores = x_train.apply(
    lambda column: abs(np.corrcoef(column, y_train)[0, 1])
).sort_values(ascending=False)
correlation_features = correlation_scores.head(NUMBER_OF_FEATURES).index.tolist()

print(f"Pearson-selected features ({NUMBER_OF_FEATURES}):", correlation_features)
display(correlation_scores.to_frame("absolute Pearson correlation"))

correlation_results = evaluate_selected_features(
    "Pearson correlation", correlation_features
)
display(correlation_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))


# %% [markdown]
#  ## 11. Embedded Methods (Regularization)
# 
# 
# 
#  Embedded methods perform feature selection as part of the model optimization objective (penalizing large weights).
# 
# 
# 
#  ### 11.1 Adapted Lasso (L1) Selection
# 
#  Lasso adds an $L_1$ penalty ($\lambda \sum \vert{}w_i\vert{}$) that shrinks weak sensor coefficients toward zero. We select the `NUMBER_OF_FEATURES` sensors with the largest non-zero $L_1$ coefficients.

# %%
adapted_lasso = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty="l1", solver="liblinear", C=0.1,
        max_iter=2000, random_state=RANDOM_STATE,
    ),
)
adapted_lasso.fit(x_train, y_train)

lasso_coefficients = np.abs(adapted_lasso[-1].coef_).ravel()
nonzero_lasso = np.flatnonzero(lasso_coefficients > 1e-8)
strongest_lasso = np.argsort(lasso_coefficients)[::-1]
lasso_indices = [
    index for index in strongest_lasso if index in nonzero_lasso
][:NUMBER_OF_FEATURES]

if lasso_indices:
    lasso_features = x_train.columns[lasso_indices].tolist()
else:
    lasso_features = x_train.columns[
        np.argsort(lasso_coefficients)[-NUMBER_OF_FEATURES:]
    ].tolist()

print(f"Adapted Lasso-selected features ({NUMBER_OF_FEATURES}):", lasso_features)
display(
    pd.Series(lasso_coefficients, index=x_train.columns)
    .sort_values(ascending=False)
    .to_frame("absolute L1 coefficient")
)

lasso_results = evaluate_selected_features("Adapted Lasso", lasso_features)
display(lasso_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))


# %% [markdown]
#  ## 12. Wrapper Methods
# 
# 
# 
#  Wrapper methods use a machine learning model to evaluate subsets of features:
# 
#  - **Recursive Feature Elimination (RFE):** Fits a model and iteratively prunes the weakest features based on coefficients or feature importances.
# 
#  - **Forward Selection (SFS):** Starts with 0 features and greedily adds the sensor that maximizes cross-validated F1.
# 
#  - **Backward Selection (SFS):** Starts with all 6 features and greedily removes the sensor whose removal hurts cross-validated F1 the least.
# 
# 
# 
#  ### 12.1 Recursive Feature Elimination (RFE)
# 
#  We first standardize `x_train` so linear and SVM coefficients are not distorted by sensor units (e.g., RPM vs. Torque), then compare four RFE base estimators. The active `RFE_SELECTOR_MODEL` from the Master Config is used for model screening.

# %%
scaler_rfe = StandardScaler()
x_train_s_df = pd.DataFrame(
    scaler_rfe.fit_transform(x_train),
    columns=x_train.columns,
    index=x_train.index,
)

rfe_selectors = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Linear SVC": SVC(
        kernel="linear", class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
}

rfe_selected_features = {}
for selector_name, estimator in rfe_selectors.items():
    selector = RFE(estimator, n_features_to_select=NUMBER_OF_FEATURES)
    selector.fit(x_train_s_df, y_train)
    rfe_selected_features[selector_name] = (
        x_train.columns[selector.get_support()].tolist()
    )

print("Features selected by each RFE base estimator:")
display(
    pd.DataFrame.from_dict(
        rfe_selected_features,
        orient="index",
        columns=[f"Feature {i + 1}" for i in range(NUMBER_OF_FEATURES)],
    )
)


# %%
rfe_features = rfe_selected_features[RFE_SELECTOR_MODEL]
print(f"Active RFE selector ({RFE_SELECTOR_MODEL}) features:", rfe_features)

rfe_results = evaluate_selected_features("RFE", rfe_features)
display(rfe_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))


# %% [markdown]
#  ### 12.2 Forward and Backward Sequential Feature Selection (SFS)
# 
#  Both wrapper methods evaluate feature subsets using `CV_FOLDS`-fold cross-validation with the configured `SFS_SELECTOR_MODEL`.

# %%
sfs_model_lookup = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    ),
}

sfs_estimator = sfs_model_lookup[SFS_SELECTOR_MODEL]
print(f"Running Forward & Backward SFS using estimator: {SFS_SELECTOR_MODEL}")

forward_selector = SequentialFeatureSelector(
    sfs_estimator,
    n_features_to_select=NUMBER_OF_FEATURES,
    direction="forward",
    scoring="f1",
    cv=CV_FOLDS,
    n_jobs=-1,
)
backward_selector = SequentialFeatureSelector(
    sfs_estimator,
    n_features_to_select=NUMBER_OF_FEATURES,
    direction="backward",
    scoring="f1",
    cv=CV_FOLDS,
    n_jobs=-1,
)

forward_selector.fit(x_train, y_train)
backward_selector.fit(x_train, y_train)
forward_features = x_train.columns[forward_selector.get_support()].tolist()
backward_features = x_train.columns[backward_selector.get_support()].tolist()

print("Forward-selected features: ", forward_features)
print("Backward-selected features:", backward_features)

forward_results = evaluate_selected_features("Forward selection", forward_features)
backward_results = evaluate_selected_features("Backward selection", backward_features)

display(forward_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))
display(backward_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))


# %% [markdown]
#  ## 13. Dimensionality Reduction
# 
# 
# 
#  ### 13.1 Principal Component Analysis (PCA)
# 
#  Unlike feature selection (which drops original sensors), PCA projects all 6 standardized sensors into orthogonal principal components that retain `PCA_VARIANCE` (95%) of the training variance.

# %%
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_val_s = scaler.transform(x_val)

pca_full = PCA(random_state=RANDOM_STATE).fit(x_train_s)
cum = np.cumsum(pca_full.explained_variance_ratio_)
print("Explained variance ratio:", np.round(pca_full.explained_variance_ratio_, 3))
print("Cumulative variance:     ", np.round(cum, 3))

plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.axhline(PCA_VARIANCE, color="red", ls="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance (training set)")
plt.show()


# %%
pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
x_train_pca = pca.fit_transform(x_train_s)
x_val_pca = pca.transform(x_val_s)
print("Components kept:", pca.n_components_)


# %%
pca_rows = []
for model_name, model in selection_models.items():
    fitted = clone(model)
    fitted.fit(x_train_pca, y_train)
    predictions = fitted.predict(x_val_pca)
    report = classification_report(
        y_val, predictions, output_dict=True, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()
    pca_rows.append({
        "Feature selection": "PCA",
        "Features": f"{pca.n_components_} components",
        "Model": model_name,
        "Validation accuracy": report["accuracy"],
        "Validation balanced accuracy": balanced_accuracy_score(y_val, predictions),
        "Validation failure precision": report["1"]["precision"],
        "Validation failure recall": report["1"]["recall"],
        "Validation failure F1": report["1"]["f1-score"],
        "Validation missed failures": fn,
        "Validation false alarms": fp,
    })

pca_results = pd.DataFrame(pca_rows)
print("PCA evaluation:")
display(pca_results.sort_values(PRIMARY_SORT_METRICS, ascending=False))


# %% [markdown]
#  ## 14. Validation Comparison and Candidate Screening
# 
# 
# 
#  We now combine all validation results from every feature-selection method × model combination.
# 
#  - **Why we do this:** Screening lets us quickly compare all combinations using default hyperparameters on the Validation set ("practice exam").
# 
#  - The Test set ("final exam") remains completely hidden.

# %%
no_type_results = evaluate_selected_features(
    "Drop Type only",
    [c for c in x_train.columns if c != "Type"],
)

result_tables = {
    "No feature selection": baseline_results,
    "Drop Type only": no_type_results,
    "ANOVA": anova_results,
    "Pearson correlation": correlation_results,
    "Adapted Lasso": lasso_results,
    "Forward selection": forward_results,
    "Backward selection": backward_results,
    "RFE": rfe_results,
    "PCA": pca_results,
}

all_comparison_results = pd.concat(
    list(result_tables.values()),
    ignore_index=True,
)

selection_order = list(result_tables)
all_comparison_results["Feature selection"] = pd.Categorical(
    all_comparison_results["Feature selection"],
    categories=selection_order,
    ordered=True,
)

grouped_ranking = (
    all_comparison_results.sort_values(
        ["Feature selection"] + PRIMARY_SORT_METRICS + ["Model"],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

print("Validation ranking within each feature-selection method:")
display(grouped_ranking)

best_per_selection = (
    grouped_ranking
    .groupby("Feature selection", observed=True, sort=False)
    .head(1)
    .sort_values(
        PRIMARY_SORT_METRICS + ["Model"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

print("Best validation model from each feature-selection method:")
display(best_per_selection)


# %% [markdown]
# ### 14.1 Selecting the Top Three Finalists (Deduplicating Identical Sensor Subsets)
# 
# When selecting the Top 3 candidates from the screening table to advance to hyperparameter fine-tuning, we must handle a common issue in small-sensor datasets: **different feature-selection algorithms often pick the exact same subset of sensors** (for example, ANOVA may return `Air temp, Process temp, RPM, Torque` in column order, while Pearson correlation returns the exact same four sensors sorted by correlation strength: `Torque, RPM, Air temp, Process temp`).
# 
# If we only deduplicate by method name (`Feature selection`), the same model architecture trained on the exact same sensor columns would occupy multiple finalist slots as identical clones. To make finalist selection fair and meaningful, we standardize the feature names alphabetically (`Canonical features`) and apply the following rules:
# 
# 1. **Merge Identical Clones (Same Model + Same Sensor Subset):** If two or more feature-selection methods select the exact same sensors for the same model, they are merged into a single entry (e.g., `ANOVA / Pearson correlation`) so we never waste time fine-tuning identical clones.
# 2. **Allow Different Models on the Same Sensor Subset:** Different model families (e.g., `Gradient Boosting` and `Random Forest`) using the same feature subset (such as `No feature selection`) are kept as separate candidates.
# 3. **Configurable Architecture Diversity (`UNIQUE_FINALIST_MODELS` in Section 7):**
#    - When `UNIQUE_FINALIST_MODELS = False`, the **same model family** can appear more than once in the Top 3 if it performs strongly on **genuinely different sensor subsets** (e.g., `Random Forest` with all 6 sensors vs. `Random Forest` with 4 sensors).
#    - When `UNIQUE_FINALIST_MODELS = True`, the Top 3 finalists are forced to be **three distinct model architectures**, each paired with its best-performing feature subset.

# %%
# 1. Create a canonical (alphabetically sorted) feature string so that
#    identical sensor subsets in different column orders are recognized as the same set.
all_comparison_results["Canonical features"] = all_comparison_results["Features"].apply(
    lambda s: s if "components" in str(s) else ", ".join(sorted([f.strip() for f in str(s).split(",")]))
)

# 2. Merge rows that have the exact same (Model + Canonical features)
unique_combinations = (
    all_comparison_results
    .sort_values(
        PRIMARY_SORT_METRICS + ["Feature selection", "Model"],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    .groupby(["Model", "Canonical features"], as_index=False, sort=False)
    .agg({
        # Combine selector names that picked the exact same subset (e.g., "ANOVA / Pearson correlation")
        "Feature selection": lambda s: " / ".join(dict.fromkeys(s.astype(str))),
        "Features": "first",
        "Validation accuracy": "first",
        "Validation balanced accuracy": "first",
        "Validation failure precision": "first",
        "Validation failure recall": "first",
        "Validation failure F1": "first",
        "Validation missed failures": "first",
        "Validation false alarms": "first",
    })
)

# 3. Select the Top 3 finalists based on the Master Config setting
dedup_cols = ["Model"] if UNIQUE_FINALIST_MODELS else ["Model", "Canonical features"]

top_three_candidates = (
    unique_combinations
    .sort_values(
        PRIMARY_SORT_METRICS + ["Model"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    .drop_duplicates(subset=dedup_cols, keep="first")
    .head(3)
    .reset_index(drop=True)
    .drop(columns=["Canonical features"])
)

print(f"Top 3 candidates advancing to Hyperparameter Fine-Tuning (UNIQUE_FINALIST_MODELS={UNIQUE_FINALIST_MODELS}):")
display(top_three_candidates)

# %% [markdown]
# ## --- IGNORE ---
# ## 15. NEW Stacking Ensemble Experiment (Standalone Validation Check) BELOW AT 17
# 
# - **What:** We take the Top 3 distinct models from screening and combine them into a single `StackingClassifier` using Logistic Regression as the meta-model (`final_estimator`).
# - **Why:** Stacking is the third ensemble paradigm in our syllabus (alongside Bagging and Boosting). We run this as a standalone validation experiment to see if combining the Top 3 models improves upon the #1 individual model. However, for simplicity and interpretability, only the Top 3 individual models advance to hyperparameter fine-tuning and final selection.

# %% [markdown]
#  ## 16. Fine-Tune the Top Three Finalists (5-Fold Cross-Validation)
# 
# 
# 
#  Screening evaluated models using default settings. Now we **fine-tune** the Top 3 finalists using 5-Fold Stratified Cross-Validation (`GridSearchCV`) on the Training set (`x_train`).
# 
#  - **Why this counts as fine-tuning:** Instead of retraining the same configuration, `GridSearchCV` systematically searches internal structural hyperparameters (such as SVM kernel type Linear/Poly/RBF, regularization $C$, tree depth, learning rate, and MLP hidden layer sizes) across 5 training/validation folds to find the optimal mathematical settings.
# 
# 
# 
#  | Split | Purpose |
#  |---|---|
#  | **Training (70%)** | Fit models and run 5-fold cross-validation hyperparameter tuning |
#  | **Validation (15%)** | Screen combinations and choose the single best tuned winner |
#  | **Test (15%)** | Untouched final exam — used once at the very end |

# %%
original_features = list(x_train.columns)

def make_tuning_model(model_name):
    """Return (model, parameter_grid) with syllabus-aligned hyperparameter grids."""
    if model_name == "SVC":
        model = make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=RANDOM_STATE),
        )
        grid = [
            {
                "svc__kernel": ["linear"],
                "svc__C": [0.1, 1, 10],
                "svc__class_weight": [None, "balanced"],
                "svc__probability": [True],
            },
            {
                "svc__kernel": ["rbf"],
                "svc__C": [0.1, 1, 10],
                "svc__gamma": ["scale", "auto"],
                "svc__class_weight": [None, "balanced"],
                "svc__probability": [True],
            },
            {
                "svc__kernel": ["poly"],
                "svc__C": [0.1, 1, 10],
                "svc__degree": [2, 3],
                "svc__class_weight": [None, "balanced"],
                "svc__probability": [True],
            },
        ]
    elif model_name == "Logistic Regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        )
        grid = [
            {
                "logisticregression__penalty": ["l1"],
                "logisticregression__solver": ["liblinear"],
                "logisticregression__C": [0.01, 0.1, 1, 10],
                "logisticregression__class_weight": [None, "balanced"],
            },
            {
                "logisticregression__penalty": ["l2"],
                "logisticregression__solver": ["lbfgs"],
                "logisticregression__C": [0.01, 0.1, 1, 10],
                "logisticregression__class_weight": [None, "balanced"],
            },
        ]
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 10],
            "class_weight": [None, "balanced"],
        }
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 5],
            "class_weight": [None, "balanced"],
        }
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
        }
    elif model_name == "MLP":
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
        )
        grid = {
            "mlpclassifier__hidden_layer_sizes": [
                (32,), (64,), (128,),
                (32, 16), (64, 16), (64, 32),
            ],
            "mlpclassifier__alpha": [0.0001, 0.001, 0.01],
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, grid

def candidate_input_features(candidate):
    if candidate["Feature selection"] == "PCA":
        return original_features
    return [feature.strip() for feature in str(candidate["Features"]).split(",")]

def build_tuning_model(model_name, use_pca):
    model, parameter_grid = make_tuning_model(model_name)
    if not use_pca:
        return model, parameter_grid

    pca_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)),
        ("model", model),
    ])
    if isinstance(parameter_grid, list):
        pca_grid = [
            {f"model__{key}": value for key, value in grid.items()}
            for grid in parameter_grid
        ]
    else:
        pca_grid = {
            f"model__{key}": value for key, value in parameter_grid.items()
        }
    return pca_model, pca_grid

cv_strategy = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

tuned_models = []
for index, candidate in top_three_candidates.iterrows():
    input_features = candidate_input_features(candidate)
    model, parameter_grid = build_tuning_model(
        candidate["Model"],
        candidate["Feature selection"] == "PCA",
    )
    print(
        f"Tuning candidate {index + 1}/{len(top_three_candidates)}: "
        f"{candidate['Model']} ({candidate['Feature selection']})"
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring={
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit="f1",
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(x_train[input_features], y_train)

    best_index = search.best_index_
    tuned_models.append({
        "Feature selection": candidate["Feature selection"],
        "Model": candidate["Model"],
        "Features": candidate["Features"],
        "Input features": input_features,
        "Search": search,
        "Best parameters": search.best_params_,
        "CV F1 mean": search.cv_results_["mean_test_f1"][best_index],
        "CV F1 std": search.cv_results_["std_test_f1"][best_index],
        "CV recall mean": search.cv_results_["mean_test_recall"][best_index],
        "CV precision mean": search.cv_results_["mean_test_precision"][best_index],
        "CV balanced accuracy mean": search.cv_results_["mean_test_balanced_accuracy"][best_index],
        "CV train F1 mean": search.cv_results_["mean_train_f1"][best_index],
        "Number of grid combinations": len(search.cv_results_["params"]),
    })

tuning_results = pd.DataFrame(tuned_models).drop(columns=["Search"]).sort_values(
    ["CV F1 mean", "CV recall mean", "CV precision mean", "Model"],
    ascending=[False, False, False, True],
    kind="mergesort",
).reset_index(drop=True)

print("Cross-validation tuning results on training data:")
display(tuning_results)


# %%
def evaluate_on_split(model, metadata, features, X_split, y_split, prefix, display_features=None):
    """Evaluate one fitted model on a named data split."""
    predictions = model.predict(X_split[features])
    report = classification_report(
        y_split,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(
        y_split,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "Feature selection": metadata["Feature selection"],
        "Model": metadata["Model"],
        "Features": (
            ", ".join(display_features)
            if isinstance(display_features, list)
            else str(display_features)
        )
        if display_features is not None
        else ", ".join(features),
        f"{prefix} F1": report["1"]["f1-score"],
        f"{prefix} recall": report["1"]["recall"],
        f"{prefix} precision": report["1"]["precision"],
        f"{prefix} balanced accuracy": balanced_accuracy_score(y_split, predictions),
        f"{prefix} missed failures": fn,
        f"{prefix} false alarms": fp,
        "Predictions": predictions,
    }

# Evaluate every tuned finalist on the validation split
validation_finalist_rows = []
for result in tuned_models:
    metadata = {
        "Feature selection": result["Feature selection"],
        "Model": result["Model"],
    }
    validation_finalist_rows.append(
        evaluate_on_split(
            result["Search"].best_estimator_,
            metadata,
            result["Input features"],
            x_val,
            y_val,
            "Validation",
            result["Features"],
        )
    )

validation_finalist_results = pd.DataFrame(
    validation_finalist_rows
).sort_values(
    ["Validation F1", "Validation recall", "Validation precision", "Model"],
    ascending=[False, False, False, True],
    kind="mergesort",
).reset_index(drop=True)

print("Tuned finalists compared on the Validation split:")
display(validation_finalist_results.drop(columns=["Predictions"]))


# %% [markdown]
#  ### 16.1 Select the Final Tuned Winner
# 
# 
# 
#  We select and freeze the winning model, feature set, and hyperparameters using **Validation metrics only**. The test set is still untouched.

# %%
final_winner = validation_finalist_results.iloc[0]
final_result = next(
    result for result in tuned_models
    if result["Model"] == final_winner["Model"]
    and result["Feature selection"] == final_winner["Feature selection"]
)
final_metadata = {
    "Feature selection": final_result["Feature selection"],
    "Model": final_result["Model"],
}

print(
    f"Selected winner: {final_winner['Model']} "
    f"({final_winner['Feature selection']})"
)
print("Selected hyperparameters:", final_result["Best parameters"])


# %% [markdown]
#  ### 16.2 Refit on the 85% Development Data
# 
# 
# 
#  Now that the winning architecture, feature subset, and hyperparameters are frozen, we combine the Training (70%) and Validation (15%) splits into an 85% development set and refit the winner once so it learns from all available non-test data.

# %%
x_development = pd.concat([x_train, x_val], axis=0)
y_development = pd.concat([y_train, y_val], axis=0)

final_estimator = clone(final_result["Search"].best_estimator_)
final_estimator.fit(
    x_development[final_result["Input features"]],
    y_development,
)

print(f"Final model refit on {len(x_development)} development samples.")
# %% [markdown]
# ## 16.3 Robustness check on the DEVELOPMENT set only (no test data)
# Run this cell AFTER 16.2 and BEFORE Section 17. It never touches x_test / y_test.
# Purpose: (1) answer "does dropping Type / dropping sensors hurt?" with a noise estimate,
# (2) answer the advisor's k-sensitivity and PCA-99% questions.
# Each seed re-splits the 8,500-row development set into 7,000 train / 1,500 validation.

# %%
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

ROBUST_SEEDS = [11, 22, 33, 44, 66]      # must NOT contain RANDOM_STATE
K_VALUES = [3, 4, 5]
assert RANDOM_STATE not in ROBUST_SEEDS

x_dev = pd.concat([x_train, x_val], axis=0)
y_dev = pd.concat([y_train, y_val], axis=0)
ALL_FEATURES = list(x_dev.columns)
NO_TYPE = [c for c in ALL_FEATURES if c != "Type"]


def screening_models(seed):
    """Same six classifiers / settings as Section 8 (SVC probability=False only for speed;
    predict() is unaffected by that flag)."""
    return {
        "SVC": make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=1000, random_state=seed)),
    }


def pick_subsets(xtr, ytr, k, seed):
    """Return {method: [features]} for the six selection methods at size k (fit on xtr only)."""
    out = {}
    skb = SelectKBest(f_classif, k=k).fit(xtr, ytr)
    out["ANOVA"] = xtr.columns[skb.get_support()].tolist()
    corr = xtr.apply(lambda c: abs(np.corrcoef(c, ytr)[0, 1])).sort_values(ascending=False)
    out["Pearson"] = corr.head(k).index.tolist()
    lasso = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=2000, random_state=seed)).fit(xtr, ytr)
    coef = np.abs(lasso[-1].coef_).ravel()
    out["Lasso"] = xtr.columns[np.argsort(coef)[::-1][:k]].tolist()
    xs = pd.DataFrame(StandardScaler().fit_transform(xtr), columns=xtr.columns, index=xtr.index)
    rfe = RFE(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed), n_features_to_select=k).fit(xs, ytr)
    out["RFE"] = xtr.columns[rfe.get_support()].tolist()
    for direction, name in (("forward", "Forward SFS"), ("backward", "Backward SFS")):
        sfs = SequentialFeatureSelector(
            DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            n_features_to_select=k, direction=direction, scoring="f1", cv=CV_FOLDS, n_jobs=-1,
        ).fit(xtr, ytr)
        out[name] = xtr.columns[sfs.get_support()].tolist()
    return out


def score_row(seed, setting, n_feat, model_name, y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "seed": seed, "setting": setting, "n_features": n_feat, "model": model_name,
        "F1": f1_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "missed": fn, "false_alarms": fp,
    }


rows = []
t0 = time.time()
for seed in ROBUST_SEEDS:
    xtr, xva, ytr, yva = train_test_split(x_dev, y_dev, test_size=1500, random_state=seed, stratify=y_dev)

    # 1) collect UNIQUE feature subsets and remember which methods chose them
    subsets = {tuple(ALL_FEATURES): ["All 6"]}
    subsets.setdefault(tuple(NO_TYPE), []).append("Drop Type only")
    for k in K_VALUES:
        for method, feats in pick_subsets(xtr, ytr, k, seed).items():
            key = tuple(c for c in ALL_FEATURES if c in feats)          # canonical order
            subsets.setdefault(key, []).append(f"{method} k={k}")

    # 2) fit every classifier once per unique subset
    for feats, labels in subsets.items():
        feats = list(feats)
        for name, model in screening_models(seed).items():
            model.fit(xtr[feats], ytr)
            pred = model.predict(xva[feats])
            for lab in labels:
                rows.append(score_row(seed, lab, len(feats), name, yva, pred))

    # 3) PCA at 95% and 99% variance
    sc = StandardScaler().fit(xtr)
    for var in (0.95, 0.99):
        pca_k = PCA(n_components=var, random_state=seed).fit(sc.transform(xtr))
        ztr, zva = pca_k.transform(sc.transform(xtr)), pca_k.transform(sc.transform(xva))
        for name, model in screening_models(seed).items():
            model.fit(ztr, ytr)
            rows.append(score_row(seed, f"PCA {int(var*100)}%", int(pca_k.n_components_), name, yva, model.predict(zva)))
    print(f"seed {seed} done ({time.time() - t0:.0f}s)")

robust_df = pd.DataFrame(rows)
os.makedirs("./Report/export", exist_ok=True)
robust_df.to_csv("./Report/export/robustness_dev_only.csv", index=False)

# %%
# ---- Summary A: best F1 per classifier by number of retained input features (mean +- std over seeds)
# NOTE: the max over several subsets is optimistic for reduced sets, so "all 6 still wins" is a conservative result.
sel = robust_df[~robust_df["setting"].str.startswith("PCA")]
best = (sel.groupby(["seed", "model", "n_features"])["F1"].max().reset_index())
summaryA = best.groupby(["model", "n_features"])["F1"].agg(["mean", "std"]).round(3)
summaryA = summaryA["mean"].astype(str) + " ± " + summaryA["std"].astype(str)
summaryA = summaryA.unstack("n_features")
pcs = robust_df[robust_df["setting"].str.startswith("PCA")].groupby(["model", "setting"])["F1"].agg(["mean", "std"]).round(3)
pcs = (pcs["mean"].astype(str) + " ± " + pcs["std"].astype(str)).unstack("setting")
summaryA = summaryA.join(pcs)
print("A. Best validation F1 by number of input features (mean ± std over seeds):")
display(summaryA)

# ---- Summary B: paired effect of dropping Type only (5 sensors vs all 6 features)
piv = robust_df[robust_df["setting"].isin(["All 6", "Drop Type only"])].pivot_table(
    index=["seed", "model"], columns="setting", values="F1")
piv["delta (drop Type - all 6)"] = piv["Drop Type only"] - piv["All 6"]
summaryB = piv.groupby("model").agg(
    all6_mean=("All 6", "mean"), noType_mean=("Drop Type only", "mean"),
    delta_mean=("delta (drop Type - all 6)", "mean"), delta_std=("delta (drop Type - all 6)", "std"),
    seeds_noType_ge_all6=("delta (drop Type - all 6)", lambda s: int((s >= 0).sum())),
).round(3)
print("B. Dropping Type only (5 sensors) vs all 6 features, paired over seeds:")
display(summaryB)

# ---- Summary C: k = 4 methods, best classifier per seed (mean +- std)
m4 = robust_df[robust_df["setting"].str.endswith("k=4") | robust_df["setting"].isin(["All 6", "Drop Type only"]) | robust_df["setting"].str.startswith("PCA")]
bestm = m4.groupby(["seed", "setting"])["F1"].max().reset_index()
summaryC = bestm.groupby("setting")["F1"].agg(["mean", "std"]).round(3).sort_values("mean", ascending=False)
print("C. Best-classifier validation F1 per feature setting (mean ± std over seeds):")
display(summaryC)

with open("./Report/export/robustness_summary.txt", "w", encoding="utf-8") as f:
    f.write("A\n" + summaryA.to_string() + "\n\nB\n" + summaryB.to_string() + "\n\nC\n" + summaryC.to_string() + "\n")
print("Saved ./Report/export/robustness_dev_only.csv and robustness_summary.txt")

# %% [markdown]
# ## 16.4 Operating-point check on the VALIDATION split only (no test data)
# Answers advisor comment 2: are the two RQ2 "regimes" a property of the classifier,
# or of class weighting / the 0.5 threshold?
# Run AFTER Section 16.3 and BEFORE nothing else; it never touches x_test / y_test.
# It is diagnostic only: it does NOT change the winner, the finalists, or any reported test result.

# %%
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, average_precision_score

RS = RANDOM_STATE                      # 2077, same as the main run
Xtr, ytr = x_train, y_train            # all six input features, seed-2077 training split
Xva, yva = x_val, y_val                # seed-2077 validation split (51 failures)
w_bal = compute_sample_weight("balanced", ytr)

# (name, model, use balanced sample weights in fit?)
models = [
    ("GB default (as screened)", GradientBoostingClassifier(random_state=RS), False),
    ("GB + balanced weights",    GradientBoostingClassifier(random_state=RS), True),
    ("RF balanced, leaf 1 (as screened)", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RS, n_jobs=-1), False),
    ("RF balanced, leaf 5",      RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_leaf=5, random_state=RS, n_jobs=-1), False),
    ("SVC balanced (as screened)", make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RS)), False),
    ("LR balanced (as screened)",  make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS)), False),
]

def counts(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return tp, fp

rows = []
for name, m, weighted in models:
    if weighted:
        m.fit(Xtr, ytr, sample_weight=w_bal)
    else:
        m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    tp05, fa05 = counts(yva, p, 0.5)
    # highest threshold that still catches >= 41 of 51 failures (recall >= 0.80)
    best_t, fa80, tp80 = None, None, None
    for t in np.unique(p)[::-1]:
        tp, fa = counts(yva, p, t)
        if tp >= 41:
            best_t, fa80, tp80 = t, fa, tp
            break
    rows.append({
        "model": name,
        "caught@0.5": f"{tp05}/51", "false_alarms@0.5": fa05,
        "threshold_for_recall>=0.80": None if best_t is None else round(float(best_t), 3),
        "caught@that_threshold": None if tp80 is None else f"{tp80}/51",
        "false_alarms@that_threshold": fa80,
        "val_PR_AUC": round(average_precision_score(yva, p), 3),
    })

thr_df = pd.DataFrame(rows)
display(thr_df)
os.makedirs("./Report/export", exist_ok=True)
thr_df.to_csv("./Report/export/threshold_check_val.csv", index=False)
print("Saved ./Report/export/threshold_check_val.csv  (validation split only; test set untouched)")

# %% [markdown]
# ## 17. Final Test Evaluation (Tuned Single Winner vs. Stacking Ensemble)
# 
# This is the **only** section that touches `y_test`. Both approaches were selected using Validation performance only and refitted on the **85% Development Set** (`x_development`, `8,500` rows):
# 1. **Selected Single-Model Winner:** The #1 fine-tuned individual classifier from Section 16.
# 2. **Selected Ensemble Benchmark (`Stacking Top 3`):** Combines our 3 fine-tuned finalist architectures with a Logistic Regression meta-model.
# 
# We evaluate both on the unseen 15% Test Set (`1,500` rows) to compare single-model deployment against multi-model stacking on unseen data.

# %%
# 1. Refit the Stacking Ensemble on the 85% Development Set using the 3 tuned finalists
tuned_base_learners = [
    (res["Model"].lower().replace(" ", "_"), clone(res["Search"].best_estimator_))
    for res in tuned_models
]

final_stacking_estimator = StackingClassifier(
    estimators=tuned_base_learners,
    final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    cv=CV_FOLDS,
    n_jobs=-1,
)
final_stacking_estimator.fit(
    x_development[final_result["Input features"]],
    y_development,
)

# 2. Evaluate the #1 Tuned Single Winner on the unseen Test Set
final_test_result = evaluate_on_split(
    final_estimator,
    final_metadata,
    final_result["Input features"],
    x_test,
    y_test,
    "Test",
    final_result["Features"],
)

# 3. Evaluate the Stacking Ensemble on the unseen Test Set
stacking_test_metadata = {
    "Feature selection": final_result["Feature selection"],
    "Model": "Stacking (Tuned Top 3)",
}
stacking_test_result = evaluate_on_split(
    final_stacking_estimator,
    stacking_test_metadata,
    final_result["Input features"],
    x_test,
    y_test,
    "Test",
    final_result["Features"],
)

stacking_row = {
    "Feature selection": stacking_test_metadata["Feature selection"],
    "Model": stacking_test_metadata["Model"],
    "Features": final_result["Features"],
    "Test F1": stacking_test_result["Test F1"],
    "Test recall": stacking_test_result["Test recall"],
    "Test precision": stacking_test_result["Test precision"],
    "Test balanced accuracy": stacking_test_result["Test balanced accuracy"],
    "Test missed failures": stacking_test_result["Test missed failures"],
    "Test false alarms": stacking_test_result["Test false alarms"],
    "Predictions": stacking_test_result["Predictions"],
}

comparison_df = pd.DataFrame([stacking_row])

# 4. Display side-by-side Final Test Table
final_test_comparison_df = pd.DataFrame([
    {k: v for k, v in final_test_result.items() if k != "Predictions"},
    {k: v for k, v in stacking_test_result.items() if k != "Predictions"},
])
print("Final Unseen Test Set Evaluation (Single Winner vs. Stacking Ensemble):")
display(final_test_comparison_df)

# 5. Plot Confusion Matrices + Combined ROC Curve (1x3 grid)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Single Winner Confusion Matrix
cm_single = confusion_matrix(y_test, final_test_result["Predictions"], labels=[0, 1])
ConfusionMatrixDisplay(cm_single, display_labels=["No failure", "Failure"]).plot(
    cmap="Blues", values_format="d", ax=axes[0]
)
axes[0].set_title(f"Single Winner: {final_winner['Model']}\n({final_winner['Feature selection']})")
axes[0].grid(False)

# Plot 2: Stacking Ensemble Confusion Matrix
cm_stack = confusion_matrix(y_test, stacking_test_result["Predictions"], labels=[0, 1])
ConfusionMatrixDisplay(cm_stack, display_labels=["No failure", "Failure"]).plot(
    cmap="Greens", values_format="d", ax=axes[1]
)
axes[1].set_title(f"Ensemble: Stacking (Tuned Top 3)\n({final_winner['Feature selection']})")
axes[1].grid(False)

# Plot 3: Combined ROC Curves on the same axes
RocCurveDisplay.from_estimator(
    final_estimator,
    x_test[final_result["Input features"]],
    y_test,
    name=f"Single: {final_winner['Model']}",
    ax=axes[2],
)
RocCurveDisplay.from_estimator(
    final_stacking_estimator,
    x_test[final_result["Input features"]],
    y_test,
    name="Stacking (Tuned Top 3)",
    ax=axes[2],
)
axes[2].plot([0, 1], [0, 1], ls="--", color="grey", label="Random (AUC = 0.50)")
axes[2].set_title("ROC Curve Comparison — Final Test Set")
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.show()

# %%
# 
# 
#  OLD VERSION: Only evaluate the #1 Tuned Single Winner on the unseen Test Set
# 
# 
final_test_result = evaluate_on_split(
    final_estimator,
    final_metadata,
    final_result["Input features"],
    x_test,
    y_test,
    "Test",
    final_result["Features"],
)

print("Final unseen test result:")
display(
    pd.DataFrame([{
        key: value
        for key, value in final_test_result.items()
        if key != "Predictions"
    }])
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1) Confusion Matrix
cm = confusion_matrix(y_test, final_test_result["Predictions"], labels=[0, 1])
ConfusionMatrixDisplay(
    cm,
    display_labels=["No failure", "Failure"],
).plot(cmap="Blues", values_format="d", ax=axes[0])
axes[0].set_title(
    f"Confusion Matrix: {final_winner['Model']}\n({final_winner['Feature selection']})"
)
axes[0].grid(False)

# 2) ROC Curve & AUC
RocCurveDisplay.from_estimator(
    final_estimator,
    x_test[final_result["Input features"]],
    y_test,
    name=f"{final_winner['Model']} ({final_winner['Feature selection']})",
    ax=axes[1],
)
axes[1].plot([0, 1], [0, 1], ls="--", color="grey", label="Random (AUC = 0.50)")
axes[1].set_title("ROC Curve — Final Test Set")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()


# %% [markdown]
#  ## 18. Term Project Conclusion and Recommendations
# 
# 
# 
#  This project demonstrates a complete, leak-free supervised machine learning workflow:
# 
# 
# 
#  1. **Explore & Preprocess:** Inspected the AI4I 2020 dataset, visualized physical failure boundaries, encoded `Type`, and removed target-leaking failure mode columns.
# 
#  2. **Strict Data Partitioning:** Split the data into 70% Training, 15% Validation, and 15% Test sets.
# 
#  3. **Feature Selection & Dimensionality Reduction:** Compared Filter (ANOVA, Pearson), Adapted Lasso, Wrapper (RFE, Forward/Backward SFS), and PCA methods fitted strictly on the training set.
# 
#  4. **Candidate Screening & Stacking:** Screened 6 model families across all feature sets on the Validation split, selected the Top 3 distinct architectures, and tested a dynamic Stacking Ensemble.
# 
#  5. **Hyperparameter Fine-Tuning:** Tuned the Top 3 finalists using 5-Fold Stratified Cross-Validation on the training split.
# 
#  6. **Final Unseen Evaluation:** Refit the validation winner on the 85% development set and evaluated once on the 15% test set using F1, Recall, Precision, Balanced Accuracy, Confusion Matrix, and ROC-AUC.

# %% [markdown]
# ## 19. Experiment Run Logger & Summary Exporter
# 
# Run the cell below every time you test a new configuration in **Section 7 (Master Experiment Configuration)**. It automatically:
# 1. **Appends** your current configuration, selected feature sets, Top 3 screening finalists, Stacking comparison, tuned winner, and final Test metrics to a new dated text log (`experiment_runs_log_2026-11-03.txt`) so prior runs stay separate.
# 2. **Generates** a clean, readable Markdown report (`latest_experiment_report.md`) and a multi-page **PDF summary (`experiment_summary.pdf`)** with comparison tables and charts so your team can easily compare approaches and pick the best one for your class presentation.

# %%
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 1. Build a clean text/markdown summary of the current configuration & results
run_summary_lines = [
    "=" * 80,
    f"EXPERIMENT RUN LOG — {timestamp}",
    "=" * 80,
    "",
    "1. MASTER CONFIGURATION SETTINGS:",
    f"   - RANDOM_STATE           : {RANDOM_STATE}",
    f"   - NUMBER_OF_FEATURES     : {NUMBER_OF_FEATURES}",
    f"   - RFE_SELECTOR_MODEL     : {RFE_SELECTOR_MODEL}",
    f"   - SFS_SELECTOR_MODEL     : {SFS_SELECTOR_MODEL}",
    f"   - PCA_VARIANCE           : {PCA_VARIANCE} ({pca.n_components_} components kept)",
    f"   - CV_FOLDS               : {CV_FOLDS}",
    f"   - UNIQUE_FINALIST_MODELS : {UNIQUE_FINALIST_MODELS}",
    "",
    "2. FEATURES SELECTED BY EACH METHOD:",
    f"   - Baseline (All 6)       : {', '.join(x_train.columns)}",
    f"   - ANOVA                  : {', '.join(anova_features)}",
    f"   - Pearson Correlation    : {', '.join(correlation_features)}",
    f"   - Adapted Lasso (L1)     : {', '.join(lasso_features)}",
    f"   - RFE ({RFE_SELECTOR_MODEL}) : {', '.join(rfe_features)}",
    f"   - Forward SFS ({SFS_SELECTOR_MODEL}) : {', '.join(forward_features)}",
    f"   - Backward SFS ({SFS_SELECTOR_MODEL}): {', '.join(backward_features)}",
    f"   - PCA ({PCA_VARIANCE*100:.0f}% var)        : {pca.n_components_} orthogonal components",
    "",
    "3. BEST MODEL PER FEATURE-SELECTION METHOD (VALIDATION SCREENING):",
    best_per_selection[[
        "Feature selection", "Model", "Validation failure F1",
        "Validation failure recall", "Validation failure precision",
        "Validation missed failures", "Validation false alarms"
    ]].to_string(index=False),
    "",
    "4. TOP 3 FINALISTS AFTER 5-FOLD CV HYPERPARAMETER TUNING (VALIDATION):",
    validation_finalist_results.drop(columns=["Predictions"])[[
        "Feature selection", "Model", "Validation F1",
        "Validation recall", "Validation precision",
        "Validation missed failures", "Validation false alarms"
    ]].to_string(index=False),
    "",
    "5. FINAL TEST COMPARISON (SINGLE WINNER VS. STACKING):",
    final_test_comparison_df.to_string(index=False),
    "",
    "6. SELECTED WINNER & TUNED HYPERPARAMETERS:",
    f"   - Winning Model          : {final_winner['Model']}",
    f"   - Winning Feature Method : {final_winner['Feature selection']} ({final_result['Features']})",
    f"   - Best Hyperparameters   : {final_result['Best parameters']}",
    "",
    "7. FINAL UNSEEN TEST SET PERFORMANCE (SINGLE WINNER VS. STACKING):",
    final_test_comparison_df.to_string(index=False),
    "=" * 80,
    "\n"
    f"   - Test Failure F1        : {final_test_result['Test F1']:.4f}",
    f"   - Test Failure Recall    : {final_test_result['Test recall']:.4f}",
    f"   - Test Failure Precision : {final_test_result['Test precision']:.4f}",
    f"   - Test Balanced Accuracy : {final_test_result['Test balanced accuracy']:.4f}",
    f"   - Test Missed Failures   : {final_test_result['Test missed failures']}",
    f"   - Test False Alarms      : {final_test_result['Test false alarms']}",
    "=" * 80,
    "\n"
]

report_text = "\n".join(run_summary_lines)

# 2. Append to cumulative text log (keeps history of all configurations you test!)
log_txt_path = "./Report/experiment_runs_log_2026-11-03.txt"
with open(log_txt_path, "a", encoding="utf-8") as f:
    f.write(report_text)

# 3. Save a clean single-run Markdown report for easy reading in VS Code
md_path = "./Report/latest_experiment_report.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Experiment Summary Report ({timestamp})\n\n```text\n{report_text}\n```\n")

print(f"Saved cumulative log to : {log_txt_path}")
print(f"Saved Markdown report to: {md_path}")
# print(f"Saved PDF report to     : {pdf_path}\n")
print(report_text)

# %% [markdown]
# ## 20. Full Results Export for the Paper (run ONCE, after Section 17)
# Saves every number and grayscale figure needed for the AUCC manuscript.
# Nothing here re-tunes or re-selects anything; it only reads existing variables.

# %%
import json
import platform
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = "./Report/export"
os.makedirs(OUT, exist_ok=True)
report = []

def add(title, obj):
    report.append("=" * 80)
    report.append(title)
    report.append("=" * 80)
    report.append(obj.to_string() if hasattr(obj, "to_string") else str(obj))
    report.append("")

# 0. Configuration + environment
config = {
    "RANDOM_STATE": RANDOM_STATE, "NUMBER_OF_FEATURES": NUMBER_OF_FEATURES,
    "RFE_SELECTOR_MODEL": RFE_SELECTOR_MODEL, "SFS_SELECTOR_MODEL": SFS_SELECTOR_MODEL,
    "PCA_VARIANCE": PCA_VARIANCE, "PCA components kept": int(pca.n_components_),
    "CV_FOLDS": CV_FOLDS, "UNIQUE_FINALIST_MODELS": UNIQUE_FINALIST_MODELS,
    "python": platform.python_version(), "scikit-learn": sklearn.__version__,
    "numpy": np.__version__, "pandas": pd.__version__,
}
add("0. CONFIGURATION & ENVIRONMENT", json.dumps(config, indent=2))

# 1. Split sizes and class counts
split_table = pd.DataFrame({
    "Training": y_train.value_counts(),
    "Validation": y_val.value_counts(),
    "Test": y_test.value_counts(),
}).fillna(0).astype(int)
split_table.loc["Total rows"] = split_table.sum()
split_table["Development (train+val)"] = split_table["Training"] + split_table["Validation"]
add("1. SPLIT SIZES (row 0 = normal, row 1 = failure)", split_table)

# 2. Per-sensor selector scores (explains why Process temp / Type are dropped)
feature_scores = pd.DataFrame({
    "ANOVA F": anova_filter.scores_,
    "ANOVA p-value": anova_filter.pvalues_,
    "abs Pearson r": correlation_scores.reindex(x_train.columns).values,
    "abs L1 coef (C=0.1)": lasso_coefficients,
}, index=x_train.columns).round(5)
add("2. PER-SENSOR SELECTOR SCORES (training split)", feature_scores)

# 3. Feature-subset matrix (Table 1 of the paper)
selected = {
    "ANOVA": anova_features,
    "Pearson": correlation_features,
    "Adapted Lasso": lasso_features,
    f"RFE ({RFE_SELECTOR_MODEL})": rfe_features,
    f"Forward SFS ({SFS_SELECTOR_MODEL})": forward_features,
    f"Backward SFS ({SFS_SELECTOR_MODEL})": backward_features,
}
subset_matrix = pd.DataFrame(
    {name: ["Y" if f in feats else "-" for f in x_train.columns] for name, feats in selected.items()},
    index=x_train.columns,
)
add("3. SELECTED SENSORS PER METHOD (Y = kept)", subset_matrix)
add("3b. RFE with all four base estimators", pd.DataFrame(
    {k: ", ".join(v) for k, v in rfe_selected_features.items()}, index=["selected"]).T)

# 4. PCA variance and loadings
pc_names = [f"PC{i + 1}" for i in range(pca.n_components_)]
add("4a. PCA explained variance (all 6 components)", pd.DataFrame({
    "explained": pca_full.explained_variance_ratio_,
    "cumulative": cum,
}, index=[f"PC{i + 1}" for i in range(len(cum))]).round(4))
add("4b. PCA loadings of the kept components", pd.DataFrame(
    pca.components_, columns=x_train.columns, index=pc_names).round(3))

# 5. Full screening table: all 9 strategies x 6 models (54 rows)
screen_cols = [
    "Feature selection", "Model", "Features", "Validation failure F1",
    "Validation failure recall", "Validation failure precision",
    "Validation balanced accuracy", "Validation missed failures", "Validation false alarms",
]
full_screening = grouped_ranking[screen_cols]
add("6a. FULL SCREENING TABLE (all strategies x all models)", full_screening)
add("6b. BEST MODEL PER FEATURE STRATEGY", best_per_selection[screen_cols])
add("6c. TOP 3 CANDIDATES SENT TO TUNING (pre-tuning validation scores)",
    top_three_candidates[[c for c in screen_cols if c in top_three_candidates.columns]])
full_screening.to_csv(f"{OUT}/screening_all_48.csv", index=False)

# 7. Tuning results (exact hyperparameters, CV mean/std, train-vs-CV gap)
tune_view = tuning_results[[
    "Feature selection", "Model", "Best parameters", "CV F1 mean", "CV F1 std",
    "CV recall mean", "CV precision mean", "CV balanced accuracy mean",
    "CV train F1 mean", "Number of grid combinations",
]]
add("7a. 5-FOLD CV TUNING ON TRAINING SPLIT", tune_view)
add("7b. EXACT BEST PARAMETERS (copy into Table 3)", json.dumps(
    {r["Model"]: r["Best parameters"] for r in tuned_models}, indent=2, default=str))
add("7c. TUNED FINALISTS ON VALIDATION SPLIT",
    validation_finalist_results.drop(columns=["Predictions"]))

# 8. Final test evaluation: confusion counts, AUC, average precision
Xt = x_test[final_result["Input features"]]
test_rows, cms = [], {}
for name, est, res in [
    (f"{final_winner['Model']} (winner)", final_estimator, final_test_result),
    ("Stacking (Tuned Top 3)", final_stacking_estimator, stacking_test_result),
]:
    proba = est.predict_proba(Xt)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, res["Predictions"], labels=[0, 1]).ravel()
    cms[name] = (tn, fp, fn, tp)
    test_rows.append({
        "Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "F1": res["Test F1"], "Recall": res["Test recall"], "Precision": res["Test precision"],
        "Balanced acc": res["Test balanced accuracy"],
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Average precision (PR-AUC)": average_precision_score(y_test, proba),
    })
add("8. FINAL TEST SET (threshold 0.5)", pd.DataFrame(test_rows).round(4))
add("8b. Classification report: winner", classification_report(
    y_test, final_test_result["Predictions"], digits=4, zero_division=0))
add("8c. Classification report: stacking", classification_report(
    y_test, stacking_test_result["Predictions"], digits=4, zero_division=0))

# Save one text file to paste back
full_text = "\n".join(report)
with open(f"{OUT}/full_results_for_paper.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
print(full_text)
print(f"\nSaved to {OUT}/full_results_for_paper.txt")

# %% [markdown]
# ### 20.1 Grayscale figures (replaces the earlier 20.1 cell)
# Fixes: clipped x-label, overlapping ROC axis labels, leftover seaborn grid/frame.

# %%
sns.reset_orig()
plt.style.use("default")
plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})

# ---------- Fig 2: best validation F1 per strategy (single column, ~82 mm) ----------
rename = {
    "No feature selection": "All 6 sensors",
    "Pearson correlation": "Pearson",
    "Forward selection": "Forward SFS",
    "Backward selection": "Backward SFS",
    "PCA": "PCA (4 comp.)",
}
bp = best_per_selection.sort_values("Validation failure F1")
labels = [rename.get(s, s) for s in bp["Feature selection"].astype(str)]
vals = bp["Validation failure F1"].values

fig, ax = plt.subplots(figsize=(3.2, 2.5))
ax.barh(labels, vals, color="0.6", edgecolor="black", linewidth=0.8)
for i, v in enumerate(vals):
    ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlim(0, 0.9)
ax.set_xlabel("Validation failure F1")
fig.savefig(f"{OUT}/fig2_validation_f1.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- Fig 3: confusion matrices + ROC (two columns, ~172 mm) ----------
Xt = x_test[final_result["Input features"]]
fig, axes = plt.subplots(
    1, 3, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.15]},
    constrained_layout=True,
)

short_names = ["Gradient Boosting", "Stacking"]
for ax, (name, (tn, fp, fn, tp)), title in zip(axes[:2], cms.items(), short_names):
    ConfusionMatrixDisplay(
        np.array([[tn, fp], [fn, tp]]), display_labels=["Normal", "Failure"]
    ).plot(cmap="Greys", values_format="d", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

roc_ax = axes[2]
RocCurveDisplay.from_estimator(
    final_estimator, Xt, y_test, ax=roc_ax, color="black", linestyle="-",
    name="Gradient Boosting")
RocCurveDisplay.from_estimator(
    final_stacking_estimator, Xt, y_test, ax=roc_ax, color="0.45", linestyle="--",
    name="Stacking")
roc_ax.plot([0, 1], [0, 1], ls=":", color="0.6", label="Chance")
roc_ax.set_xlabel("False positive rate")
roc_ax.set_ylabel("True positive rate")
roc_ax.set_title("ROC curves", fontsize=8)
roc_ax.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=1)

fig.savefig(f"{OUT}/fig3_test_results.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
