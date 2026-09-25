import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(BASE_DIR, "titanic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD SAVED TITANIC DATASET
# ============================================================

print("=" * 70)
print("MODULE 2 - PART B: MODELING")
print("=" * 70)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "titanic.csv not found. Run 01_eda.py first."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded from saved titanic.csv")
print("Shape:", df.shape)


# ============================================================
# 3. CLASSIFICATION DATA
# ============================================================

TARGET = "survived"

FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex",
    "embarked",
]

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nClass balance in full dataset:")
print(y.value_counts(normalize=True).sort_index())


# Save class balance used for modeling
class_balance = pd.DataFrame(
    {
        "class": y.value_counts().sort_index().index,
        "count": y.value_counts().sort_index().values,
        "percentage": (
            y.value_counts(normalize=True)
            .sort_index()
            .values
            * 100
        ),
    }
)

class_balance.to_csv(
    os.path.join(METRIC_DIR, "modeling_class_balance.csv"),
    index=False,
)


# ============================================================
# 5. PREPROCESSING
# ============================================================

NUMERIC_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

CATEGORICAL_FEATURES = [
    "sex",
    "embarked",
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ]
)


# ============================================================
# 6. THREE CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),
}


fitted_pipelines = {}
classification_results = []


# ============================================================
# 7. TRAIN + EVALUATE CLASSIFIERS
# ============================================================

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print("Training:", model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("AUC      :", round(auc, 4))

    print("Confusion Matrix:")
    print(cm)

    classification_results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }
    )

    fitted_pipelines[model_name] = pipeline


classification_results_df = pd.DataFrame(
    classification_results
)

classification_results_df.to_csv(
    os.path.join(
        METRIC_DIR,
        "classification_comparison.csv",
    ),
    index=False,
)


# ============================================================
# 8. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4),
)

for ax, (model_name, pipeline) in zip(
    axes,
    fitted_pipelines.items(),
):

    predictions = pipeline.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    ax.imshow(cm)

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Not Survived", "Survived"]
    )

    ax.set_yticklabels(
        ["Not Survived", "Survived"]
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "classification_confusion_matrices.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 9. ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, pipeline in fitted_pipelines.items():

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc:.3f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random baseline",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Classification Models")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "classification_roc_curves.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 10. DECISION TREE VISUALIZATION
# ============================================================

decision_tree_pipeline = fitted_pipelines[
    "Decision Tree"
]

tree_preprocessor = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
)

tree_classifier = (
    decision_tree_pipeline
    .named_steps["classifier"]
)

feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

plt.figure(figsize=(22, 12))

plot_tree(
    tree_classifier,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived",
    ],
    filled=True,
    rounded=True,
    fontsize=8,
)

plt.title("Decision Tree Classifier")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "decision_tree.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 11. CLASS IMBALANCE - BASELINE
# ============================================================

imbalance_results = []


baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train,
)

baseline_pred = baseline_pipeline.predict(
    X_test
)

imbalance_results.append(
    {
        "method": "Baseline",
        "precision": precision_score(
            y_test,
            baseline_pred,
        ),
        "recall": recall_score(
            y_test,
            baseline_pred,
        ),
        "f1": f1_score(
            y_test,
            baseline_pred,
        ),
    }
)


# ============================================================
# 12. CLASS WEIGHT = BALANCED
# ============================================================

balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

balanced_pipeline.fit(
    X_train,
    y_train,
)

balanced_pred = balanced_pipeline.predict(
    X_test
)

imbalance_results.append(
    {
        "method": "Class Weight Balanced",
        "precision": precision_score(
            y_test,
            balanced_pred,
        ),
        "recall": recall_score(
            y_test,
            balanced_pred,
        ),
        "f1": f1_score(
            y_test,
            balanced_pred,
        ),
    }
)


# ============================================================
# 13. SMOTE - TRAINING DATA ONLY
# ============================================================

smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "smote",
            SMOTE(
                random_state=42,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

smote_pipeline.fit(
    X_train,
    y_train,
)

smote_pred = smote_pipeline.predict(
    X_test
)

imbalance_results.append(
    {
        "method": "SMOTE",
        "precision": precision_score(
            y_test,
            smote_pred,
        ),
        "recall": recall_score(
            y_test,
            smote_pred,
        ),
        "f1": f1_score(
            y_test,
            smote_pred,
        ),
    }
)


imbalance_df = pd.DataFrame(
    imbalance_results
)

imbalance_df.to_csv(
    os.path.join(
        METRIC_DIR,
        "imbalance_comparison.csv",
    ),
    index=False,
)

print("\nImbalance comparison:")
print(imbalance_df)


# ============================================================
# 14. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "-" * 70)
print("Random Forest GridSearchCV")

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

param_grid = {
    "classifier__n_estimators": [
        100,
        200,
    ],
    "classifier__max_depth": [
        None,
        5,
        10,
    ],
    "classifier__max_features": [
        "sqrt",
        "log2",
    ],
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid_search.fit(
    X_train,
    y_train,
)

best_rf_pipeline = grid_search.best_estimator_

best_rf_classifier = (
    best_rf_pipeline
    .named_steps["classifier"]
)

best_params = grid_search.best_params_
oob_score = best_rf_classifier.oob_score_

print("\nBest Parameters:")
print(best_params)

print(
    "Best Cross Validation F1:",
    round(grid_search.best_score_, 4),
)

print(
    "OOB Score:",
    round(oob_score, 4),
)

grid_results = pd.DataFrame(
    [
        {
            "best_cv_f1": grid_search.best_score_,
            "oob_score": oob_score,
            "best_params": str(best_params),
        }
    ]
)

grid_results.to_csv(
    os.path.join(
        METRIC_DIR,
        "random_forest_gridsearch.csv",
    ),
    index=False,
)


# ============================================================
# 15. SAVE COMPLETE FITTED PIPELINE
# ============================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_classifier_pipeline.joblib",
)

joblib.dump(
    best_rf_pipeline,
    MODEL_FILE,
)

print("\nComplete pipeline saved:")
print(MODEL_FILE)


# ============================================================
# 16. RELOAD PIPELINE AND TEST RAW INPUT
# ============================================================

loaded_pipeline = joblib.load(
    MODEL_FILE
)

sample_raw_input = X_test.iloc[[0]]

original_prediction = best_rf_pipeline.predict(
    sample_raw_input
)

loaded_prediction = loaded_pipeline.predict(
    sample_raw_input
)

print("\nPipeline reload test:")
print(
    "Original prediction:",
    original_prediction,
)

print(
    "Reloaded prediction:",
    loaded_prediction,
)

print(
    "Predictions match:",
    np.array_equal(
        original_prediction,
        loaded_prediction,
    ),
)


# ============================================================
# 17. REGRESSION - PREDICT FARE
# ============================================================

print("\n" + "-" * 70)
print("Regression: Predict Fare")

REGRESSION_TARGET = "fare"

REGRESSION_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked",
]

X_reg = df[REGRESSION_FEATURES].copy()
y_reg = df[REGRESSION_TARGET].copy()


# Split first, before preprocessing
X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )
)


reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
]

reg_categorical_features = [
    "sex",
    "embarked",
]


reg_numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

reg_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_pipeline,
            reg_numeric_features,
        ),
        (
            "categorical",
            reg_categorical_pipeline,
            reg_categorical_features,
        ),
    ]
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor,
        ),
        (
            "regressor",
            LinearRegression(),
        ),
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train,
)

fare_predictions = regression_pipeline.predict(
    X_reg_test
)


# ============================================================
# 18. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_reg_test,
    fare_predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        fare_predictions,
    )
)

r2 = r2_score(
    y_reg_test,
    fare_predictions,
)

# Number of predictors after preprocessing
reg_feature_count = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

n = len(y_reg_test)
p = reg_feature_count

adjusted_r2 = (
    1
    - ((1 - r2) * (n - 1) / (n - p - 1))
)

print("\nRegression metrics:")
print("MAE        :", round(mae, 4))
print("RMSE       :", round(rmse, 4))
print("R2         :", round(r2, 4))
print("Adjusted R2:", round(adjusted_r2, 4))
print("Predictors after encoding:", p)


regression_results = pd.DataFrame(
    [
        {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2,
            "predictor_count_after_encoding": p,
        }
    ]
)

regression_results.to_csv(
    os.path.join(
        METRIC_DIR,
        "regression_metrics.csv",
    ),
    index=False,
)


# ============================================================
# 19. RESIDUAL PLOT
# ============================================================

residuals = (
    y_reg_test.values
    - fare_predictions
)

plt.figure(figsize=(8, 6))

plt.scatter(
    fare_predictions,
    residuals,
    alpha=0.6,
)

plt.axhline(
    0,
    linestyle="--",
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Plot - Fare Regression")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "fare_regression_residuals.png",
    ),
    dpi=150,
)

plt.close()


# ============================================================
# 20. FINAL CLASSIFIER COMPARISON
# ============================================================

final_comparison = classification_results_df.copy()

final_comparison.to_csv(
    os.path.join(
        METRIC_DIR,
        "final_classifier_comparison.csv",
    ),
    index=False,
)


# ============================================================
# 21. SAVE MODELING SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "modeling_summary.txt",
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "MODULE 2 - MODELING SUMMARY\n"
    )
    f.write(
        "============================\n\n"
    )

    f.write(
        "CLASSIFICATION FEATURES\n"
    )
    f.write(
        str(FEATURES)
        + "\n\n"
    )

    f.write(
        "TRAIN/TEST SPLIT\n"
    )
    f.write(
        "test_size=0.20, random_state=42, stratify=y\n"
    )
    f.write(
        "Stratification preserves the original "
        "survival class proportion in train and test sets.\n\n"
    )

    f.write(
        "CLASSIFICATION RESULTS\n"
    )
    f.write(
        classification_results_df
        .to_string(index=False)
    )
    f.write("\n\n")

    f.write(
        "IMBALANCE COMPARISON\n"
    )
    f.write(
        imbalance_df
        .to_string(index=False)
    )
    f.write("\n\n")

    f.write(
        "RANDOM FOREST GRID SEARCH\n"
    )
    f.write(
        f"Best parameters: {best_params}\n"
    )
    f.write(
        f"Best CV F1: {grid_search.best_score_:.4f}\n"
    )
    f.write(
        f"OOB score: {oob_score:.4f}\n\n"
    )

    f.write(
        "REGRESSION RESULTS\n"
    )
    f.write(
        f"MAE: {mae:.4f}\n"
    )
    f.write(
        f"RMSE: {rmse:.4f}\n"
    )
    f.write(
        f"R2: {r2:.4f}\n"
    )
    f.write(
        f"Adjusted R2: {adjusted_r2:.4f}\n"
    )
    f.write(
        f"Predictors after encoding: {p}\n\n"
    )

    f.write(
        "PIPELINE RELOAD TEST\n"
    )
    f.write(
        "Predictions match: "
        + str(
            np.array_equal(
                original_prediction,
                loaded_prediction,
            )
        )
        + "\n"
    )


print("\n" + "=" * 70)
print("MODULE 2 - PART B COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("- Classification comparison")
print("- Confusion matrix chart")
print("- ROC curve chart")
print("- Decision tree chart")
print("- Imbalance comparison")
print("- Random Forest GridSearch results")
print("- Complete joblib pipeline")
print("- Regression metrics")
print("- Regression residual plot")
print("- Modeling summary")