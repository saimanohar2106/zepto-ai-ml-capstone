import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 70)
print("ZEPTO AI/ML CAPSTONE - MODULE 2")
print("TITANIC EDA / ANALYTICS PIPELINE")
print("=" * 70)


# ============================================================
# TASK 1 — LOAD DATASET EXACTLY ONCE
# ============================================================

print("\n[TASK 1] Loading Titanic dataset...")

# IMPORTANT:
# sns.load_dataset() is called ONLY ONCE.
df = sns.load_dataset("titanic")

print("\nDataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Save raw dataset inside /analytics
titanic_path = os.path.join(BASE_DIR, "titanic.csv")
df.to_csv(titanic_path, index=False)

print("\nRaw dataset saved to:")
print(titanic_path)


# ============================================================
# TASK 2 — DATA PROFILING
# ============================================================

print("\n" + "=" * 70)
print("[TASK 2] DATA PROFILING")
print("=" * 70)

print("\n--- DATA INFO ---")
df.info()

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe(include="all"))

print("\n--- SHAPE ---")
print(df.shape)


# ============================================================
# MISSING VALUES
# ============================================================

missing_pct = (df.isnull().mean() * 100).sort_values(
    ascending=False
)

print("\n--- MISSING VALUE PERCENTAGES ---")
print(missing_pct)

missing_output = missing_pct.reset_index()
missing_output.columns = [
    "column",
    "missing_percentage"
]

missing_output.to_csv(
    os.path.join(
        METRIC_DIR,
        "missing_value_percentages.csv"
    ),
    index=False
)


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

print("\n--- MISSING VALUE HANDLING STRATEGY ---")

cleaned_df = df.copy()

strategies = []

for column in df.columns:

    percentage = missing_pct[column]

    if percentage == 0:

        strategy = "No missing values"

    elif percentage < 5:

        cleaned_df = cleaned_df.dropna(
            subset=[column]
        )

        strategy = (
            f"Drop rows because missing percentage "
            f"is {percentage:.2f}% (<5%)"
        )

    elif percentage <= 30:

        if pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):

            median_value = cleaned_df[column].median()

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna(median_value)
            )

            strategy = (
                f"Median imputation because missing "
                f"percentage is {percentage:.2f}% "
                f"(5%-30%)"
            )

        else:

            mode_value = cleaned_df[column].mode()

            if len(mode_value) > 0:
                fill_value = mode_value.iloc[0]
            else:
                fill_value = "Missing"

            # Handle categorical columns safely
            if isinstance(
                cleaned_df[column].dtype,
                pd.CategoricalDtype
            ):

                if fill_value not in (
                    cleaned_df[column]
                    .cat
                    .categories
                ):

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .cat
                        .add_categories(
                            [fill_value]
                        )
                    )

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna(fill_value)
            )

            strategy = (
                f"Mode imputation because missing "
                f"percentage is {percentage:.2f}% "
                f"(5%-30%)"
            )

    else:

        # High-missing categorical column
        # is retained and missing values become
        # a separate category.
        if pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):

            median_value = cleaned_df[column].median()

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna(median_value)
            )

            strategy = (
                f"Median imputation for high-missing "
                f"numeric column ({percentage:.2f}%). "
                f"Column retained to avoid unnecessary "
                f"data loss."
            )

        else:

            if isinstance(
                cleaned_df[column].dtype,
                pd.CategoricalDtype
            ):

                if "Missing" not in (
                    cleaned_df[column]
                    .cat
                    .categories
                ):

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .cat
                        .add_categories(
                            ["Missing"]
                        )
                    )

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna("Missing")
            )

            strategy = (
                f"Encode missing values as 'Missing' "
                f"because missing percentage is "
                f"{percentage:.2f}% (>30%). "
                f"Column retained as a separate category."
            )

    strategies.append(
        {
            "column": column,
            "missing_percentage": round(
                percentage,
                2
            ),
            "strategy": strategy
        }
    )


strategy_df = pd.DataFrame(strategies)

print(
    strategy_df.to_string(index=False)
)

strategy_df.to_csv(
    os.path.join(
        METRIC_DIR,
        "missing_value_strategies.csv"
    ),
    index=False
)

print("\nCleaned dataset shape:")
print(cleaned_df.shape)


# ============================================================
# SAVE CLEANED DATA
# ============================================================

cleaned_path = os.path.join(
    DATA_DIR,
    "titanic_cleaned.csv"
)

cleaned_df.to_csv(
    cleaned_path,
    index=False
)

print("\nCleaned dataset saved to:")
print(cleaned_path)


# ============================================================
# CLASS BALANCE
# ============================================================

print("\n--- CLASS BALANCE ---")

class_balance = (
    cleaned_df["survived"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(class_balance)

class_balance_output = (
    class_balance
    .reset_index()
)

class_balance_output.columns = [
    "survived",
    "percentage"
]

class_balance_output.to_csv(
    os.path.join(
        METRIC_DIR,
        "class_balance.csv"
    ),
    index=False
)


# ============================================================
# TASK 3 — UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[TASK 3] UNIVARIATE ANALYSIS")
print("=" * 70)


# ============================================================
# AGE OUTLIERS
# ============================================================

age_q1 = cleaned_df["age"].quantile(0.25)
age_q3 = cleaned_df["age"].quantile(0.75)

age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = cleaned_df[
    (cleaned_df["age"] < age_lower)
    | (cleaned_df["age"] > age_upper)
]

print("\n--- AGE IQR OUTLIERS ---")
print(f"Q1: {age_q1:.2f}")
print(f"Q3: {age_q3:.2f}")
print(f"IQR: {age_iqr:.2f}")
print(f"Lower bound: {age_lower:.2f}")
print(f"Upper bound: {age_upper:.2f}")
print(
    f"Outlier count: {len(age_outliers)}"
)


# ============================================================
# FARE OUTLIERS
# ============================================================

fare_q1 = cleaned_df["fare"].quantile(0.25)
fare_q3 = cleaned_df["fare"].quantile(0.75)

fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = cleaned_df[
    (cleaned_df["fare"] < fare_lower)
    | (cleaned_df["fare"] > fare_upper)
]

print("\n--- FARE IQR OUTLIERS ---")
print(f"Q1: {fare_q1:.2f}")
print(f"Q3: {fare_q3:.2f}")
print(f"IQR: {fare_iqr:.2f}")
print(f"Lower bound: {fare_lower:.2f}")
print(f"Upper bound: {fare_upper:.2f}")
print(
    f"Outlier count: {len(fare_outliers)}"
)


# ============================================================
# FARE STATISTICS
# ============================================================

fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode().iloc[0]

print("\n--- FARE STATISTICS ---")
print(f"Mean: {fare_mean:.2f}")
print(f"Median: {fare_median:.2f}")
print(f"Mode: {fare_mode:.2f}")

if fare_mean > fare_median > fare_mode:

    fare_skewness = "Right-skewed"

elif fare_mean < fare_median < fare_mode:

    fare_skewness = "Left-skewed"

else:

    fare_skewness = (
        "Approximately symmetric / mixed"
    )

print(
    f"Skewness classification: "
    f"{fare_skewness}"
)


# ============================================================
# SAVE UNIVARIATE SUMMARY
# ============================================================

univariate_summary = pd.DataFrame(
    {
        "metric": [
            "age_q1",
            "age_q3",
            "age_iqr",
            "age_lower_bound",
            "age_upper_bound",
            "age_outlier_count",
            "fare_q1",
            "fare_q3",
            "fare_iqr",
            "fare_lower_bound",
            "fare_upper_bound",
            "fare_outlier_count",
            "fare_mean",
            "fare_median",
            "fare_mode"
        ],
        "value": [
            age_q1,
            age_q3,
            age_iqr,
            age_lower,
            age_upper,
            len(age_outliers),
            fare_q1,
            fare_q3,
            fare_iqr,
            fare_lower,
            fare_upper,
            len(fare_outliers),
            fare_mean,
            fare_median,
            fare_mode
        ]
    }
)

univariate_summary.to_csv(
    os.path.join(
        METRIC_DIR,
        "univariate_summary.csv"
    ),
    index=False
)


# ============================================================
# AGE HISTOGRAM + BOXPLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

sns.histplot(
    cleaned_df["age"],
    kde=True,
    ax=axes[0]
)

axes[0].set_title(
    "Age Distribution"
)

axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")

sns.boxplot(
    x=cleaned_df["age"],
    ax=axes[1]
)

axes[1].set_title(
    "Age Box Plot"
)

axes[1].set_xlabel("Age")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "age_histogram_boxplot.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# FARE HISTOGRAM + BOXPLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

sns.histplot(
    cleaned_df["fare"],
    kde=True,
    ax=axes[0]
)

axes[0].set_title(
    "Fare Distribution"
)

axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Count")

sns.boxplot(
    x=cleaned_df["fare"],
    ax=axes[1]
)

axes[1].set_title(
    "Fare Box Plot"
)

axes[1].set_xlabel("Fare")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "fare_histogram_boxplot.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# TASK 4 — BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[TASK 4] BIVARIATE ANALYSIS")
print("=" * 70)


# ============================================================
# SURVIVAL BY SEX
# ============================================================

survival_by_sex = (
    cleaned_df
    .groupby("sex")["survived"]
    .mean()
    .sort_values(
        ascending=False
    )
)

print("\n--- SURVIVAL RATE BY SEX ---")
print(survival_by_sex)


# ============================================================
# SURVIVAL BY PCLASS
# ============================================================

survival_by_pclass = (
    cleaned_df
    .groupby("pclass")["survived"]
    .mean()
    .sort_index()
)

print(
    "\n--- SURVIVAL RATE BY PCLASS ---"
)

print(survival_by_pclass)


# ============================================================
# SURVIVAL BY SEX AND PCLASS
# ============================================================

survival_by_sex_pclass = (
    cleaned_df
    .groupby(
        ["sex", "pclass"]
    )["survived"]
    .mean()
)

print(
    "\n--- SURVIVAL RATE BY SEX AND PCLASS ---"
)

print(
    survival_by_sex_pclass
)


# ============================================================
# BOOLEAN MASKING
# ============================================================

female_first_class = cleaned_df[
    (cleaned_df["sex"] == "female")
    & (cleaned_df["pclass"] == 1)
]

male_third_class = cleaned_df[
    (cleaned_df["sex"] == "male")
    & (cleaned_df["pclass"] == 3)
]

print("\n--- BOOLEAN MASKING EXAMPLES ---")

print(
    "Female + 1st class rows:",
    len(female_first_class)
)

print(
    "Male + 3rd class rows:",
    len(male_third_class)
)


# ============================================================
# SAVE SURVIVAL TABLES
# ============================================================

survival_by_sex.to_csv(
    os.path.join(
        METRIC_DIR,
        "survival_by_sex.csv"
    )
)

survival_by_pclass.to_csv(
    os.path.join(
        METRIC_DIR,
        "survival_by_pclass.csv"
    )
)

survival_by_sex_pclass.to_csv(
    os.path.join(
        METRIC_DIR,
        "survival_by_sex_pclass.csv"
    )
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = (
    cleaned_df[
        correlation_columns
    ].corr()
)

print(
    "\n--- SIX-COLUMN CORRELATION MATRIX ---"
)

print(correlation_matrix)

correlation_matrix.to_csv(
    os.path.join(
        METRIC_DIR,
        "correlation_matrix.csv"
    )
)


# ============================================================
# TOP 2 CORRELATIONS
# ============================================================

correlation_pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        col1 = correlation_columns[i]
        col2 = correlation_columns[j]

        coefficient = (
            correlation_matrix
            .loc[col1, col2]
        )

        correlation_pairs.append(
            {
                "feature_1": col1,
                "feature_2": col2,
                "correlation": coefficient,
                "absolute_correlation": abs(
                    coefficient
                )
            }
        )

correlation_pairs_df = (
    pd.DataFrame(
        correlation_pairs
    )
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
)

top_two_correlations = (
    correlation_pairs_df.head(2)
)

print(
    "\n--- TWO STRONGEST CORRELATIONS ---"
)

print(
    top_two_correlations
    .to_string(index=False)
)

top_two_correlations.to_csv(
    os.path.join(
        METRIC_DIR,
        "top_two_correlations.csv"
    ),
    index=False
)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Titanic Six-Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "correlation_heatmap.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# TASK 5 — MULTIVARIATE DATA STORY
# ============================================================

print("\n" + "=" * 70)
print("[TASK 5] MULTIVARIATE DATA STORY")
print("=" * 70)


# ============================================================
# CHART 1 — SURVIVAL BY SEX
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=cleaned_df,
    x="sex",
    y="survived"
)

plt.title(
    "Survival Rate by Sex"
)

plt.ylabel(
    "Survival Rate"
)

plt.xlabel("Sex")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "survival_by_sex.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# CHART 2 — SURVIVAL BY PCLASS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived"
)

plt.title(
    "Survival Rate by Passenger Class"
)

plt.ylabel(
    "Survival Rate"
)

plt.xlabel(
    "Passenger Class"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "survival_by_pclass.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# CHART 3 — SEX + PCLASS
# ============================================================

plt.figure(
    figsize=(9, 5)
)

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Sex and Passenger Class"
)

plt.ylabel(
    "Survival Rate"
)

plt.xlabel(
    "Passenger Class"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "survival_by_sex_pclass.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# CHART 4 — AGE VS FARE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

sns.scatterplot(
    data=cleaned_df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.title(
    "Age vs Fare by Survival"
)

plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "age_vs_fare_survival.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# CHART 5 — FARE BY SURVIVAL
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=cleaned_df,
    x="survived",
    y="fare"
)

plt.title(
    "Fare Distribution by Survival"
)

plt.xlabel(
    "Survived"
)

plt.ylabel(
    "Fare"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "fare_by_survival.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# TASK 6 — EDA-ONLY STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("[TASK 6] EDA-ONLY STANDARDIZATION")
print("=" * 70)

standardization_df = (
    cleaned_df[
        ["age", "fare"]
    ].copy()
)

before_mean_age = (
    standardization_df["age"].mean()
)

before_std_age = (
    standardization_df["age"].std()
)

before_mean_fare = (
    standardization_df["fare"].mean()
)

before_std_fare = (
    standardization_df["fare"].std()
)

scaler = StandardScaler()

standardized_values = (
    scaler.fit_transform(
        standardization_df
    )
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=[
        "age",
        "fare"
    ]
)

after_mean_age = (
    standardized_df["age"].mean()
)

after_std_age = (
    standardized_df["age"].std()
)

after_mean_fare = (
    standardized_df["fare"].mean()
)

after_std_fare = (
    standardized_df["fare"].std()
)

print("\nBefore standardization:")

print(
    f"Age  -> mean: "
    f"{before_mean_age:.4f}, "
    f"std: {before_std_age:.4f}"
)

print(
    f"Fare -> mean: "
    f"{before_mean_fare:.4f}, "
    f"std: {before_std_fare:.4f}"
)

print("\nAfter standardization:")

print(
    f"Age  -> mean: "
    f"{after_mean_age:.4f}, "
    f"std: {after_std_age:.4f}"
)

print(
    f"Fare -> mean: "
    f"{after_mean_fare:.4f}, "
    f"std: {after_std_fare:.4f}"
)


# ============================================================
# SAVE STANDARDIZATION SUMMARY
# ============================================================

standardization_summary = pd.DataFrame(
    {
        "feature": [
            "age",
            "fare"
        ],
        "before_mean": [
            before_mean_age,
            before_mean_fare
        ],
        "before_std": [
            before_std_age,
            before_std_fare
        ],
        "after_mean": [
            after_mean_age,
            after_mean_fare
        ],
        "after_std": [
            after_std_age,
            after_std_fare
        ]
    }
)

standardization_summary.to_csv(
    os.path.join(
        METRIC_DIR,
        "standardization_summary.csv"
    ),
    index=False
)


# ============================================================
# EDA INTERPRETATIONS
# ============================================================

interpretations = []

interpretations.append(
    "Chart 1 - Survival by Sex: "
    "The survival rate differs substantially between "
    "female and male passengers. Female passengers "
    "have a higher observed survival rate in this dataset."
)

interpretations.append(
    "Chart 2 - Survival by Passenger Class: "
    "Survival rate changes across passenger classes. "
    "First-class passengers show a higher observed "
    "survival rate than lower classes."
)

interpretations.append(
    "Chart 3 - Survival by Sex and Passenger Class: "
    "Combining sex and passenger class reveals additional "
    "structure that is not visible from either variable "
    "alone. The survival pattern differs across both groups."
)

interpretations.append(
    "Chart 4 - Age vs Fare by Survival: "
    "Age and fare show different distributions for "
    "survivors and non-survivors. Fare contains several "
    "high-value observations, while age is distributed "
    "across a broad range."
)

interpretations.append(
    "Chart 5 - Fare by Survival: "
    "Survivors generally show a different fare distribution "
    "from non-survivors. The box plot also highlights the "
    "presence of high-fare observations."
)

interpretations.append(
    f"Fare skewness: {fare_skewness}. "
    f"The mean is {fare_mean:.2f}, median is "
    f"{fare_median:.2f}, and mode is "
    f"{fare_mode:.2f}. "
    "The ordering indicates the direction of skewness."
)

interpretations.append(
    f"Age IQR analysis identified "
    f"{len(age_outliers)} outliers, while fare "
    f"IQR analysis identified "
    f"{len(fare_outliers)} outliers using "
    "the 1.5*IQR rule."
)

if len(top_two_correlations) >= 2:

    first = top_two_correlations.iloc[0]

    second = top_two_correlations.iloc[1]

    interpretations.append(
        f"Strongest correlation: "
        f"{first['feature_1']} and "
        f"{first['feature_2']} with coefficient "
        f"{first['correlation']:.3f}. "
        "The absolute value indicates the strongest "
        "linear association among the six selected variables."
    )

    interpretations.append(
        f"Second strongest correlation: "
        f"{second['feature_1']} and "
        f"{second['feature_2']} with coefficient "
        f"{second['correlation']:.3f}. "
        "This represents the second-largest absolute "
        "linear association in the selected correlation matrix."
    )

interpretations.append(
    "Standardization check: Age and fare were standardized "
    "only for EDA using z-score scaling on the full cleaned "
    "dataset. The standardized variables are not used as "
    "inputs to the classification modeling pipeline."
)


# ============================================================
# SAVE INTERPRETATIONS
# ============================================================

interpretation_path = os.path.join(
    OUTPUT_DIR,
    "eda_interpretations.txt"
)

with open(
    interpretation_path,
    "w",
    encoding="utf-8"
) as file:

    for i, interpretation in enumerate(
        interpretations,
        start=1
    ):

        file.write(
            f"{i}. {interpretation}\n\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODULE 2 - PART A COMPLETED")
print("=" * 70)

print("\nRaw dataset:")
print(titanic_path)

print("\nCleaned dataset:")
print(cleaned_path)

print("\nMetrics saved in:")
print(METRIC_DIR)

print("\nCharts saved in:")
print(CHART_DIR)

print("\nInterpretations saved in:")
print(interpretation_path)

print("\n" + "=" * 70)
print("EDA PIPELINE FINISHED SUCCESSFULLY")
print("=" * 70)