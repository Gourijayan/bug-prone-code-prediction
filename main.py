
import pandas as pd

# Load dataset
df = pd.read_csv("data/kc1.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset size
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display dataset information
print("\nDataset information:")
df.info()

# STEP 3: DATA CLEANING

# 1. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 2. Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 3. Remove duplicate rows
df = df.drop_duplicates().copy()

# 4. Convert target column to 0 and 1
df["defects"] = df["defects"].astype(int)

# 5. Check target values
print("\nDefect distribution:")
print(df["defects"].value_counts())

# 6. Check cleaned dataset shape
print("\nCleaned dataset shape:")
print(df.shape)

# 7. Save cleaned dataset
df.to_csv("data/cleaned_kc1.csv", index=False)

print("\nData cleaning completed successfully!")


# STEP 4: DATA VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder
os.makedirs("images", exist_ok=True)

# Graph 1: Defect Distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="defects",
    hue="defects",
    palette="Set2",
    legend=False
)

plt.title("Defect-Prone vs Non-Defect-Prone Modules")
plt.xlabel("Defect Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Modules")

plt.tight_layout()
plt.savefig("images/defect_distribution.png")
plt.show()


# Graph 2: Lines of Code Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="loc",
    bins=30,
    color="skyblue"
)

plt.title("Distribution of Lines of Code")
plt.xlabel("Lines of Code")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("images/loc_distribution.png")
plt.show()


# Graph 3: Code Complexity vs Defects

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="defects",
    y="v(g)",
    hue="defects",
    palette="Set2",
    legend=False
)

plt.title("Code Complexity vs Defects")
plt.xlabel("Defect Status")
plt.ylabel("Cyclomatic Complexity")

plt.tight_layout()
plt.savefig("images/complexity_vs_defects.png")
plt.show()


# Graph 4: Correlation Heatmap

plt.figure(figsize=(14, 10))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap of Software Metrics")

plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()


# STEP 5: MODEL TRAINING

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Separate features and target
X = df.drop(columns=["defects"])
y = df["defects"].astype(int)

# 2. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Logistic Regression pipeline
lr_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train Logistic Regression
lr_model.fit(X_train, y_train)

# Predict
lr_predictions = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_predictions))

# 4. Random Forest pipeline
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

# Train Random Forest
rf_model.fit(X_train, y_train)

# Predict
rf_predictions = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_predictions))

# 5. Save trained models

# Save trained models
import os
import joblib

# Find the folder containing main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create models folder if it doesn't exist
models_folder = os.path.join(BASE_DIR, "models")
os.makedirs(models_folder, exist_ok=True)

# Save both models
joblib.dump(
    lr_model,
    os.path.join(models_folder, "logistic_model.pkl")
)

joblib.dump(
    rf_model,
    os.path.join(models_folder, "random_forest_model.pkl")
)

print("\nModels saved in:", models_folder)
print("Both models saved successfully!")

print("\nBoth models trained and saved successfully!")



# STEP 6: MODEL EVALUATION

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

# Function to evaluate models
def evaluate_model(name, model, predictions):
    print("\n============================")
    print(name)
    print("============================")

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }


# Evaluate Logistic Regression
lr_results = evaluate_model(
    "Logistic Regression",
    lr_model,
    lr_predictions
)

# Evaluate Random Forest
rf_results = evaluate_model(
    "Random Forest",
    rf_model,
    rf_predictions
)


# Confusion Matrix for Random Forest

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_predictions,
    display_labels=["No Defect", "Defect"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.show()



# Compare Logistic Regression and Random Forest

results_df = pd.DataFrame(
    [lr_results, rf_results],
    index=["Logistic Regression", "Random Forest"]
)

print("\nModel Comparison:")
print(results_df.round(4))

# Save comparison results
results_df.to_csv("models/model_comparison.csv")

# Plot comparison
results_df.plot(kind="bar", figsize=(10, 6))

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("images/model_comparison.png")
plt.show()



# STEP 9: FINAL MODEL VERIFICATION

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("\n===== FINAL MODEL VERIFICATION =====")

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

print("\nCleaned dataset rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
