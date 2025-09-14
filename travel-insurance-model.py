# -------------------
# Imports
# -------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    roc_curve, auc, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # supports SMOTE inside pipeline

# -------------------
# Load data
# -------------------
df = pd.read_csv("travel insurance.csv")
df['Claim'] = df['Claim'].map({'No': 0, 'Yes': 1})

X = df.drop('Claim', axis=1)
y = df['Claim']
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------
# Define models + grids
# -------------------
param_grids = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["saga"]
    },
    "Random Forest": {
        "clf__n_estimators": [100, 150],
        "clf__max_depth": [5, 8],
        "clf__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [100, 150],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0]
    },
    "XGBoost": {
        "clf__n_estimators": [100, 150],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8],
        "clf__colsample_bytree": [0.8]
    }
}

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# -------------------
# Cross-validation setup
# -------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

best_model = None
best_score = -1
best_name = None

# -------------------
# Training with GridSearchCV
# -------------------
for name, model in models.items():
    print(f"\n===== {name} =====")
    
    # Logistic regression needs scaling, all models use SMOTE to handle imbalance
    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler(with_mean=False)) if name == "Logistic Regression" else ("identity", "passthrough"),
        ("clf", model)
    ])
    
    # Grid search
    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        verbose=2
    )
    
    grid.fit(X_train, y_train)
    
    print("Best Params:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)
    
    # Evaluate on test set
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:,1]
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    print("Test ROC-AUC:", roc_auc)
    print("Test PR-AUC:", pr_auc)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Store results
    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "CV ROC-AUC": grid.best_score_,
        "Test ROC-AUC": roc_auc,
        "Test PR-AUC": pr_auc
    })
    
    # Track best model
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = grid.best_estimator_
        best_name = name

# -------------------
# Results summary
# -------------------
results_df = pd.DataFrame(results).sort_values(by="Test ROC-AUC", ascending=False)
print("\n=== Summary of Results ===")
print(results_df)

# -------------------
# Save best model for deployment
# -------------------
print(f"\nBest model: {best_name} with Test ROC-AUC={best_score:.4f}")
joblib.dump(best_model, "best_insurance_claim_model.pkl")
print("Model saved as best_insurance_claim_model.pkl")
