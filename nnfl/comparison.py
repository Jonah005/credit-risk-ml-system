import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load and preprocess dataset
df = pd.read_csv('german.data-numeric', sep='\s+', header=None)
df.columns = [f'Feature_{i}' for i in range(1, 25)] + ['Target']

# Drop rows with nulls
df.dropna(inplace=True)

# Drop ID-like columns (if any)
id_like_cols = [col for col in df.columns if 'id' in col.lower() or df[col].nunique() == len(df)]
df.drop(columns=id_like_cols, inplace=True)

# Split features and target
X = df.drop('Target', axis=1)
y = df['Target'].apply(lambda val: 0 if val == 1 else 1)  # 1: bad credit

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load pre-trained ANN model
ann_model = load_model('ann_model.h5') if os.path.exists('ann_model.h5') else None

# Store predictions and results
results = {}

# --- Logistic Regression ---
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]
results['Logistic Regression'] = (y_test, y_pred_lr, y_proba_lr)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = (y_test, y_pred_rf, y_proba_rf)

# --- XGBoost ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
results['XGBoost'] = (y_test, y_pred_xgb, y_proba_xgb)

# --- ANN (pretrained) ---
if ann_model:
    y_proba_ann = ann_model.predict(X_test)
    y_pred_ann = (y_proba_ann > 0.5).astype(int).flatten()
    results['ANN'] = (y_test.to_numpy(), y_pred_ann, y_proba_ann.flatten())

# --- Evaluation and Reporting ---
os.makedirs('output/comparison', exist_ok=True)
report_lines = []

plt.figure(figsize=(8, 6))
for name, (yt, yp, yp_proba) in results.items():
    # Print classification report
    report = classification_report(yt, yp, digits=2)
    report_lines.append(f"Model: {name}\n{report}\n")
    auc = roc_auc_score(yt, yp_proba)
    fpr, tpr, _ = roc_curve(yt, yp_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

# Save reports
with open("output/comparison/model_comparison_report.txt", "w") as f:
    f.writelines("\n".join(report_lines))

# Final ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.tight_layout()
plt.savefig("output/comparison/roc_comparison.png")
plt.close()

print("\u2705 Model comparison complete. Reports and ROC saved in output/comparison/")
