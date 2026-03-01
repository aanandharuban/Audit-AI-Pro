import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("\n🧠 --- Training Hybrid Forensic Engine (Calibrated Mode) ---")

# ===============================
# 1️⃣ Load Data
# ===============================
df = pd.read_csv("master_audit_dataset.csv")

# Basic cleaning
df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
df['Narration'] = df['Narration'].astype(str).fillna("No Description")

print("\n📊 Fraud Distribution:")
print(df['Is_Fraud'].value_counts(normalize=True))

# ===============================
# 2️⃣ Text Vectorization
# ===============================
tfidf = TfidfVectorizer(
    max_features=50,
    stop_words='english',
    max_df=0.85,
    min_df=5
)
text_features = tfidf.fit_transform(df['Narration']).toarray()

# ===============================
# 3️⃣ Feature Assembly
# ===============================
numeric_cols = ['Total', 'is_round', 'is_q_end', 'v_count', 'kw_flag']
numeric = df[numeric_cols].values

X = np.hstack((numeric, text_features))
y = df['Is_Fraud']

print(f"\n📦 Total Features: {X.shape[1]}")

# ===============================
# 4️⃣ Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle class imbalance properly
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"⚖️ scale_pos_weight: {scale_pos_weight:.2f}")

# ===============================
# 5️⃣ Base XGBoost Model
# ===============================
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

# ===============================
# 6️⃣ Probability Calibration (CRITICAL FIX)
# ===============================
print("🔧 Calibrating probabilities...")
model = CalibratedClassifierCV(xgb, method='sigmoid', cv=3)
model.fit(X_train, y_train)

# ===============================
# 7️⃣ Evaluation
# ===============================
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:\n")
print(classification_report(y_test, pred))
print("ROC-AUC Score:", roc_auc_score(y_test, prob))

# ===============================
# 8️⃣ Isolation Forest (SAFE ONLY)
# ===============================
print("🔍 Training Isolation Forest on SAFE transactions...")
iso_forest = IsolationForest(
    contamination=0.05,
    random_state=42
)
iso_forest.fit(numeric[y == 0])

# ===============================
# 9️⃣ Save Artifacts
# ===============================
joblib.dump(model, "audit_brain_xgb.pkl")
joblib.dump(iso_forest, "anomaly_detector.pkl")
joblib.dump(tfidf, "text_processor.pkl")

print("\n✅ Training Complete.")
print("Saved Files:")
print(" - audit_brain_xgb.pkl")
print(" - anomaly_detector.pkl")
print(" - text_processor.pkl")