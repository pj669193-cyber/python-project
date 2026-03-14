import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import sys
sys.path.append("src")
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess()

print("\n⏳ Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

joblib.dump(model, "data/fraud_model.pkl")
print("\n✅ Model saved to data/fraud_model.pkl")
