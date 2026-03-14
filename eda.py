import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

# ── Basic Info ──────────────────────────────
print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\nFraud %: {df['Class'].mean()*100:.2f}%")
print("\nMissing Values:", df.isnull().sum().sum())

# ── Plot 1: Class Imbalance ──────────────────
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette=['steelblue','red'])
plt.title("Class Distribution (0=Normal, 1=Fraud)")
plt.xticks([0,1], ['Normal', 'Fraud'])
plt.savefig("data/class_distribution.png")
plt.show()

# ── Plot 2: Transaction Amount ───────────────
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df[df['Class']==0]['Amount'], bins=50, color='steelblue')
plt.title("Normal Transaction Amounts")

plt.subplot(1,2,2)
sns.histplot(df[df['Class']==1]['Amount'], bins=50, color='red')
plt.title("Fraud Transaction Amounts")
plt.tight_layout()
plt.savefig("data/amount_distribution.png")
plt.show()

# ── Plot 3: Correlation Heatmap ───────────────
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("data/correlation_heatmap.png")
plt.show()

print("\n✅ EDA Done!")