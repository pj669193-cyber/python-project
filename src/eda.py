import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\nFraud %: {df['Class'].mean()*100:.2f}%")
print("\nMissing Values:", df.isnull().sum().sum())

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette=['steelblue','red'])
plt.title("Class Distribution (0=Normal, 1=Fraud)")
plt.savefig("data/class_distribution.png")
plt.show()

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

print("\n✅ EDA Done!")
