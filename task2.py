import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

df.hist(figsize=(10,8))
plt.suptitle("Histograms of Features")
plt.savefig("histograms.png")

sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot for Age and Fare")
plt.savefig("boxplot.png")

corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("correlation.png")

sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']])
plt.savefig("pairplot.png")

print("\n✅ EDA completed successfully!")