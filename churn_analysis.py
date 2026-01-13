# Customer Churn Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")


print("Dataset Loaded Successfully!")
print(df.head())

# -------------------------------
# 2. Data Cleaning
# -------------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# -------------------------------
# 3. Feature Engineering
# -------------------------------

# Tenure Bands
bins = [0, 3, 12, 24, 48, 72]
labels = ['<3 Months', '3–12 Months', '1–2 Years', '2–4 Years', '4–6 Years']
df['TenureBand'] = pd.cut(df['tenure'], bins=bins, labels=labels)

# -------------------------------
# 4. Basic Statistics
# -------------------------------
total_customers = len(df)
churned_customers = df['Churn'].sum()
churn_rate = (churned_customers / total_customers) * 100

print(f"Total Customers: {total_customers}")
print(f"Churned Customers: {churned_customers}")
print(f"Churn Rate: {churn_rate:.2f}%")

# -------------------------------
# 5. Visualizations
# -------------------------------
sns.set_style("whitegrid")

# 1. Churn Count
plt.figure()
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# 2. Contract Type vs Churn
plt.figure()
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.show()

# 3. Tenure Distribution
plt.figure()
sns.histplot(df['tenure'], bins=30)
plt.title("Tenure Distribution")
plt.show()

# 4. Monthly Charges Distribution
plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# 5. Heatmap (Correlation)
plt.figure()
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 6. Tenure Band vs Churn
plt.figure()
sns.countplot(x='TenureBand', hue='Churn', data=df)
plt.title("Tenure Band vs Churn")
plt.show()

# 7. Internet Service vs Churn
plt.figure()
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title("Internet Service vs Churn")
plt.show()

# 8. Payment Method vs Churn
plt.figure()
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title("Payment Method vs Churn")
plt.show()

# 9. Gender vs Churn
plt.figure()
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Gender vs Churn")
plt.show()

# 10. Senior Citizen vs Churn
plt.figure()
sns.countplot(x='SeniorCitizen', hue='Churn', data=df)
plt.title("Senior Citizen vs Churn")
plt.show()

# 11. Contract Type for <3 Month Tenure
short_tenure = df[df['tenure'] < 3]
monthly_churn = short_tenure[short_tenure['Contract'] == 'Month-to-month']['Churn'].mean() * 100

print(f"Churn rate for monthly contract users (<3 months): {monthly_churn:.2f}%")

# -------------------------------
# 6. Key Insight
# -------------------------------
print("INSIGHT:")
print("A high percentage of churned users are on month-to-month contracts with less than 3 months tenure.")
print("Early engagement strategies are recommended.")
