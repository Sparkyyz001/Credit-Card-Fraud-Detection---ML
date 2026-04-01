import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix
)

# --- 1. ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
data = pd.read_csv("creditcard.csv")

# --- 2. АНАЛИЗ КЛАССОВ ---
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))

print(f"Доля мошенничества: {outlierFraction:.5f}")
print(f"Мошеннических операций: {len(fraud)}")
print(f"Обычных операций: {len(valid)}")

# --- 3. ТЕПЛОВАЯ КАРТА (КОРРЕЛЯЦИЯ) ---
print("\nСоздание тепловой карты...")
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True, cmap="RdBu_r")
plt.title("Correlation Matrix")
plt.show()

# --- 4. ПОДГОТОВКА ДАННЫХ ---
print("\nПодготовка данных для обучения...")
X = data.drop(['Class'], axis = 1)
Y = data["Class"]
xData = X.values
yData = Y.values

xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size = 0.2, random_state = 42
)

# --- 5. ОБУЧЕНИЕ МОДЕЛИ (RANDOM FOREST) ---
print("\nОбучение модели (это может занять 1-3 минуты)...")
rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)

# --- 6. ОЦЕНКА РЕЗУЛЬТАТОВ ---
print("\n--- ИТОГОВЫЕ МЕТРИКИ ---")
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")

# --- 7. МАТРИЦА ОШИБОК (ВИЗУАЛИЗАЦИЯ) ---
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()