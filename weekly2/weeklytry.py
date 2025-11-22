# ==========================================
# 1. Import Libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

from tensorflow.keras import models, layers

# ==========================================
# 2. Load & Merge Data
# ==========================================
df_purchase = pd.read_csv("User_product_purchase_details_p2.csv")
df_user = pd.read_csv("user_demographics.csv")

# Merge on User_ID
df = pd.merge(df_purchase, df_user, on="User_ID", how="left")

# ==========================================
# 3. Data Preparation
# ==========================================
# Create binary target
df["High_Value_Purchase"] = (df["Purchase"] >= 10000).astype(int)

# Drop unnecessary columns
df = df.drop(["Product_ID"], axis=1)

# Handle missing values
df = df.fillna(0)

# Encode categorical variables
df = pd.get_dummies(df, columns=["Gender","Age","City_Category","Stay_In_Current_City_Years"], drop_first=True)

# Features & target
X = df.drop(["High_Value_Purchase","Purchase"], axis=1)
y = df["High_Value_Purchase"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. Baseline Logistic Regression
# ==========================================
log = LogisticRegression(max_iter=2000)
log.fit(X_train_scaled, y_train)
pred_lr = log.predict(X_test_scaled)

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# Extra metrics
precision_lr = precision_score(y_test, pred_lr)
recall_lr = recall_score(y_test, pred_lr)
f1_lr = f1_score(y_test, pred_lr)

print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1-Score:", f1_lr)

# ==========================================
# 5. MLP Classifier (Keras)
# ==========================================
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["precision"])

history = model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print("\n=== MLP Classifier Results ===")
print("Test Accuracy:", acc)

# Predictions for metrics
pred_mlp = (model.predict(X_test_scaled) > 0.5).astype("int32")

precision_mlp = precision_score(y_test, pred_mlp)
recall_mlp = recall_score(y_test, pred_mlp)
f1_mlp = f1_score(y_test, pred_mlp)

print("Precision:", precision_mlp)
print("Recall:", recall_mlp)
print("F1-Score:", f1_mlp)

# ==========================================
# 6. Compare Models (Precision-focused)
# ==========================================
print("\n=== Precision Validation ===")
print("Logistic Regression Precision:", precision_lr)
print("MLP Precision:", precision_mlp)

if precision_lr > precision_mlp:
    print("Logistic Regression performs better in terms of precision.")
elif precision_mlp > precision_lr:
    print("MLP performs better in terms of precision.")
else:
    print("Both models have equal precision.")

# Plot training history
plt.plot(history.history['precision'], label='Train Accuracy')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.title("MLP Training Precision")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()