import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# === Load and preprocess daily data ===
df = pd.read_csv("daily_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['symbol', 'timestamp'])
df['gap'] = df.groupby('symbol')['close'].diff()
df = df.dropna(subset=['gap'])
df['date'] = df['timestamp'].dt.date

# === Daily average gap ===
daily_gap = df.groupby('date')['gap'].mean().reset_index()
daily_gap.rename(columns={'gap': 'avg_gap'}, inplace=True)

# === Detect actual crises (peaks/troughs) ===
threshold = daily_gap['avg_gap'].std() * 4
peaks, _ = find_peaks(daily_gap['avg_gap'], prominence=threshold)
troughs, _ = find_peaks(-daily_gap['avg_gap'], prominence=threshold)
crisis_days = set(daily_gap['date'].iloc[np.concatenate((peaks, troughs))])

# === Build supervised dataset ===
lookback_days = 10
crisis_horizon_days = 3
X, y, dates = [], [], []

for i in range(lookback_days, len(daily_gap) - crisis_horizon_days):
    window = daily_gap['avg_gap'].iloc[i - lookback_days:i].values
    future_dates = daily_gap['date'].iloc[i:i + crisis_horizon_days]
    label = int(any(date in crisis_days for date in future_dates))
    X.append(window)
    y.append(label)
    dates.append(daily_gap['date'].iloc[i])

X = np.array(X)
y = np.array(y)

# === Time-based split (80% train, 20% test) ===
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
test_dates = dates[split_index:]

# === Train logistic regression with balanced class weights ===
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# === Predict with threshold tuning ===
proba = model.predict_proba(X_test)[:, 1]
threshold = 0.6  # << You can adjust this (0.5–0.7 usually)
y_pred = (proba > threshold).astype(int)

# === Evaluation ===
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# === Plot: Actual vs Predicted Crises ===
plt.figure(figsize=(14, 6))
plt.step(test_dates, y_test, label="Actual Crisis", where='post', linewidth=2, color='black')
plt.step(test_dates, y_pred, label="Predicted Crisis", where='post', linestyle='--', linewidth=2, color='orange')
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.ylim(-0.1, 1.1)
plt.title(f"Crisis Prediction vs Actual (Threshold = {threshold})")
plt.xlabel("Date")
plt.ylabel("Crisis (0/1)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
