import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# PARAMETERS
AGG_FILE = r"C:\Users\ivane\OneDrive\Bureau\CSV_file\aggregated_features.csv"
TARGET_COLUMN = "occupancy"  # Ground truth: number of people in the room

# STEP 1: Load the Aggregated Data
df = pd.read_csv(AGG_FILE, sep=";", decimal=".")
print("Data loaded. Sample:")
print(df.head())

# STEP 2: Verify Columns and Create 'avg_n_devices' if not present
# We expect columns: datetime, unique_devices, total_requests, avg_rssi, std_rssi, occupancy, n_devices
if 'avg_n_devices' not in df.columns:
    # If the raw column n_devices exists, we assume each row already represents an aggregated value.
    df['avg_n_devices'] = df['n_devices']

# STEP 3: Select Features and Target
# We focus on using avg_n_devices along with other features to predict occupancy.
feature_columns = ['unique_devices', 'total_requests', 'avg_rssi', 'std_rssi', 'avg_n_devices']
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing columns in aggregated data: {missing_features}")

X = df[feature_columns]
y = df[TARGET_COLUMN]

# STEP 4: Data Preprocessing
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")

# STEP 6: Model Training

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# STEP 7: Model Evaluation

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
    print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_test):.2f}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
    print(f"Test RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
    print(f"Train R2:   {r2_score(y_train, y_pred_train):.2f}")
    print(f"Test R2:    {r2_score(y_test, y_pred_test):.2f}")

evaluate_model(lr_model, X_train, X_test, y_train, y_test, model_name="Linear Regression")
evaluate_model(rf_model, X_train, X_test, y_train, y_test, model_name="Random Forest")

# STEP 8: Feature Importance (for Random Forest)
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_columns, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# STEP 9: Save the Best Model and Scaler for Future Use
joblib.dump(scaler, r"C:\Users\ivane\OneDrive\Bureau\CSV_file\scaler.pkl")
joblib.dump(rf_model, r"C:\Users\ivane\OneDrive\Bureau\CSV_file\occupancy_rf_model.pkl")
print("Model and scaler saved.")
