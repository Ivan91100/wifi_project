import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# PARAMETERS
FILE_PATH = r"C:\Users\ivane\OneDrive - ISEP\CSV_file"  # Update this with your actual file path
TARGET_COLUMN = "occupancy"  # Number of people in the room

# STEP 1: Load the Raw Data
df = pd.read_csv(FILE_PATH, sep=";", decimal=".")
print("Data loaded. Sample:")
print(df.head())

# STEP 2: Feature Selection
# Using 'n_devices' and other potential useful features
feature_columns = ["n_devices", "rssi", "ch_freq", "idx", "seq_num"]  # Add more if relevant

# Check for missing columns
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing columns in dataset: {missing_features}")

X = df[feature_columns]
y = df[TARGET_COLUMN]

# STEP 3: Data Preprocessing
X.fillna(0, inplace=True)  # Replace missing values with 0
y.fillna(0, inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")

# STEP 5: Train Machine Learning Models
# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# STEP 6: Model Evaluation
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

# STEP 7: Feature Importance (for Random Forest)
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_columns, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# STEP 8: Save the Best Model and Scaler
joblib.dump(scaler, r"C:\Users\ivane\OneDrive - ISEP\CSV_file\scaler.pkl")
joblib.dump(rf_model, r"C:\Users\ivane\OneDrive - ISEP\CSV_file\occupancy_rf_model.pkl")
print("Model and scaler saved.")
