import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Step 1: Download dataset using kagglehub
path = kagglehub.dataset_download("bhavikjikadara/car-price-prediction-dataset")
print("Path to dataset files:", path)

# Print the contents of the path
print("\nContents of the downloaded dataset path:")
for item in os.listdir(path):
    print(item)

# Load the dataset
file_path = f"{path}/car_prediction_data.csv"  # Adjust path if necessary
df = pd.read_csv(file_path)

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize distributions of numerical columns
plt.figure(figsize=(10, 6))
df.hist(figsize=(12, 8), bins=30)
plt.show()

numeric_features = df.select_dtypes(include=[np.number]).columns
# Visualize correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 3: Data Preprocessing
# No Missing Values
# Encode categorical variables (example: car make, model)
df = pd.get_dummies(df, drop_first=True)

# Step 4: Splitting data into features and target variable
X = df.drop(columns=['Selling_Price'])  # 'price' is the target column, adjust as per your dataset
y = df['Selling_Price']

# Save the training columns for use in the API
training_columns = list(X.columns)
joblib.dump(training_columns, './models/training_columns.pkl')
print("Training columns saved as 'training_columns.pkl'")
print("Expected input after preprocessing:", training_columns)

# Step 5: Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 6: Feature Scaling (Normalization/Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use in the API
joblib.dump(scaler, './models/scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# Step 7: Model Building (Random Forest Regression)
from sklearn.model_selection import GridSearchCV

# Step 1: Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Step 2: Set up the GridSearchCV with RandomForestRegressor
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2,
                           scoring='neg_mean_absolute_error')  # Use negative MAE for minimization

# Step 3: Fit the GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Step 4: Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation MAE:", -grid_search.best_score_)

# Step 5: Evaluate the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Step 8: Model Evaluation
y_pred = best_model.predict(X_test_scaled)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")

# Visualize residuals
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_pred, lowess=False, line_kws={'color': 'red', 'lw': 1})
plt.title('Residual Plot')
plt.show()

# Step 9: Save the trained model using joblib
joblib.dump(best_model, './models/car_price_model.pkl')
print("Model saved successfully as 'car_price_model.pkl'")
