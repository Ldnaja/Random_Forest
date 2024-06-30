# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'change to the path of dataset'
data = pd.read_excel(file_path)

# Select the relevant variables
X = data[['year', 'ws_1', 'temp_2', 'temp_1', 'average']]
y = data['actual']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create the Random Forest model with configurable hyperparameters
def create_rf_model(bootstrap, n_estimators, max_depth, min_samples_leaf, min_samples_split):
    model = RandomForestRegressor(
        bootstrap=bootstrap,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42
    )
    return model

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Test configurations
configs = [
    {'bootstrap': True, 'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 4, 'min_samples_split': 5, 'label': 'Config 1'},
    {'bootstrap': True, 'n_estimators': 400, 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'label': 'Config 2'},
    {'bootstrap': True, 'n_estimators': 600, 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'label': 'Config 3'},
    {'bootstrap': True, 'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 3, 'min_samples_split': 10, 'label': 'Config 4'},
    {'bootstrap': True, 'n_estimators': 500, 'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 2, 'label': 'Config 5'},
    {'bootstrap': False, 'n_estimators': 400, 'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 5, 'label': 'Config 6'},
    {'bootstrap': False, 'n_estimators': 600, 'max_depth': 60, 'min_samples_leaf': 1, 'min_samples_split': 2, 'label': 'Config 7'},
    {'bootstrap': False, 'n_estimators': 300, 'max_depth': 25, 'min_samples_leaf': 4, 'min_samples_split': 2, 'label': 'Config 8'},
    {'bootstrap': False, 'n_estimators': 500, 'max_depth': 35, 'min_samples_leaf': 2, 'min_samples_split': 10, 'label': 'Config 9'},
    {'bootstrap': True, 'n_estimators': 700, 'max_depth': 60, 'min_samples_leaf': 1, 'min_samples_split': 5, 'label': 'Config 10'}
]

results = []

# Train and evaluate the model using KFold cross-validation for each configuration
for config in configs:
    maes, mses = [], []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = create_rf_model(
            bootstrap=config['bootstrap'],
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_leaf=config['min_samples_leaf'],
            min_samples_split=config['min_samples_split']
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, y_pred))
        mses.append(mean_squared_error(y_test, y_pred))

    results.append({
        'config': config['label'],
        'mae': np.mean(maes),
        'mse': np.mean(mses),
        'maes': maes,
        'mses': mses,
        'model': model
    })

# Print the average performance metrics for each configuration
for result in results:
    print(f"Configuration: {result['config']}")
    print(f"Mean Absolute Error (MAE): {result['mae']:.4f}")
    print(f"Mean Squared Error (MSE): {result['mse']:.4f}\n")

# Determine the importance of the variables for the best model
best_model = min(results, key=lambda x: x['mae'])['model']
importances = best_model.feature_importances_
feature_names = ['year', 'ws_1', 'temp_2', 'temp_1', 'average']

# Display the importance of variables
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Plot error convergence curves for each configuration
plt.figure(figsize=(12, 8))
for result in results:
    plt.plot(result['maes'], label=f"{result['config']} MAE", linestyle='-', marker='o')
    plt.plot(result['mses'], label=f"{result['config']} MSE", linestyle='--', marker='x')

plt.title('Convergence of Errors Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

# Prepare data for boxplot
mae_data = []
mse_data = []
labels = []

for result in results:
    mae_data.extend(result['maes'])
    mse_data.extend(result['mses'])
    labels.extend([result['config']] * len(result['maes']))

# Create dataframe for visualization
df_mae = pd.DataFrame({'Configuration': labels, 'MAE': mae_data})
df_mse = pd.DataFrame({'Configuration': labels, 'MSE': mse_data})

# Boxplot of MAE errors for each configuration
plt.figure(figsize=(14, 8))
sns.boxplot(x='Configuration', y='MAE', data=df_mae)
plt.title('Boxplot of MAE for Different Configurations')
plt.xlabel('Configuration')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xticks(rotation=45)
plt.show()

# Boxplot of the MSE errors for each configuration
plt.figure(figsize=(14, 8))
sns.boxplot(x='Configuration', y='MSE', data=df_mse)
plt.title('Boxplot of MSE for Different Configurations')
plt.xlabel('Configuration')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(rotation=45)
plt.show()

# Preparing data for bar charts
mae_means = [result['mae'] for result in results]
mse_means = [result['mse'] for result in results]
labels = [result['config'] for result in results]

# Create bar chart for MAE and MSE
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(labels))

plt.bar(index, mae_means, bar_width, label='MAE', color='skyblue')
plt.bar(index + bar_width, mse_means, bar_width, label='MSE', color='orange')

plt.xlabel('Configuration')
plt.ylabel('Error')
plt.title('Comparison of Mean Absolute Error (MAE) and Mean Squared Error (MSE)')
plt.xticks(index + bar_width / 2, labels, rotation=45)
plt.legend()

plt.show()
