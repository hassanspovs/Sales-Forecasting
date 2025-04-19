import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import zipfile
import os

zip_path = "C:/Users/hassa/Downloads/Sales Forecasting.zip"
extract_path = "C:/Users/hassa/Downloads/SalesData"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    print("Extracted files:", zip_ref.namelist())

csv_file = os.path.join(extract_path, 'stores_sales_forecasting.csv')
df = pd.read_csv(csv_file, encoding='ISO-8859-1')

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

# Select relevant features
features = ['Month', 'Year', 'Quantity', 'Discount']
target = 'Sales'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regressor Performance:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Random Forest)')
plt.grid(True)
plt.show()
