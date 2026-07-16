import pandas as pd
import numpy as np

#load csv file as dataframe
df = pd.read_csv("used_cars.csv")

#remove duplicates rows
df = df.drop_duplicates()

#fill missing mileage with median and engine size with "Unknown"
df['Mileage(km)'] = df['Mileage(km)'].fillna(df['Mileage(km)'].median())
df['EngineSize(L)'] = df['EngineSize(L)'].fillna("Unknown")

#convert categorical variables into 0s and 1s
df = pd.get_dummies(df, columns=['FuelType', 'Transmission', 'Brand'])


#store current year
CURRENT_YEAR = 2026

#create new feature CarAge by subtracting Year from current year if 'Year' column exists
if 'Year' in df.columns:
    df['CarAge'] = CURRENT_YEAR - df['Year']
else:
    print("Column 'Year' missing; skipping CarAge.")

#create bins for grouping milages
bins = [0, 20000, 60000, 100000, 200000, df['Mileage(km)'].max()]
#create labels for all bins
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
#assign each car to mileage category based on its mileage
df['MileageBin'] = pd.cut(df['Mileage(km)'], bins=bins, labels=labels, include_lowest=True)
#convert mileage categories into dummy variables 
df = pd.get_dummies(df, columns=['MileageBin'])


#PLOTS

import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


sns.scatterplot(x='Mileage(km)', y='Price', data=df, alpha=0.5)
plt.title("Mileage vs Price")
plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.tight_layout()
plt.show()



# PREPARE DATA FOR MODELING
# ===============================

# target variable
y = df['Price']

# feature variables
X = df.drop(columns=['Price'])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# LINEAR REGRESSION MODEL
# ===============================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# predictions
lr_preds = lr_model.predict(X_test)

# evaluation
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print("Linear Regression Results")
print("MAE:", lr_mae)
print("R2 Score:", lr_r2)
print()


# ===============================
# RANDOM FOREST REGRESSION
# ===============================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# predictions
rf_preds = rf_model.predict(X_test)

# evaluation
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("Random Forest Results")
print("MAE:", rf_mae)
print("R2 Score:", rf_r2)
print()


# ===============================
# MODEL COMPARISON
# ===============================

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [lr_mae, rf_mae],
    'R2 Score': [lr_r2, rf_r2]
})

print(results)


# ===============================
# ACTUAL VS PREDICTED PLOT
# ===============================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_preds, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)
