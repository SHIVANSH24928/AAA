import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train=pd.read_csv('/content/train (3).csv')
df_test=pd.read_csv('/content/test.csv')
df_train
categorical_cols = x_train.select_dtypes(include='object').columns

x_train = pd.get_dummies(x_train, columns=categorical_cols, dummy_na=False)
x_test = pd.get_dummies(x_test, columns=categorical_cols, dummy_na=False)

# Align columns after one-hot encoding
x_train, x_test = x_train.align(x_test, join='inner', axis=1)

from sklearn.impute import SimpleImputer

numerical_cols_with_na = x_train.select_dtypes(include=np.number).columns[x_train.select_dtypes(include=np.number).isnull().any()]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train[numerical_cols_with_na] = imputer.fit_transform(x_train[numerical_cols_with_na])
x_test[numerical_cols_with_na] = imputer.transform(x_test[numerical_cols_with_na])

print("Missing values in x_train after imputation:")
print(x_train.isnull().sum().sum())
print("\nMissing values in x_test after imputation:")
print(x_test.isnull().sum().sum())
model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
predictions = model.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("MSE:", mse)
print("R2 Score:", r2)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction App")

# Input fields
area = st.number_input("Living Area (sq ft)", value=1500)
year_built = st.number_input("Year Built", value=2000)
garage = st.number_input("Garage Cars", value=2)
bath = st.number_input("Full Bathrooms", value=2)
quality = st.slider("Overall Quality (1-10)", 1, 10, 5)
basement = st.number_input("Total Basement Area (sq ft)", value=500)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, quality, garage, basement, bath, year_built]],
                              columns=['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ${int(prediction):,}")
