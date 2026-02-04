import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.title("Sales Prediction using Linear Regression")

# Load dataset
df = pd.read_csv("sales_1000_data.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features & target
X = df[["AdvertisingSpend", "StoreVisitors", "Discount"]]
Y = df["Sales"]

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(Y)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Evaluation
pred = model.predict(X_test)
r2 = r2_score(Y_test, pred)
mse = mean_squared_error(Y_test, pred)
mae = mean_absolute_error(Y_test, pred)

st.subheader("Model Performance")
st.write(f"R² Score: {r2*100:.2f}%")
st.write(f"MSE: {mse:.2f}")
st.write(f"MAE: {mae:.2f}")

# Prediction section
st.subheader("Predict New Sales")

ad_spend = st.number_input("Advertising Spend", min_value=0)
visitors = st.number_input("Store Visitors", min_value=0)
discount = st.number_input("Discount", min_value=0)

if st.button("Predict"):
    new_data = [[ad_spend, visitors, discount]]
    prediction = model.predict(new_data)

    st.write("Predicted Sales Value:", prediction[0])

    if prediction[0] > 0:
        st.success("Sales Profit 📈")
    else:
        st.error("Sales Loss 📉")