import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('Housing.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Features and target
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🏠 House Price Predictor")

area = st.number_input("Area (sqft)", value=2000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=2)
parking = st.number_input("Parking spaces", value=1)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking]],
                             columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking'])

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: {prediction[0]:,.2f}")
