import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load Dataset
data = pd.read_csv("data.csv")

# Select Features & Target
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]
y = data['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Trained and Saved!")