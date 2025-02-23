from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load Trained Mo
model_path = os.path.join(os.getcwd(), "model", "house_price_model.pkl")
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    sqft_living = float(data['sqft_living'])
    bedrooms = int(data['bedrooms'])
    bathrooms = float(data['bathrooms'])
    floors = int(data['floors'])

    features = np.array([[sqft_living, bedrooms, bathrooms, floors]], dtype=float)
    price = model.predict(features)[0]

    return render_template('index.html', predicted_price=round(price, 2))
if __name__ == '__main__':
    port=int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)