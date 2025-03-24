from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Train model dynamically if model.pkl not found
if not os.path.exists("model.pkl"):
    df = pd.read_csv("crop_recommendation.csv")
    features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    labels = df['label']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(features, labels)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
else:
    model = pickle.load(open("model.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    crop = None
    if request.method == 'POST':
        try:
            input_data = [
                int(request.form['nitrogen']),
                int(request.form['phosphorous']),
                int(request.form['potassium']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            prediction = model.predict([input_data])[0]
            crop = prediction
        except Exception as e:
            crop = f"Error: {e}"
    return render_template('index.html', crop=crop)

if __name__ == "__main__":
    app.run(debug=True)