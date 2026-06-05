from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Ungaloda model file name
model = joblib.load('model (4).pkl')

@app.route('/')
def home():
    return render_template('forms.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperatu = float(request.form['temperatu'])
        humidity = float(request.form['humidity'])
        apparent_t = float(request.form['apparent_t'])
        pressure = float(request.form['pressure'])

        final_features = np.array([[temperatu, humidity, apparent_t, pressure]])
        prediction = model.predict(final_features)

        # Un dataset la irukura labels ithu - theva panna maathiko
        label = {0: "Partly Cloudy", 1: "Mostly Cloudy", 2: "Clear", 3: "Rainy", 4: "Foggy"}
        output = label.get(prediction[0], f"Class {prediction[0]}")

        # Background ku class name - space eduthudum: "Partly Cloudy" -> "PartlyCloudy"
        weather_class = output.replace(" ", "")

        return render_template('forms.html',
                             prediction_text=f'Predicted Weather: {output}',
                             weather_class=weather_class)

    except Exception as e:
        return render_template('forms.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)