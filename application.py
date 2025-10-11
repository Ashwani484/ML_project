from flask import Flask,render_template,request
import joblib
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


app = Flask(__name__)
paths=Path(r"artifacts/models")
file_name = next(paths.glob("*")).name

model_path = os.path.join(paths, file_name)
scaler_path = "artifacts//processed//scaler.pkl"
print("Model path:",model_path)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template("index.html" , predictions=None)

@app.route('/predict',methods=["POST"])
def predict():
    try:
        healthcare_cost = float(request.form["healthcare_costs"])
        tumor_size = float(request.form["tumor_size"])
        Age = int(request.form["Age"])
        Incidence_Rate_per_100K = int(request.form["Incidence_Rate_per_100K"])
        mortality_rate = float(request.form["mortality_rate"])

        input = np.array([[healthcare_cost,tumor_size,Age,Incidence_Rate_per_100K,mortality_rate]])

        scaled_input = scaler.transform(input)

        prediction = model.predict(scaled_input)[0]

        return render_template('index.html' , prediction=prediction)
    
    except Exception as e:
        return str(e)
    
if __name__=="__main__":
    app.run(debug=True , host="0.0.0.0" , port=5000)

    