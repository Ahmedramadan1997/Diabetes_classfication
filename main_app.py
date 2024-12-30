from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class LogTransformer(BaseEstimator, TransformerMixin):
    
    # fit
    def fit(self, X, y=None):
        self.n_features_in = X.shape[1]
        return self
    
    # transformer
    def transform(self, X, y=None):
        assert self.n_features_in == X.shape[1]
        return np.log(X)
    

app = Flask(__name__)

model = joblib.load("oversampling_pipeline.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    age = int(request.form['age'])
    location = request.form['location']
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    smoking_history = request.form['smoking_history']
    bmi = float(request.form['bmi'])
    hbA1c_level = float(request.form['hbA1c_level'])
    blood_glucose_level = int(request.form['blood_glucose_level'])
    input_data = pd.DataFrame([[gender, age, location, hypertension, heart_disease, smoking_history, bmi, hbA1c_level, blood_glucose_level]], columns = ["gender", "age", "location", "hypertension", "heart_disease", "smoking_history", "bmi", "hbA1c_level", "blood_glucose_level"])
    print(type(age))
    print(input_data)
    

    prediction = model.predict(input_data)
    print(prediction)
    prediction_text = 'Patient has Diabetes' if prediction[0] == 1 else 'Patient does not have Diabetes'
    
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)