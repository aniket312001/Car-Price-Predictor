from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd 
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open('LinearRegressionModel.pkl','rb'))
df = pd.read_csv('Clean_car.csv')

@app.route('/',methods=['POST','GET'])
def home():
    companies = sorted(df['company'].unique())
    model_name = sorted(df['name'].unique())
    year = sorted(df['year'].unique(),reverse=True)
    fuel_type = df['fuel_type'].unique()
    companies.insert(0,"Select Company")
    return render_template('index.html', companies=companies, car_models=model_name, years=year, fuel_types=fuel_type)


@app.route('/predict',methods=['POST','GET'])
def pred():
    company = request.form.get('company')
    car_model = request.form.get('car_model') 
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven,fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))

    return str(np.round(prediction[0],2))


if __name__ == "__main__":
    app.run(debug=True)