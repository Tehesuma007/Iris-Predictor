from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('iris.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('iris.html', **locals())

@app.route('/predict', methods=['POST','GET'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return render_template('iris.html', **locals())

        

if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)