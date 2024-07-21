import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



app = Flask(__name__)



def predict(values):
    if len(values) == 18:
        model_pkl = pickle.load(open('Python Notebooks/kidney.pkl','rb'))
        final = np.asarray(values)
        return model_pkl.predict(final.reshape(1, -1))[0]


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')


@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)


if __name__ == '__main__':
	app.run(debug = True)