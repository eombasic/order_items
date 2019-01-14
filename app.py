__author__ = "Emir Ombasic"

from flask import Flask
from flask import jsonify
from flask import request
import os
import importlib


app = Flask(__name__)

def getBasePath():
    return os.path.dirname(os.path.realpath(__file__))

@app.route("/ping")
def ping():
    return "pong"

@app.route("/predict/delivery/<customer>", methods=['POST'])
def predict_delivery(customer):

    try:
        customerpredictormodule = importlib.import_module('delivery.' + customer)
    except:
        response = {
            'error': True,
            'error_msg': 'Wrong Customer. Module doesnt exist.'
        }
        return jsonify(response)

    data = request.get_json()
    deliveryPredictor = customerpredictormodule.DeliveryPredictor()
    deliveryPredictor.setBasePath(getBasePath())
    prediction = deliveryPredictor.predict(data)

    del deliveryPredictor
    del customerpredictormodule
    del data

    response = {
        'error': False,
        'customer': customer,
        'prediction': prediction
    }

    return jsonify(response)
