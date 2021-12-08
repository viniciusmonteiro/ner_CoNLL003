import string
import flask
import json
from flask import request, jsonify
from simpletransformers.ner import NERModel,NERArgs
from os import path

app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config['JSON_AS_ASCII'] = False


# NER model
chosen_pretrained = 'dslim/bert-base-NER'
model = NERModel('bert', chosen_pretrained, use_cuda=False)

@app.route('/', methods=['GET'])
def home():
    return '''<h1>NER API</h1>'''

def get_prediction(sentence):
    if sentence is None:
        return "Invalid string."
    
    prediction, model_output = model.predict([sentence])
    return prediction, model_output


@app.route('/api/predict', methods=['GET'])
def predict():
    
    if 'sentence' in request.args :
        sentence = str(request.args['sentence'])
    else:
        return "Error: No sentence  provided. Please specify a sentence param."
    
    prediction, model_output = get_prediction(sentence)
    
    return jsonify({
        "sentence": sentence,
        "model_output": str(model_output),
        "predicted": prediction
    })

app.run(host="0.0.0.0", port=8323)