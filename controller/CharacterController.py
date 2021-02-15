from flask import Blueprint, request
import torch
import random
import ConvNet as Net
import Prediction as inference

net = Net.net
net.load_state_dict(torch.load("Sinhala_conv_net30.pt", map_location=torch.device('cpu')))
net.eval()

classes = inference.classes

characterBP = Blueprint('character', __name__)


@characterBP.route('/predict/', methods=['GET', 'POST'])
def predict():
    string_data = request.get_data().decode('utf-8')
    prediction = inference.get_prediction(string_data, net)
    return prediction


@characterBP.route('/suggest/', methods=['GET', 'POST'])
def suggest():
    suggestion = random.choice(classes)
    return suggestion
