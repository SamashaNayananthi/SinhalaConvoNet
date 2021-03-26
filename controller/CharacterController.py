from flask import Blueprint, request
import torch
import random
import ConvNet as Net
import Prediction as inference
import CharacterInfoLogic as infoLogic

net = Net.net
net.load_state_dict(torch.load("Sinhala_conv_net_whiteBG.pt", map_location=torch.device('cpu')))
net.eval()

classes = inference.classesList

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


@characterBP.route('/info/<character_class>', methods=['GET'])
def getInfo(character_class):
    info = infoLogic.get_info(int(character_class))
    return info
