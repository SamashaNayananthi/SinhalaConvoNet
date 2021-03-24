import json
from base64 import b64decode
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage
import torch
from torchvision import transforms
import model.Character as characterModel
import torch.nn.functional as F
import pandas as pd

df = pd.read_csv('CharacterData.csv', header=0)
classesList = df["Character"].tolist()

def url_to_img(dataURL):
    string = str(dataURL)
    comma = string.find(",")
    code = string[comma + 1:]
    decoded = b64decode(code)
    buf = BytesIO(decoded)
    img = Image.open(buf)

    converted = img.convert("LA")
    la = np.array(converted)
    la[la[..., -1] == 0] = [255, 255]
    whiteBG = Image.fromarray(la)

    converted = whiteBG.convert("L")
    inverted = ImageOps.invert(converted)

    bounding_box = inverted.getbbox()
    padded_box = tuple(map(lambda i, j: i + j, bounding_box, (-5, -5, 5, 5)))
    cropped = inverted.crop(padded_box)

    thick = cropped.filter(ImageFilter.MaxFilter(5))

    ratio = 48.0 / max(thick.size)
    new_size = tuple([int(round(x * ratio)) for x in thick.size])
    res = thick.resize(new_size, Image.LANCZOS)

    arr = np.asarray(res)
    com = ndimage.measurements.center_of_mass(arr)
    result = Image.new("L", (64, 64))
    box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
    result.paste(res, box)
    return result


def transformImg(img):
    my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return my_transforms(img).unsqueeze(0)


def get_prediction(url, net):
    img = url_to_img(url)
    transformed = transformImg(img)
    output = net(transformed)
    probabilities, predictions = torch.topk(output.data, 2)

    print("prediction - ", classesList[predictions.data[0][0]], " - ", predictions.data[0][0].item(),
          " : Probability - ", probabilities.data[0][0].item())

    print("prediction - ", classesList[predictions.data[0][1]], " - ", predictions.data[0][1].item(),
          " : Probability - ", probabilities.data[0][1].item())
    confidence1 = int(round(probabilities.data[0][0].item() * 100))
    confidence2 = int(round(probabilities.data[0][1].item() * 100))
    guess = characterModel.Character(classesList[predictions.data[0][0]], confidence1,
                                     classesList[predictions.data[0][1]], confidence2)
    return json.dumps(guess.__dict__)
