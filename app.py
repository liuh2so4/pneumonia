from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
import torch
import torch.nn as nn
from torchvision.models import alexnet, vgg16,  resnet18
from torchvision import transforms
import numpy as np
# from gevent.pywsgi import WSGIServer

class AlexNetCustom(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNetCustom, self).__init__()
        self.alexnet = alexnet(pretrained=False)
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)
    
class VGG16Custom(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16Custom, self).__init__()
        self.vgg16 = vgg16(pretrained=False)
        in_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)

class ResNetCustom(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetCustom, self).__init__()
        self.resnet = resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

app = Flask(__name__)
AlexNet = AlexNetCustom(num_classes=2)
AlexNet.load_state_dict(torch.load('models/alexnet_model_state_dict.pth'))

VGG = VGG16Custom(num_classes=2)
VGG.load_state_dict(torch.load('models/vgg16_model_state_dict.pth'))

ResNet = ResNetCustom(num_classes=2)
ResNet.load_state_dict(torch.load('models/resnet_model_state_dict.pth'))

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/set_model/<model_name>',methods=['POST','GET'])
def set_model(model_name):
    global Model_Path
    # Set model path based on the button clicked
    if model_name == 'AlexNet':
        Model_Path = 'models/alexnet_model_state_dict.pth'
    elif model_name == 'VGG':
        Model_Path = 'models/vgg16_model_state_dict.pth'
    elif model_name == 'ResNet':
        Model_Path = 'models/resnet_model_state_dict.pth'

    print(f"Selected model: {model_name}. Current model path: {Model_Path}")
    return render_template('index.html',model=model_name)

@app.route('/',methods=['POST','GET'])
def predict():
    imagefile= request.files["imagefile"]
    image_path ='./static/' + imagefile.filename
    imagefile.save(image_path)
    img = load_img(image_path,target_size=(224,224))
    preprocess = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)

    model_name = ""
    if 'alexnet' in Model_Path:
        AlexNet.eval()
        with torch.no_grad():
            output = AlexNet(img_tensor)
        model_name = "AlexNet"
    elif 'vgg16' in Model_Path:
        VGG.eval()
        with torch.no_grad():
            output = VGG(img_tensor)
        model_name = "VGG16"
    elif 'resnet' in Model_Path:
        ResNet.eval()
        with torch.no_grad():
            output = ResNet(img_tensor)
        model_name = "ResNet"

    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    if predicted_class == 1:
        classification = 'Positive'
    else:
        classification = 'Negative'

    print(f"Current model path: {Model_Path}")
    print(probabilities)

    return render_template('index.html', prediction=classification, imagePath=image_path, model_used=model_name)

if __name__ == '__main__':
    app.run(port=5000,debug=True)