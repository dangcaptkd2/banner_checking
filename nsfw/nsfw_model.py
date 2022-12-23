import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import timm

import torchvision.transforms.functional as F

import random
np.random.seed(42)

_FILL = (128,128,128)

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, _FILL, 'constant')


class NSFW():
    def __init__(self, config) -> None:
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model = None
        self.path_ckp = config['models']['sexy']
        self.softmax_layer = nn.Softmax(dim=1)
        self.class_names = config['models']['class_name_sexy']
        self.thres = config['threshold']['sexy']
        self.model_name = config['models']['name_model_sexy']
        if self.model_name == "coatnet" or 'student' in self.path_ckp:
            a = 256
            b = 224
        else:
            a = 128
            b = 112
        self.transform = transforms.Compose([
                                    SquarePad(),
                                    transforms.Resize(a),
                                    transforms.CenterCrop(b),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                            
    
    def load_model(self):
        if self.model is not None:
            return self.model
        if  self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=False)
            numrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(numrs, len(self.class_names))
        elif self.model_name == 'efficientnet_b1':
            if 'student' not in self.path_ckp:
                model = models.efficientnet_b1(pretrained=False)
                numrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(numrs, len(self.class_names))
            else:
                model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=len(self.class_names))
        elif self.model_name == 'coatnet':
            model = timm.create_model("coatnet_1_rw_224", pretrained=False, num_classes=len(self.class_names), drop_path_rate=0.05)

        model.load_state_dict(torch.load(self.path_ckp, map_location=torch.device('cpu')))
        model.to(self.device)
        model.eval()

        return model

    def predict(self, image):
        with torch.no_grad():
            x = self.transform(image)
            x = x.unsqueeze(0)
            x = x.to(self.device)
            self.model = self.load_model()
            r = self.model(x)
            re = self.softmax_layer(r)
            
            score , pred = torch.max(re, 1)
            pred = pred.cpu().detach().numpy()
            score = score.cpu().detach().numpy()
            if self.class_names[pred[0]] != 'neural' and score[0]>self.thres:
                return True, self.class_names[pred[0]], score[0]
            return False, 'neural', score[0]


# if __name__ == "__main__":
#     model = NSFW()
#     img = Image.open('C:/Users/quyennt72/Downloads/Banner_Block/15.png')
#     print(model.predict(img))

    