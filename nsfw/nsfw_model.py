import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

import torchvision.transforms.functional as F

import random
# torch.manual_seed(42)
# random.seed(42)
np.random.seed(42)

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')


class NSFW():
    def __init__(self, config) -> None:
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model = None
        self.path_ckp = config['models']['sexy']
        self.transform = transforms.Compose([
                                        SquarePad(),
                                        transforms.Resize(128),
                                        transforms.CenterCrop(112),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
        self.softmax_layer = nn.Softmax(dim=1)
        self.class_names = config['models']['class_name_sexy']
        self.thres = config['threshold']['sexy']
        self.model_name = config['models']['name_model_sexy']
    
    def load_model(self):
        if self.model is not None:
            return self.model
        if  self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
        elif self.model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=True)
        numrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(numrs, len(self.class_names))

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
            # re = torch.sigmoid(r)
            print(">>>> result nsfw:", re)
            score , pred = torch.max(re, 1)
            pred = pred.cpu().detach().numpy()
            score = score.cpu().detach().numpy()
            print("score sexy:", score, pred)
            if self.class_names[pred[0]] != 'neural' and score[0]>self.thres:
                return True 
            return False


# if __name__ == "__main__":
#     model = NSFW()
#     img = Image.open('C:/Users/quyennt72/Downloads/Banner_Block/15.png')
#     print(model.predict(img))

    