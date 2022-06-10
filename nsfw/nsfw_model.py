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
    def __init__(self) -> None:
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model = None
        self.path_ckp = '../models/my_model_nsfw_42.pt'
        self.transform = transforms.Compose([
                                        SquarePad(),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
        self.softmax_layer = nn.Softmax(dim=1)
        self.class_names = ['neural', 'sexy']
    
    def load_model(self):
        if self.model is not None:
            return self.model
        model = models.resnet18(pretrained=True)
        numrs = model.fc.in_features
        model.fc = nn.Linear(numrs, 2)
        model.load_state_dict(torch.load(self.path_ckp, map_location=torch.device('cpu')))
        model.to(self.device)
        model.eval()

        return model

    def predict(self, image, thres=0.5):
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
            # print(score)
            # print(self.class_names[pred[0]])
            if self.class_names[pred[0]] == 'sexy' and score[0]>thres:
                return True 
            return False
        # return {"status_sexy": self.class_names[pred[0]]}


# if __name__ == "__main__":
#     model = NSFW()
#     img = Image.open('C:/Users/quyennt72/Downloads/Banner_Block/15.png')
#     print(model.predict(img))

    