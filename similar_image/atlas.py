import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np

from similar_image.model import Network

ATLAS_FILE = './data/data_atlas_0805.npy'
PATH_SAVE_MODEL = './models/model_similar_atlas.pt'

class SIMILAR_MODEL:
    model = Network()
    model.load_state_dict(torch.load(PATH_SAVE_MODEL, map_location=torch.device('cpu')))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __init__(self) -> None:
        self.data = np.load(ATLAS_FILE)
        self.model.eval()

    def _get_embedding(self,img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)
        features =  self.model(img_tensor)
        features = features.squeeze().detach().numpy()
        return features
    
    def check_similar(self, thres=0.9999, img_path=None):
        v = self._get_embedding(img_path=img_path)
        cos_sim_list = cosine_similarity([v], self.data)
        score = max(cos_sim_list[0])
        print(">>>", min(cos_sim_list[0]), max(cos_sim_list[0]))
        if score>=thres:
            return True 
        return False


if __name__ == '__main__':
    module = SIMILAR_MODEL()
    img_path = '/home/quyennt72/banner_checking_fptonline/static/uploads/342484722_1267069444203793_888146610111216141_n (1).png'
    a = module.check_similar(img_path=img_path)
    print(module.data[0])
    
