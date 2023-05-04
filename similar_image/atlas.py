import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np

ATLAS_FILE = './data/data_atlas.npy'

class SIMILAR_MODEL:
    resnet50 = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(resnet50.children())[:-1])
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
    
    def check_similar(self, thres=0.77, img_path=None):
        v = self._get_embedding(img_path=img_path)
        print("???", v)
        cos_sim_list = cosine_similarity([v], self.data)
        score = max(cos_sim_list[0])
        print(">>>", score)
        if score>=thres:
            return True 
        return False


if __name__ == '__main__':
    module = SIMILAR_MODEL()
    from glob import glob 
    import cv2
    # for img_path in tqdm(glob('/home/quyennt72/banner_checking_fptonline/static/uploads/*.jpg')):
    #     if module.check_similar(img_path=img_path):
    #         print(img_path)
    img_path = '/home/quyennt72/banner_checking_fptonline/static/uploads/342484722_1267069444203793_888146610111216141_n (1).png'
    a = module.check_similar(img_path=img_path)
    print(module.data[0])
    
