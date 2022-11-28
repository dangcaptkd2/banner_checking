from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import os
from PIL import Image
import yaml

# sys.path.append()

class RECOGNITION_VN():
    def __init__(self, device) -> None:
        curdir = os.path.dirname(__file__)

        cfg_vgg = os.path.join(curdir, 'config', 'vgg-transformer.yml')        
        cfg_base = os.path.join(curdir, 'config', 'base.yml')
        
        with open(cfg_base, encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
            
        with open(cfg_vgg, encoding='utf-8') as f:
            vgg_config = yaml.safe_load(f)
          
        base_config.update(vgg_config)
        self.config = Cfg(base_config)
        
        self.config['weights'] = './models/transformerocr.pth'
        self.config['cnn']['pretrained']=False
        self.config['device'] = device
        self.config['predictor']['beamsearch']=False

        self.model = None

    def get_model(self):
        if self.model is not None:
            return self.model         
        self.model = Predictor(self.config)
        return self.model


    def predict(self, image_array, thres=0.6):
        model = self.get_model()
        text_boundings = [Image.fromarray(B) for B in image_array]
        texts, scores = model.predict_batch(text_boundings,return_prob=True)
        final_text = [texts[i] for i in range(len(texts)) if scores[i]>thres]
        return final_text
    
