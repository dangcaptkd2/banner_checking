import torch
import logging

class FISH:
    model_boob = None 
    model_flag = None 
    model_crypto = None 
    model_weapon = None
    model_human = None
    def __init__(self, cfg) -> None:
        self.path_model_boob = cfg['models']['detect_boob']
        self.path_model_flag = cfg['models']['detect_flag']
        self.path_model_crypto = cfg['models']['detect_crypto']
        self.path_model_weapon = cfg['models']['detect_weapon']
        self.path_model_human = cfg['models']['detect_human']

        self.thres_boob = cfg['threshold']['detect_boob']
        self.thres_flag = cfg['threshold']['detect_flag']
        self.thres_crypto = cfg['threshold']['detect_crypto']
        self.thres_weapon = cfg['threshold']['detect_weapon']
        self.thres_human = cfg['threshold']['detect_human']

    def get_img_path(self, img_path):
        self.img_path = img_path

    def __get_model(self, model, path):
        if model is not None:
            return model
        model = torch.hub.load('/servers/hubs/yolov5', 'custom', path, source='local')
        return model

    def __predict(self, model, image=None, thres=0.5, imgz=320, name=None):
        if image is None:
            image = self.img_path
        r=[]
        result = model(image, size=imgz)
        results = result.crop(save=False)
        for dic in results:
            box = dic['box']
            box = [int(i) for i in box] 
            conf = float(dic['conf'])
            cls = int(dic['cls'])
            if conf>thres:
                if name=='human':
                    if cls==0:
                        r.append([cls, box, conf])
                else:
                    r.append([cls, box, conf])
        logging.getLogger('root').debug(f"Detect {name}: {r}")
        return r

    def get_boob(self, image=None):
        self.model_boob = self.__get_model(model=self.model_boob, path=self.path_model_boob)
        return self.__predict(self.model_boob, image, thres=self.thres_boob, imgz=160, name='boob')
        
    def get_flag(self):
        self.model_flag = self.__get_model(model=self.model_flag, path=self.path_model_flag)
        return self.__predict(self.model_flag, thres=self.thres_flag, name='flag')

    def get_crypto(self):
        self.model_crypto = self.__get_model(model=self.model_crypto, path=self.path_model_crypto)
        return self.__predict(self.model_crypto, thres=self.thres_crypto, name='crypto')

    def get_weapon(self):
        self.model_weapon = self.__get_model(model=self.model_weapon, path=self.path_model_weapon)
        return self.__predict(self.model_weapon, thres=self.thres_weapon, name='weapon')
    
    def get_human(self):
        self.model_human = self.__get_model(model=self.model_human, path=self.path_model_human)
        return self.__predict(self.model_human, thres=self.thres_human, name='human')

if __name__ == '__main__':
    import yaml 
    with open('/home/quyennt72/banner_checking_fptonline/configs/common.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    module = FISH(cfg=config) 
    img_path = '/home/quyennt72/banner_checking_fptonline/static/uploads/2023.jpg'
    module.get_img_path(img_path=img_path)
    result = module.get_human()
    print(result)