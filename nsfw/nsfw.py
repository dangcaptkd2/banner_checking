from nsfw.nsfw_model import NSFW 
from nsfw.yolov5.detect import YOLOV5
from nsfw.crop_human import human_filter, convert, convert_filter
# from nsfw.deepface import search_single_face

# from TextFuseNet.mid_process import 

import cv2 
from PIL import Image
import os

import gc 
import torch
import random 
import numpy as np

np.random.seed(42)

class IMAGE_DETECT(YOLOV5):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.draw = self.config["utils"]["draw"]
        self.check_politician = self.config["run"]["politician"]
        self.save_image = self.config['utils']['save_image']
        self.sexy_model = NSFW(config=self.config)
        self.checking_boob = self.config['utils']['checking_boob']

    def draw_image(self, out_yolo):
        img = cv2.imread(self.img_path)
        h,w,_ = img.shape
        for lst in out_yolo:
            cls, x1,x2,y1,y2, score = convert(w, h, lst)
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
        name = self.img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
        cv2.imwrite('./static/uploads/'+name, img)

    def detect_flag(self):
        ban_list = ['ba_que', 'my', 'nga', 'trieu_tien', 'trung_quoc', 'ukraina', 'viet_nam',]
        out_yolo = self.get_flag()
        if out_yolo is None:
            return False 
        if self.draw:
            self.draw_image(out_yolo)

        names = [i[0] for i in out_yolo]
        ban_names = [i for i in names if i in ban_list]
        return ban_names

    def detect_weapon(self):
        out_yolo = self.get_weapon()
        if out_yolo is None:
            return [] 
        if self.draw:
            self.draw_image(out_yolo)
        names = [i[0] for i in out_yolo]
        return names

    def detect_crypto(self):
        out_yolo = self.get_crypto()
        if out_yolo is None:
            return [] 
        if self.draw:
            self.draw_image(out_yolo)
        names = [i[0] for i in out_yolo]
        return names

    def detect_nsfw(self):
        path_save = self.config["path_save"]

        if not os.path.isdir(path_save['human4boob_detect']):
            os.mkdir(path_save['human4boob_detect'])
        if self.save_image:
            if not os.path.isdir(path_save['neural']):
                os.mkdir(path_save['neural'])
            if not os.path.isdir(path_save['nude']):
                os.mkdir(path_save['nude'])
            if not os.path.isdir(path_save['bikini']):
                os.mkdir(path_save['bikini'])
            if not os.path.isdir(path_save['sexy_half']):
                os.mkdir(path_save['sexy_half'])

        out_yolo = self.get_human()

        if out_yolo is None:
            return None
        img = cv2.imread(self.img_path)  
        h,w,_ = img.shape  
        
        image_draw = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        cordinates = 0
        cordinates = human_filter(w=w, h=h, lst=out_yolo, return_only_biggest_box=False)
        if not cordinates:
            return None
        for cordinate in cordinates:
            x1,x2,y1,y2 = cordinate
            if self.draw:
                image_draw = cv2.rectangle(image_draw, (x1,y1), (x2,y2), (0,0,255), 1)
                name = self.img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
                cv2.imwrite('./static/uploads/'+name, image_draw)
            crop_rgb = img_rgb[y1:y2, x1:x2]
            crop_image = Image.fromarray(crop_rgb.astype('uint8'), 'RGB')
            
            if crop_image.size[0]*crop_image.size[1]>=self.config['threshold']['size_human']:
                result_nsfw, name_sexy, score_sexy = self.sexy_model.predict(crop_image)

                if result_nsfw:
                    result_boob = None
                    if self.checking_boob:
                        name = len(os.listdir(path_save['human4boob_detect']))
                        path_ = "{}/{}.jpg".format(path_save['human4boob_detect'], name)
                        crop_image.save(path_)
                        result_boob = self.get_boob(boob_img_path=path_)
                        if result_boob is not None:
                            if self.save_image:
                                tmp_name = len(os.listdir(path_save[name_sexy]))
                                crop_image.save(f"{path_save[name_sexy]}/{tmp_name}.jpg")

                            human_img_array = cv2.imread(path_)
                            h,w,_ = human_img_array.shape
                            boob_cor = convert_filter(w, h, result_boob)
                            for cor in boob_cor:
                                x1,x2,y1,y2 = cor
                                human_img_array = cv2.rectangle(human_img_array, (x1,y1), (x2,y2), (255,0,0), 1)
                                name = self.img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'__.jpg'
                            cv2.imwrite('./static/uploads/'+name, human_img_array)

                            return result_nsfw 
                        else:
                            if self.save_image:
                                tmp_name = len(os.listdir(path_save['sexy_half']))
                                crop_image.save(f"{path_save['sexy_half']}/{tmp_name}.jpg")
                    else:
                        if self.save_image and score_sexy>0.7 and crop_image.size[0]*crop_image.size[1]>150*150:
                            tmp_name = len(os.listdir(path_save[name_sexy]))
                            crop_image.save(f"{path_save[name_sexy]}/{tmp_name}.jpg")
                        return result_nsfw 
                else:
                    if self.save_image and crop_image.size[0]*crop_image.size[1]>200*200 and score_sexy>0.7:
                        tmp_name = len(os.listdir(path_save[name_sexy]))
                        if tmp_name<=2000:
                            crop_image.save(f"{path_save[name_sexy]}/{tmp_name}.jpg")
        #### politician
        # if self.check_politician:
        #     cordinates = human_filter(w=w, h=h, lst=out_yolo, return_only_biggest_box=False)
        #     if cordinates:
        #         scores = []
        #         names = []
        #         xys = []
        #         threshold = self.config['threshold']['same_face']
        #         for cor in cordinates:
        #             x1,x2,y1,y2 = cor
        #             crop_bgr = img[y1:y2, x1:x2]
        #             score, who = search_single_face(crop_bgr)
        #             scores.append(score)
        #             names.append(who)
        #             xys.append([x1,x2,y1,y2])
                    
        #         if min(scores) < threshold:
        #             idx = np.argmin(scores)
        #             x1,x2,y1,y2 = xys[idx]
        #             color = (0, 255, 255)
        #             if self.draw:
        #                 image_draw = cv2.rectangle(image_draw, (x1,y1), (x2,y2), color, 1)
        #                 name = self.img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
        #                 cv2.imwrite('./static/uploads/'+name, image_draw) 
        #             return names[idx]
        # return False

if __name__ == "__main__":
    print("hello world")