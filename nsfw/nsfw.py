from nsfw.nsfw_model import NSFW 
from nsfw.yolov5 import get_human, get_flag, get_weapon, get_crypto
from nsfw.crop_human import human_filter, convert
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

def draw_image(img_path, out_yolo):
    img = cv2.imread(img_path)
    h,w,_ = img.shape
    for lst in out_yolo:
        cls, x1,x2,y1,y2 = convert(w, h, lst)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
    name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
    cv2.imwrite('./static/uploads/'+name, img)

def detect_flag(img_path, draw = False):
    ban_list = ['ba_que', 'my', 'nga', 'trieu_tien', 'trung_quoc', 'ukraina', 'viet_nam',]
    # img_path = os.path.join(root_image_path, img_path)
    out_yolo = get_flag(img_path=img_path)
    if out_yolo is None:
        return False 
    if draw:
        draw_image(img_path, out_yolo)

    names = [i[0] for i in out_yolo]
    ban_names = [i for i in names if i in ban_list]
    return ban_names

def detect_weapon(img_path, draw = False):
    # img_path = os.path.join(root_image_path, img_path)
    out_yolo = get_weapon(img_path=img_path)
    if out_yolo is None:
        return [] 
    if draw:
        draw_image(img_path, out_yolo)
    names = [i[0] for i in out_yolo]
    return names

def detect_crypto(img_path, draw = False):
    # img_path = os.path.join(root_image_path, img_path)
    out_yolo = get_crypto(img_path=img_path)
    if out_yolo is None:
        return [] 
    if draw:
        draw_image(img_path, out_yolo)
    names = [i[0] for i in out_yolo]
    return names

def detect_nsfw(img_path, draw = False):
    out_yolo = get_human(img_path=img_path)

    if out_yolo is None:
        return False
    img = cv2.imread(img_path)   
    
    image_draw = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    cordinates = 0
    cordinates = human_filter(w=w, h=h, lst=out_yolo, return_only_biggest_box=False)
    if cordinates:
        for cordinate in cordinates:
            x1,x2,y1,y2 = cordinate
            if draw:
                image_draw = cv2.rectangle(image_draw, (x1,y1), (x2,y2), (0,0,255), 1)
                name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
                cv2.imwrite('./static/uploads/'+name, image_draw)
            crop_rgb = img_rgb[y1:y2, x1:x2]
            crop_image = Image.fromarray(crop_rgb.astype('uint8'), 'RGB')
            
            #######save image to create data
            if crop_image.size[0]*crop_image.size[1]>=5000:
                if not os.path.isdir('./data_sexy'):
                    os.mkdir('./data_sexy')
                tmp_name = len(os.listdir('./data_sexy'))
                crop_image.save(f"./data_sexy/{tmp_name}.jpg")
                
                model = NSFW()
                result_nsfw = model.predict(crop_image)
                
                del model
                gc.collect()
                torch.cuda.empty_cache()
                if result_nsfw:
                    return result_nsfw 
        
    # cordinates = human_filter(w=w, h=h, lst=out_yolo, return_only_biggest_box=False)
    # print(">>> num human and hello", len(cordinates))
    # if cordinates:
    #     scores = []
    #     names = []
    #     xys = []
    #     threshold = 0.3
    #     for cor in cordinates:
    #         x1,x2,y1,y2 = cor
    #         crop_bgr = img[y1:y2, x1:x2]
    #         score, who = search_single_face(crop_bgr)
    #         scores.append(score)
    #         names.append(who)
    #         xys.append([x1,x2,y1,y2])
            
    #     if min(scores) < threshold:
    #         idx = np.argmin(scores)
    #         x1,x2,y1,y2 = xys[idx]
    #         color = (0, 255, 255)
    #         if draw:
    #             image_draw = cv2.rectangle(image_draw, (x1,y1), (x2,y2), color, 1)
    #             name = img_path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.gif', '')+'_.jpg'
    #             cv2.imwrite('./static/uploads/'+name, image_draw) 
    #         return names[idx]
    return False

if __name__ == "__main__":
    print(detect_nsfw('C:/Users/quyennt72/Desktop/nguyenxuanphuc2.jpg'))
        
    