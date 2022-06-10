from fileinput import filename
import os
import cv2
from tqdm import tqdm
import shutil
import json
from PIL import Image

os.makedirs('./debugs/crop_images', exist_ok=True)

def up(a, b):
  if a[1] >= b[3]:
    return True 
  return False

def beside(a, b):
  if a[0] >= b[0]:
    if (b[1] <= a[1] and a[1]<=b[3]) or (b[1] <= a[3] and a[3]<= b[3]) or (a[1] <= b[1] and b[1] <= a[3]):
      return True
  return False

def swap_dic(dic, key_a, key_b):
    temp = dic[key_a]
    dic[key_a] = dic[key_b]
    dic[key_b] = temp

def check_near_box(l1, l2, thres_x=10, thres_y=5):
  x1,y1,x2,y2 = l1
  x1_,y1_,x2_,y2_ = l2

  if abs(x2-x1_)<thres_x and abs(y1-y1_)<thres_y and abs(y2-y2_)< thres_y:
    return True
  return False

def merge_2_boxes(d, id1, id2):
  x1,y1,x2,y2 = d[id1]
  x1_,y1_,x2_,y2_ = d[id2]

  new_cor = [x1,y1,x2_,y2_]
  d[id1] = new_cor
  del d[id2]
  return d

def merge(dic):
  d=dic
  sta = 0
  end = 1
  max_id = len(d)
  while end<max_id:
    if check_near_box(d[sta], d[end]):
      d = merge_2_boxes(d, sta, end) 
      end+=1
    else:
      sta = end
      end = sta+1
  return d

def action_merge(sorted_cor, name, image_path):

  img = cv2.imread(image_path)

  des = './debugs/crop_images/' + name
  if os.path.isdir(des):
    shutil.rmtree(des)
  os.makedirs(des)

  new_r = merge(sorted_cor[name])
  key = sorted(list(map(int, list(new_r.keys()))))
  final_dict = {}
  for idx, key in enumerate(key):
    final_dict[idx] = new_r[key]

  for key, box in final_dict.items():
    crop = img[box[1]:box[3], box[0]:box[2]]
    final_path = des + '/' + str(key) +'.jpg'
    cv2.imwrite(final_path, crop)
  
def mid_process(name, path_image, result_detect):
  print('>>>> name in mid process:', name)
  cor = result_detect

  sorted_cor = {}
  for n, dic in list(cor.items()):
    list_key = list(dic.keys())
    for i in range(len(list_key)-1):
      for j in range(i+1, len(list_key)):
          if (up(dic[list_key[i]], dic[list_key[j]]) or beside(dic[list_key[i]], dic[list_key[j]])):
            swap_dic(dic, list_key[i], list_key[j])
    sorted_cor[n] = dic

  image = cv2.imread(path_image)

  list_arr = []

  for index, box in sorted_cor[name].items():
      crop = image[box[1]:box[3], box[0]:box[2]]

      list_arr.append(Image.fromarray(crop.astype('uint8'), 'RGB').convert('L'))
      # final_path = des + '/' + str(index) +'.jpg'
      # cv2.imwrite(final_path, crop)

  #action_merge(sorted_cor, name, image)

  return list_arr, sorted_cor