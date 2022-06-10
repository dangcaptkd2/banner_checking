from yolov5 import get_human, get_weapon
from deepface import search_face

img_path = '/home/quyennt72/banner_checking_fptonline/static/uploads/sungluc.jpg'

print(get_weapon(img_path))