from yolov5 import get_human, get_weapon, get_crypto
from deepface import search_face

img_path = '/home/quyennt72/banner_checking_fptonline/static/uploads/crypto_ex.jpg'

print(get_crypto(img_path))