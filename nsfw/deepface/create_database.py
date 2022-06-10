from deepface.commons.functions import preprocess_face, find_input_shape, normalize_input
from deepface.DeepFace import build_model

import os 
from glob import glob 
import numpy as np
from tqdm import tqdm
import cv2

root = 'C:/Users/quyennt72/Desktop/chinh_khach'
database = './database'
model = build_model('Facenet512')
input_shape_x, input_shape_y = find_input_shape(model)

for name in tqdm(os.listdir(root)):
    list_path = glob(os.path.join(root, name, '*.jpg'))
    result = []
    for i in tqdm(range(len(list_path)), leave=True):
        image = cv2.imread(list_path[i])
        img = preprocess_face(img=image, detector_backend='retinaface', target_size=(input_shape_y, input_shape_x), enforce_detection=True)
        img = normalize_input(img = img, normalization = 'Facenet2018')
        embedding = model.predict(img)[0].tolist()
        result.append(embedding)
    
    with open(f'./database/{name}.npy', 'wb') as f:
        np.save(f, result)

