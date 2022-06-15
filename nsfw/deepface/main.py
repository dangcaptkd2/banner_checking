from cv2 import threshold
import numpy as np
import os

from scipy import spatial

# import sys 
# sys.path.append(os.path.abspath(os.path.join('..', 'nsfw')))
# sys.path.append(os.path.abspath(os.path.join('.', 'deepface')))
# sys.path.append(os.path.abspath(os.path.join('.', 'deepface/deepface')))

# for i in sys.path:
#     print(i)

from deepface.deepface.commons.functions import normalize_input
from deepface.deepface.DeepFace import build_model
from deepface.deepface.commons.functions import preprocess_face, find_input_shape
from deepface.deepface.DeepFace import build_model

model = build_model('Facenet512')
input_shape_x, input_shape_y = find_input_shape(model)

path_db = './deepface/database/'
# a = np.load('./database/nguyen_xuan_phuc.npy')
# b = np.load('./database/pham_minh_chinh.npy')
db = {}
for filename in os.listdir(path_db):
    name = filename[:-4]
    db[name] = np.load(f'./deepface/database/{filename}')

def embedding_single_face(image_query_path):
    img = preprocess_face(img=image_query_path, detector_backend='retinaface', target_size=(input_shape_y, input_shape_x), enforce_detection=False)
    img = normalize_input(img = img, normalization = 'Facenet2018')
    embedding1 = model.predict(img)[0].tolist()
    return embedding1

def search_single_face(image_query_path):
    threshold = 0.2553
    embed_query = embedding_single_face(image_query_path)
    
    lst_name = list(db.keys())
    scores = []
    for name in lst_name:
        tmp = []
        for embed in db[name]:
            dis = spatial.distance.cosine(embed_query, embed)
            tmp.append(dis)
        tmp = np.mean(tmp)
        scores.append(tmp)
    return min(scores), lst_name[np.argmin(scores)]
