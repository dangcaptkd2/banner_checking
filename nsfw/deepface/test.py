import numpy as np

from scipy import spatial

from deepface.commons.functions import normalize_input
from deepface.DeepFace import build_model

model = build_model('Facenet512')

a = np.load('./database/nguyen_xuan_phuc.npy')
b = np.load('./database/pham_minh_chinh.npy')

pos = []
neg = []
for i, embedding1 in enumerate(a):
    for j, embedding2 in enumerate(a):
        dis = spatial.distance.cosine(embedding1, embedding2)
        pos.append(dis)
for i, embedding1 in enumerate(b):
    for j, embedding2 in enumerate(b):
        dis = spatial.distance.cosine(embedding1, embedding2)
        pos.append(dis)

for i, embedding1 in enumerate(a):
    for j, embedding2 in enumerate(b):
        dis = spatial.distance.cosine(embedding1, embedding2)
        neg.append(dis)

print(pos)
print('='*50)
print(neg)
