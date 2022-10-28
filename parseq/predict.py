import torch
from PIL import Image
from parseq.strhub.data.module import SceneTextDataModule
import glob

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

def get_predict(list_img: list, thres: float = 0.8):
    r = []
    for img in list_img:
        r.append(img_transform(img))
    r = torch.stack(r)
    print(r.shape)

    logits = parseq(r)
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    confidence = [float(torch.mean(i)) for i in confidence]
    label = [label[i] for i in range(len(label)) if confidence[i]>thres]
    return ' '.join(label)

if __name__ == '__main__':
    l = []
    for img_path in glob.glob('./tmp_images/imgs/*.PNG'):
        l.append(Image.open(img_path).convert('RGB'))

    get_predict(l)