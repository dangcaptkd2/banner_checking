import torch
from PIL import Image
from parseq.strhub.data.module import SceneTextDataModule
import glob



class RECOGNITION():
    def __init__(self, device='cpu') -> None:
        self._device=device
        # Load model and image transforms
        self.parseq = torch.hub.load('/servers/hubs/parseq', 'parseq', pretrained=True, source='local').eval().to(self._device)
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def predict(self, list_img: list, thres: float = 0.8):
        r = []
        for img in list_img:
            r.append(self.img_transform(img))
        r = torch.stack(r).to(self._device)

        logits = self.parseq(r)
        pred = logits.softmax(-1)
        label, confidence = self.parseq.tokenizer.decode(pred)
        confidence = [float(torch.mean(i)) for i in confidence]
        label = [label[i] for i in range(len(label)) if confidence[i]>thres]
        return ' '.join(label)

if __name__ == '__main__':
    l = []
    for img_path in glob.glob('./tmp_images/imgs/*.PNG'):
        l.append(Image.open(img_path).convert('RGB'))

    model = RECOGNITION()
    print(model.predict(l))