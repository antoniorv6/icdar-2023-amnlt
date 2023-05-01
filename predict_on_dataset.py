import os
import cv2
import torch

import numpy as np

from fire import Fire
from rich import progress
from loguru import logger
from itertools import groupby
from torchvision import transforms
from ModelManager import get_model, LighntingE2EModelUnfolding
from utils import check_and_retrieveVocabulary

def load_images(images_path,reduce_ratio):
    images = []
    images_name = []
    for filename in progress.track(os.listdir(images_path), description=f"Loading {images_path}"):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(f"{images_path}/{filename}", 0)
            width = int(np.ceil(img.shape[1] * reduce_ratio))
            height = int(np.ceil(img.shape[0] * reduce_ratio))    
            img = cv2.resize(img, (width, height))
            images.append(img)
            images_name.append(filename)
    
    return images, images_name

@logger.catch
def main(images_path, model, checkpoint_path, corpus_name, output_path, reduce_ratio=0.5):

    logger.info(f"Performing predictions with the following data: \n Images path {images_path} \n Images reduce ratio {reduce_ratio} \n Model: {model} \n Weights Path: {checkpoint_path} \n Corpus name {corpus_name}")

    os.makedirs(output_path, exist_ok=True)
    
    imgs, names = load_images(images_path, reduce_ratio)
    
    m_width = np.max([img.shape[1] for img in imgs])
    m_height = np.max([img.shape[0] for img in imgs])

    _, i2w = check_and_retrieveVocabulary([], "vocab/", f"{corpus_name}")

    model, torchmodel = get_model(maxwidth=m_width, maxheight=m_height, 
                      in_channels=1, out_size=len(i2w)+1, 
                      blank_idx=len(i2w), i2w=i2w, 
                      model_name=model, output_path="")

    model = LighntingE2EModelUnfolding.load_from_checkpoint(checkpoint_path, model=torchmodel)
    model.eval()
    tensorTransform = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for x, outname in progress.track(zip(imgs, names), description=f"Performing transcription"):
            x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
            x = cv2.flip(x, 1)
            x = tensorTransform(x).to(device)
            pred = model(x.unsqueeze(0))
            pred = pred.permute(1,0,2).contiguous()
            pred = pred[0]
            out_best = torch.argmax(pred,dim=1)
            out_best = [k for k, g in groupby(list(out_best))]
            decoded = []
            for c in out_best:
                if c.item() != len(i2w):
                    decoded.append(c.item())
        
            decoded = [i2w[tok] for tok in decoded]

            decoded = "".join(decoded)
            decoded = decoded.replace("<t>", "\t")
            decoded = decoded.replace("<b>", "\n")

            with open(f"{output_path}/{outname.split('.')[0]}.krn", "w+") as krnfile:
                krnfile.write(decoded)
    
    logger.success(f"Transcription performed correctly and outputted in {output_path}")
    
if __name__ == "__main__":
    Fire(main)
