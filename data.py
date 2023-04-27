import gin
import torch
import cv2
import os
import re

import numpy as np

from loguru import logger
from rich import progress
from torch.utils.data import Dataset
from torchvision import transforms
from utils import check_and_retrieveVocabulary


@logger.catch
def batch_preparation_ctc(data):
    images = [sample[0] for sample in data]
    gt = [sample[1] for sample in data]
    L = [sample[2] for sample in data]
    T = [sample[3] for sample in data]

    max_image_width = max([img.shape[2] for img in images])
    max_image_height = max([img.shape[1] for img in images])

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        c, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])
    Y_train = torch.zeros(size=[len(gt),max_length_seq])
    for i, seq in enumerate(gt):
        Y_train[i, 0:len(seq)] = torch.from_numpy(np.asarray([char for char in seq]))

    return X_train, Y_train, L, T


@gin.configurable
def load_set(path, reduce_ratio=0.5):
    x = []
    y = []
    for filename in progress.track(os.listdir(path)):
        if filename.endswith(".krn"):
            with open(f"{path}/{filename}") as krnfile:
                krn_content = krnfile.read()
                img = cv2.imread(f"{path}/{filename.split('.')[0]}.png", 0)
                width = int(np.ceil(img.shape[1] * reduce_ratio))
                height = int(np.ceil(img.shape[0] * reduce_ratio))    
                img = cv2.resize(img, (width, height))
                y.append([content + '\n' for content in krn_content.split("\n")])
                x.append(img)

    return x, y

class AMNLTDataset(Dataset):
    def __init__(self, data_path, set_name) -> None:
        self.x, self.y = load_set(f"{data_path}{set_name}")
        self.x = self.preprocess_images(self.x)
        self.y = self.preprocess_gt(self.y)
        
        self.tensorTransform = transforms.ToTensor()

        self.w2i, self.i2w = None, None
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        image = self.tensorTransform(self.x[index])
        gt = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        
        return image, gt, (image.shape[2] // 4) * (image.shape[1] // 32), len(gt)

    
    def preprocess_images(self, X, flip=True):
        for idx, image in enumerate(X):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if flip:
                image = cv2.flip(image, 1)
            X[idx] = image

        return X

    def preprocess_gt(self, Y):
        for idx, sequence in enumerate(Y):
            nseq = []
            for jdx, element in enumerate(sequence[1:]):
                element = element.replace("\t", " <t> ")
                element = element.replace("\n", " <b>")
                element = element.split(" ")    
                for zdx, token in enumerate(element):
                    if zdx == 2:
                        if jdx > 0 and jdx < len(sequence)-2:
                            for character in element[zdx]:
                                nseq.append(character)
                        else:
                            nseq.append(token)
                    else:
                        nseq.append(token)
        
            strseq = " ".join(nseq)
            strseq = re.sub(r'\. <t> \. <b> ', '', strseq)
            nseq = strseq.split(" ")
            Y[idx] = nseq

        return Y
    
    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_width, m_height

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
    
    def get_i2w(self):
        return self.i2w


def load_dataset(data_path=None, corpus_name=None):

    train_dataset = AMNLTDataset(data_path=data_path, set_name="train")
    val_dataset = AMNLTDataset(data_path=data_path, set_name="val")
    test_dataset = AMNLTDataset(data_path=data_path, set_name="test")

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()], "vocab/", f"{corpus_name}")

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset



