import numpy as np
import os
from PIL import Image
import cv2

def load_dataset(dataset_path, mtcnn, resnet, device='cuda:0'):
    dataset, labels = [], []
    for label in os.listdir(dataset_path):
        for filename in os.listdir(os.path.join(dataset_path, label)):
            img_path = os.path.join(dataset_path, label, filename)
            img = Image.open(img_path).resize((448, 448))
            img_cropped = mtcnn(img)
            if img_cropped is not None:
                img_embedding = resnet(img_cropped.to(device))
                dataset.append(img_embedding.cpu().detach().numpy())
                labels.append(label)
    print('Process Dataset Successfully')
    return np.array(dataset), labels
