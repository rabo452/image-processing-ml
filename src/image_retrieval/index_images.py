import os 
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
import numpy as np
from torchvision import datasets

from image_retrieval.ModelInformationHelper import ModelInformationHelper
from image_retrieval.train_model import loadModel, loadPreproccessor
from image_retrieval.vector_database import vectorDBClient as client
from image_retrieval.const import MODEL_VECTOR_DIMENSION, IMAGE_COLLECTION_NAME 

def loadImage(path):
    return Image.open(path).convert("RGB")

def getImageVector(model, preprocess, image):
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    # Extract features
    with torch.no_grad():
        vector = model(input_tensor)  # This is the feature map before flattening

    return np.array(vector.view(vector.size(0), -1).squeeze()) # Flatten to a 1D vector and convert to np array

def indexImages(imagesPath: str):
    dataset = datasets.ImageFolder(root=imagesPath, transform=loadPreproccessor())
    images = [
    {
        'img_path': os.path.relpath(sample[0], start=imagesPath),
        'class': dataset.classes[sample[1]],
    }
    for sample in dataset.samples
    ]

    n_classes = ModelInformationHelper.loadModelInformation()['class_count']
    model = loadModel(n_classes)

    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    preprocess = loadPreproccessor()

    loadImgFn = lambda imgObj: {
        **imgObj, 
        'vector': getImageVector(
            model, 
            preprocess, 
            loadImage(os.path.join(imagesPath, imgObj['img_path']))
        )
    }
    with ThreadPoolExecutor(max_workers=16) as executor:
        images = list(executor.map(loadImgFn, images))
        
        if not client.has_collection(IMAGE_COLLECTION_NAME):
            client.create_collection(IMAGE_COLLECTION_NAME, dimension=MODEL_VECTOR_DIMENSION, auto_id=True)
        client.insert(collection_name=IMAGE_COLLECTION_NAME, data=images)