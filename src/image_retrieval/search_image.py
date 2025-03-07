from PIL import Image
import torch
import torchvision.transforms as transforms

from image_retrieval.ModelInformationHelper import ModelInformationHelper
from image_retrieval.index_images import getImageVector
from image_retrieval.train_model import loadModel, loadPreproccessor
from image_retrieval.vector_database import vectorDBClient as client
from image_retrieval.const import IMAGE_COLLECTION_NAME

def getClosestImages(image: Image):
    n_classes = ModelInformationHelper.loadModelInformation()['class_count']
    model = loadModel(n_classes)

    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    preprocess = loadPreproccessor()

    vector = getImageVector(model, preprocess, image)
    result = list(client.search(
        collection_name=IMAGE_COLLECTION_NAME,
        data=[vector],
        limit=10,
        output_fields=['class', 'img_path']
    )[0])
    sortedResult = [*reversed(sorted(result, key=lambda image: image['distance']))]
    return sortedResult