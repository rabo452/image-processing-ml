import os
import json 

import torch

class ModelInformationHelper:
    __modelInformationDir = os.path.join(os.getcwd(), 'model_information')
    __weightPath = f'{__modelInformationDir}/weights.pth'
    __fisherPath = f'{__modelInformationDir}/fisher.pt'
    __modelInformationPath = f'{__modelInformationDir}/model_information.json'

    @classmethod
    def initInformationDir(cls):
        if not (os.path.exists(cls.__modelInformationDir) and os.path.isdir(cls.__modelInformationDir)):
            os.mkdir(cls.__modelInformationDir)

    @classmethod
    def saveModelWeights(cls, weights):
        cls.initInformationDir()
        torch.save(weights, cls.__weightPath)
    
    @classmethod
    def saveFisherInformation(cls, fisherInformation):
        cls.initInformationDir()
        torch.save(fisherInformation, cls.__fisherPath)
    
    @classmethod
    def saveModelInformation(cls, information: dict):
        cls.initInformationDir()
        jsonText = json.dumps(information)
        f = open(cls.__modelInformationPath, 'w+')
        f.write(jsonText)
        f.close()
    
    @classmethod
    def loadModelInformation(cls) -> dict|None:
        modelInformation = None 
        if os.path.exists(cls.__modelInformationPath) and os.path.isfile(cls.__modelInformationPath):
            f = open(cls.__modelInformationPath, 'r+')
            modelInformation = json.loads(f.read())
            f.close()
        return modelInformation

    @classmethod
    def loadModelWeights(cls):
        weights = None
        if os.path.exists(cls.__weightPath) and os.path.isfile(cls.__weightPath):
            weights = torch.load(cls.__weightPath, weights_only=True)
        return weights
    
    @classmethod
    def loadFisherInformation(cls):
        fisherInformation = None
        if os.path.exists(cls.__fisherPath) and os.path.isfile(cls.__fisherPath):
            fisherInformation = torch.load(cls.__fisherPath, weights_only=True)
        return fisherInformation