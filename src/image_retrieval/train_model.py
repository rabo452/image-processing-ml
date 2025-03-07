import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam

from image_retrieval.const import MODEL_IMAGE_RESOLUTION
from image_retrieval.ModelInformationHelper import ModelInformationHelper

# Function to calculate Fisher Information
def compute_fisher_information(model, data_loader, criterion, optimizer):
    fisher_information = {}
    
    # Set model to evaluation mode to prevent dropout/batch norm updates
    model.eval()
    
    # Initialize Fisher Information for each parameter as zeros
    for param_name, param in model.named_parameters():
        # fc.weights and fc.bias are the last layer bis and weights which are replaced on every training, so it's not required to store fisher information for them
        if 'fc.' in param_name:
            continue
        fisher_information[param_name] = torch.zeros_like(param)
    
    # Accumulate squared gradients (Fisher Information) over the entire dataset
    for inputs, labels in data_loader:
        # Forward pass and compute loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass to compute gradients
        loss.backward()

        # Accumulate Fisher Information (squared gradients) for each parameter
        for param_name, param in model.named_parameters():
            # fc.weights and fc.bias are the last layer bis and weights which are replaced on every training, so it's not required to store fisher information for them
            if 'fc.' in param_name:
                continue
            fisher_information[param_name] += param.grad ** 2

        # Zero the gradients after each batch
        optimizer.zero_grad()

    # Normalize Fisher Information by the number of batches
    num_batches = len(data_loader)
    for param_name in fisher_information:
        fisher_information[param_name] /= num_batches
    
    return fisher_information

def ewc_loss(model, fisher_information, previous_params, lambda_ewc=1000):
    loss = 0
    
    # Iterate through model parameters using vectorized computation
    for param_name, param in model.named_parameters():
        if param_name in fisher_information:
            fisher = fisher_information[param_name]
            param_diff = param - previous_params[param_name]
            
            # Compute the EWC penalty in a vectorized manner
            loss += torch.sum(fisher * (param_diff ** 2)) / 2
    
    # Scale the loss by lambda_ewc
    return lambda_ewc * loss

def train_model(model: models.ResNet, trainedClasses: int, 
                dataset: datasets.ImageFolder, fisherInformation = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learningRate = 1e-4
    newClasses = len(dataset.classes)
    # Replace the classifier with the correct number of output classes
    model.fc = nn.Linear(model.fc.in_features, trainedClasses + newClasses)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=learningRate)

    # Prepare DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    previousWeights = {name: param.clone().to(device) for name, param in model.state_dict().items()}

    bestWeights = previousWeights
    leastLoss = float('inf')

    # Training Loop
    num_epochs = 4
    model.train()
    for epoch in range(num_epochs):
        epoch += 1
        total = correct = running_loss = 0
        i = 1
        
        for inputs, labels in train_loader:
            # Move data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)
            labels += trainedClasses
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if fisherInformation is not None:
                loss += ewc_loss(model, fisherInformation, previousWeights)

            loss.backward()
            optimizer.step()

            loss_calc = loss.item()

            if loss_calc <= leastLoss:
                bestWeights = {name: param.clone().to(device) for name, param in model.state_dict().items()}
                leastLoss = loss_calc

            running_loss += loss_calc
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f'Epoch {epoch}, trained on batch {i}, loss: {loss_calc}')
            i += 1
    
        print(f"Epoch {epoch}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct/total}%")
    
    print(f'The least found loss: {leastLoss}')
    model.load_state_dict(bestWeights)
    print('computing fisher information')
    fisherInformation = compute_fisher_information(model, train_loader, criterion, optimizer)
    return model, fisherInformation

def getStandardMeanStd():
    return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

def loadPreproccessor():
    mean, std = getStandardMeanStd()

    # Prepare data transformation pipeline
    return transforms.Compose([
        transforms.Resize(MODEL_IMAGE_RESOLUTION),  # Resize to fit MobileNet input size
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize with ImageNet mean and std
    ])

def loadModel(trainedClasses: int):
    model = models.resnet50(pretrained=True)
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if trainedClasses != 0:
        model.fc = nn.Linear(model.fc.in_features, trainedClasses)
    weights = ModelInformationHelper.loadModelWeights()
    if weights:
        model.load_state_dict(weights)

    return model

def trainModelOnDataset(pathDataset: str):
    modelInformation = ModelInformationHelper.loadModelInformation()
    fisherInformation = ModelInformationHelper.loadFisherInformation()
    classCountKey = 'class_count'

    dataset = datasets.ImageFolder(root=pathDataset, transform=loadPreproccessor())

    if modelInformation is not None and classCountKey in modelInformation:
        trainedClasses = modelInformation[classCountKey]
    else:
        trainedClasses = 0

    model = loadModel(trainedClasses)
    model, fisherInformation = train_model(model, trainedClasses, dataset, fisherInformation)

    ModelInformationHelper.saveFisherInformation(fisherInformation)
    ModelInformationHelper.saveModelWeights(model.state_dict())
    modelInformation = {
        classCountKey: trainedClasses + len(dataset.classes),
    }
    ModelInformationHelper.saveModelInformation(modelInformation) 