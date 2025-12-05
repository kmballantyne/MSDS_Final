import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from config import *

def _build_optimizer(model):
    if OPTIMIZER.lower() == "adam":
        return optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        return optim.SGD(model.parameters(), lr=LEARNING_RATE)

def local_train(model, train_dataset):
    
    model.to(DEVICE)  # Move model to the appropriate device (GPU or CPU)
    model.train()
    
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = _build_optimizer(model)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(LOCAL_EPOCHS):
        for images, labels in loader: # Assuming images and labels are the data and targets (X, y)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()  # Return the model's state dict after training

