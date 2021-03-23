import numpy as np
from CNN import *
from DNN import *
from devices import *
from dataset import loaddataset
import torch.optim as optim



def whichModel(model="DNN"):
    if model == "DNN":
        model = Net().to(device)  # inital network
        return model
    elif model == "CNN":
        model = CNN().to(device)  # inital network
        return model
    else:
        print("Error, the selected module doesn't exist!")

def train(model,epoch=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)  # create a Adam optimizer

    model.train()
    epochs = epoch
    for ep in range(epochs):
        running_loss = 0.0
        for i, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)
            # training process
            optimizer.zero_grad()  # clear the gradient calculated previously
            predicted = model(X)  # put the mini-batch training data to Nerual Network, and get the predicted labels
            loss = F.nll_loss(predicted, y)  # compare the predicted labels with ground-truth labels
            loss.backward()  # compute the gradient
            optimizer.step()  # optimize the network
            running_loss += loss.item()
            if i % 100 == 99:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (ep + 1, i + 1, running_loss / 100))
                running_loss = 0.0

def eval(model):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            correct += (torch.argmax(output, dim=1) == y).sum().item()
            total += y.size(0)

    print(f'Training data Accuracy: {correct}/{total} = {round(correct / total, 3)}')

    # Evaluation the testing data
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            correct += (torch.argmax(output, dim=1) == y).sum().item()
            total += y.size(0)

    print(f'testing data Accuracy: {correct}/{total} = {round(correct / total, 3)}')

if __name__ == "__main__":
    trainloader, testloader = loaddataset()
    print("device : ", device)
    print("每個batch之長度 : ",len(trainloader))

    model = whichModel("CNN")
    train(model,epoch=3,lr=0.001)
    eval(model)

