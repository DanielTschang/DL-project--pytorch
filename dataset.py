from devices import *
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor, Lambda, Compose

def loaddataset():
    # Download test data from open datasets.
    #transform 幫你把dataset轉成Tensor
    trainset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    testset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # 將資料集切成batch(batch_size)，shuffle則決定是否打亂資料再切batch，pin_memory=True能加快資料從CPU轉到GPU的時間
    trainloader  = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, pin_memory=True)
    return trainloader, testloader