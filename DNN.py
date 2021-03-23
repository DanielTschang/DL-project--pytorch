from devices import *
import torch
import torch.nn.functional as F
from torch import nn

class Net(nn.Module):
    def __init__(self):  # __init__()定義Neural Network架構
        super().__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 28, 64).cuda()  # the image size is 28 * 28. 所以input維784, output可自定此處為64
        self.fc2 = nn.Linear(64, 64).cuda()  # 上層的output為64，故此處input也要為64，output同樣可自訂
        self.fc3 = nn.Linear(64, 64).cuda()
        self.fc4 = nn.Linear(64, 10).cuda()  # 輸出層output必須為10，因為MNIST資料集為0~9共10個數字的便是

    def forward(self, x):  # forward()定義資料x如何一層一層傳遞，最後回傳
        x = F.relu(self.fc1(x))  # 經過fully connected layer後通過relu activiation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # 輸出經過softmax後取log, dim=1代表依照每個row所有element來做softmax(output每個row為1筆training data的10個class機率值輸出)
        # 例如batch size = 100，則x的size為 100 * 10(100 row, 10 column)
        return F.log_softmax(x, dim=1)