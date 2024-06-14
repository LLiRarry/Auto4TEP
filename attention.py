import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import linear_model


def Delete_(List) -> list:
    n = len(List)
    # New_list=[]
    i = 0
    while i < len(List):
        if len(str((List[i])[0])) == 1:
            List = List[0: i - 1]
            return List
        if i == n - 2:
            return List
        i = i + 1


class Input:
    P = 0
    I = 0
    U = 0
    T_1500 = 0
    timelist = []
    temlist = []


def Read(P, I, path, Lay):  # 第一次参数提取
    listu = [50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 250]
    i = 0
    j = 0
    while i < 26:
        path_data_p_list = pd.read_csv(path, usecols=[i + 1], skiprows=0)
        path_data_t_list = pd.read_csv(path, usecols=[i], skiprows=0)
        listW = path_data_p_list.values.tolist()
        listT = path_data_t_list.values.tolist()
        listTT = Delete_(listT)
        listWW = Delete_(listW)
        i = i + 2
        layyer = Input()
        layyer.U = listu[j]
        j = j + 1
        layyer.I = I
        layyer.P = P
        T_copy = [float(i) for sublist in listTT for i in sublist]
        original_list = [float(i) for sublist in listTT for i in sublist]
        target_length = 200
        x = np.linspace(0, len(original_list) - 1, target_length)
        new_list1 = np.interp(x, np.arange(len(original_list)), original_list)
        layyer.timelist = new_list1
        # 只选取小于1500的数据
        Index = 0
        W_copy = [float(i) for sublist in listWW for i in sublist]
        for num in range(len(W_copy)):
            if W_copy[num] > float(1500):
                Index = num
        original_list = [float(i) for sublist in listWW for i in sublist if float(i) < 1500]
        target_length = 200
        x = np.linspace(0, len(original_list) - 1, target_length)
        new_list = np.interp(x, np.arange(len(original_list)), original_list)
        layyer.temlist = new_list
        layyer.T_1500 = T_copy[Index]
        Lay.append(layyer)


def Return_All():
    print("数据处理中..........")
    Lay1 = []
    Read(2, 1, '1.csv', Lay1)
    Read(2, 1, '2.csv', Lay1)
    Read(2, 2, '2.csv', Lay1)
    Read(2, 3, '3.csv', Lay1)
    Read(2, 4, '4.csv', Lay1)
    Read(2, 5, '5.csv', Lay1)
    Lay2 = []
    Read(1, 1, '6.csv', Lay2)
    Read(1, 2, '7.csv', Lay2)
    Read(1, 3, '8.csv', Lay2)
    Read(1, 4, '9.csv', Lay2)
    Read(1, 5, '10.csv', Lay2)
    Lay3 = []
    Read(3, 1, '11.csv', Lay3)
    Read(3, 2, '12.csv', Lay3)
    Read(3, 3, '13.csv', Lay3)
    Read(3, 4, '14.csv', Lay3)
    Read(3, 5, '15.csv', Lay3)
    Lay4 = []
    Read(4, 1, '16.csv', Lay4)
    Read(4, 2, '17.csv', Lay4)
    Read(4, 3, '18.csv', Lay4)
    Read(4, 4, '19.csv', Lay4)
    Read(4, 5, '20.csv', Lay4)
    Lay5 = []
    Read(5, 1, '21.csv', Lay5)
    Read(5, 2, '22.csv', Lay5)
    Read(5, 3, '23.csv', Lay5)
    Read(5, 4, '24.csv', Lay5)
    Read(5, 5, '25.csv', Lay5)
    ALL_lay = Lay1 + Lay2 + Lay3 + Lay4 + Lay5
    return ALL_lay


Data = Return_All()
print("数据处理完成")
print(len(Data), "组数据")

# 数据筛选
temlist = []
timelist = []
indextodelete = []
for i in range(len(Data)):
    timelist = Data[i].timelist
    if max(timelist) > 1e-7:
        indextodelete.append(i)
    temlist = Data[i].temlist
    if max(temlist) < 1200:
        indextodelete.append(i)

print(indextodelete,"组数据被删除")
Data0 = [Data[i] for i in range(len(Data)) if i not in indextodelete]
Data=Data0




class SelfAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.query = nn.Linear(input_size, output_size)
        self.key = nn.Linear(input_size, output_size)
        self.value = nn.Linear(input_size, output_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        compressed = torch.matmul(attention_weights, value)
        return compressed
# 开始训练
Data_Tem = []
for i in range(len(Data)):
    Data_Tem.append(Data[i].temlist)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
Data_Tem = scaler.fit_transform(Data_Tem)
Data_Tem = torch.FloatTensor(Data_Tem)

num_epochs = 200

class Atention_AE(nn.Module):
    def __init__(self):
        super(Atention_AE, self).__init__()
        self.compression_layer1 = SelfAttention(200, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.compression_layer2 = SelfAttention(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.compression_layer3 = SelfAttention(64, 3)
        self.bn3 = nn.BatchNorm1d(3)
        self.reconstruction_layer1 = SelfAttention(3, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.reconstruction_layer2 = SelfAttention(64, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.reconstruction_layer3 = SelfAttention(128, 200)
        self.bn6 = nn.BatchNorm1d(200)
    def forward(self, x):
        compressed = self.compression_layer1(x)
        compressed = self.bn1(compressed)
        compressed = self.compression_layer2(compressed)
        compressed = self.bn2(compressed)
        compressed = self.compression_layer3(compressed)
        compressed = self.bn3(compressed)
        reconstructed = self.reconstruction_layer1(compressed)
        reconstructed = self.bn4(reconstructed)
        reconstructed = self.reconstruction_layer2(reconstructed)
        reconstructed = self.bn5(reconstructed)
        reconstructed = self.reconstruction_layer3(reconstructed)
        reconstructed = self.bn6(reconstructed)
        return reconstructed

    def Get_hidden_3(self,x):
        compressed = self.compression_layer1(x)
        compressed = self.compression_layer2(compressed)
        compressed = self.compression_layer3(compressed)
        return compressed
model = Atention_AE()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

from torch.utils.data import Dataset, DataLoader

train_loader = DataLoader(dataset=Data_Tem, batch_size=8, shuffle=True)
for epoch in range(num_epochs):
    for data in train_loader:
        output = model(data)
        loss = criterion(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

torch.save(model.state_dict(), 'sim_autoencoder.pth')



# 保存hidden_3

hidden_3 = np.array([])
all=[]
for i in range(len(Data)):
    tem=Data_Tem[i]
    tem=torch.FloatTensor(tem)
    tem = tem.reshape(1, 200)
    hidden_torch=model.Get_hidden_3(tem)
    hidden_torch=hidden_torch.detach().numpy()
    data_to_save=[Data[i].P,Data[i].I,Data[i].U,hidden_torch[0],hidden_torch[1],hidden_torch[2],Data[i].T_1500]
    all.append(data_to_save)

name=['P','I','U','hidden_1','hidden_2','hidden_3','T_1500']
all=pd.DataFrame(columns=name,data=all)
all.to_csv('sim_hidden_3.csv',index=False)
print("hidden_3保存完成")

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Load the CSV file
data = pd.read_csv('sim_hidden_3.csv')

# Separate the features (X) and targets (y)
X = data.iloc[:, :3]  # First three columns as features
y = data.iloc[:, 3:]  # Last three columns as targets

# Create polynomial features
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

# Perform polynomial regression
regressor = LinearRegression()
regressor.fit(X_poly, y)
# 20x8

plt.figure(figsize=(20, 6))
for i in range(30):
    # Example input data
    input_data = [Data[i].P, Data[i].I, Data[i].U]
    X_input = np.array(input_data).reshape(1, -1)
    X_input_poly = poly.transform(X_input)
    y_pred = regressor.predict(X_input_poly)
    T_1500 = y_pred[0][3]
    time_line = np.linspace(0, T_1500, num=200)
    hidden_3 = np.array([y_pred[0][0], y_pred[0][1], y_pred[0][2]])
    hidden_3 = torch.FloatTensor(hidden_3)
    Temlist = model.Hidden_3_Decoder(hidden_3)
    Temlist = Temlist.squeeze(0)
    Temlist = Temlist.detach().numpy()
    Temlist = scaler.inverse_transform(Temlist)
    Temlist = Temlist.reshape(-1, 1)
    Temlist = Temlist.tolist()
    plt.subplot(121)
    plt.title('Predicted Temperature')
    plt.plot(Data[i].timelist, Temlist)
    plt.subplot(122)
    plt.title('Real Temperature')
    plt.plot(Data[i].timelist, Data[i].temlist)

plt.show()




