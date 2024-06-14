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
        target_length = 50
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
        target_length = 50
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



import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv1d(1, 1, kernel_size=6, stride=2, padding=0)
        self.cv2 = nn.Conv1d(1, 1, kernel_size=6, stride=2, padding=0)
        self.cv3 = nn.Conv1d(1, 1, kernel_size=6, stride=2, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.W = nn.Linear(dim, dim)

    def forward(self, x):
        # 计算注意力权重
        attn_weights = torch.softmax(self.W(x), dim=-1)
        output = torch.sum(attn_weights * x, dim=-1)
        return output


class Decoder(nn.Module):
    def __init__(self):  # Decoder#输入（1，1，3）,要求输出是（1，1，200）
        super(Decoder, self).__init__()
        self.cv1 = nn.ConvTranspose1d(1, 1, kernel_size=5, stride=1, padding=1)
        self.l1 = nn.Linear(5, 20)
        self.attn1 = Attention(20)
        self.cv2 = nn.ConvTranspose1d(1, 1, kernel_size=10, stride=1, padding=1)
        self.l2 = nn.Linear(27, 80)
        self.attn2 = Attention(80)
        self.cv3 = nn.ConvTranspose1d(1, 1, kernel_size=20, stride=1, padding=1)
        self.l3 = nn.Linear(97, 200)
        self.attn3 = Attention(200)

    def forward(self, x):
        x = self.cv1(x)
        x = self.l1(x)
        x = self.attn1(x) + x
        x = self.cv2(x)
        x = self.l2(x)
        x = self.attn2(x) + x
        x = self.cv3(x)
        x = self.l3(x)
        x = self.attn3(x) + x
        return x
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.linear = nn.Linear(21, 3)
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        hidden_3 = self.linear(encoded)
        decoded = self.decoder(hidden_3)
        return decoded

    def Get_hidden_3(self, x):
        encoded = self.encoder(x)
        hidden_3 = self.linear(encoded)
        hidden_3 = hidden_3.squeeze(0)
        hidden_3 = hidden_3.squeeze(0)
        return hidden_3

    def Hidden_3_Decoder(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        decoded = self.decoder(x)
        return decoded


model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 开始训练
Data_Tem = []
for i in range(len(Data)):
    Data_Tem.append(Data[i].temlist)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
Data_Tem = scaler.fit_transform(Data_Tem)
Data_Tem = torch.FloatTensor(Data_Tem)

num_epochs = 200

for epoch in range(num_epochs):
    for tem in Data_Tem:
        tem = tem.unsqueeze(0)
        tem = tem.unsqueeze(0)
        outputs = model(tem)
        loss = criterion(outputs, tem)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 保存模型

torch.save(model.state_dict(), 'conv_autoencoder.pth')

# 保存hidden_3

hidden_3 = np.array([])
all=[]
for i in range(len(Data)):
    tem=Data_Tem[i]
    tem=torch.FloatTensor(tem)
    tem = tem.unsqueeze(0)
    tem = tem.unsqueeze(0)
    hidden_torch=model.Get_hidden_3(tem)
    hidden_torch=hidden_torch.detach().numpy()
    data_to_save=[Data[i].P,Data[i].I,Data[i].U,hidden_torch[0],hidden_torch[1],hidden_torch[2],Data[i].T_1500]
    all.append(data_to_save)

name=['P','I','U','hidden_1','hidden_2','hidden_3','T_1500']
all=pd.DataFrame(columns=name,data=all)
all.to_csv('hidden_3.csv',index=False)
print("hidden_3保存完成")

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model.load_state_dict(torch.load('conv_autoencoder.pth'))
# Load the CSV file
data = pd.read_csv('hidden_3.csv')

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
