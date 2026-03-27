import torch

# net
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,100)
        self.predict = torch.nn.Linear(100,n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out



import numpy as np
import re

ff = open("src/data/housing.data", "r").readlines()
data = []
for item in ff:
    out = re.sub(r"\s{2,}"," ",item).strip()
    # print(out)
    data.append(out.split(" "))

data = np.array(data).astype(float)
print(data.shape)

y = data[:,-1]
X = data[:,0:-1]
print(y.shape)
print(X.shape)

y_train = y[0:496,...]
y_test = y[496:,...]
X_train = X[0:496, ...]
X_test = X[496:, ...]

print(y_train.shape)
print(y_test.shape)
print(X_train.shape)
print(X_test.shape)


net = torch.load("src/models/model.pkl", weights_only=False)

loss_func = torch.nn.MSELoss()

x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

pred = net.forward(x_test)
# print(pred.shape)
pred = pred.squeeze()
loss_test = loss_func(pred, y_test) * 0.001

print(f"loss_test:{loss_test}")
