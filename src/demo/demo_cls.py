import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# data
train_data = datasets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
#
test_data = datasets.MNIST(root='./data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=False)

# batchsize
train_loader = data_utils.DataLoader(train_data,
                                     batch_size=64,
                                     shuffle=True)
test_loader = data_utils.DataLoader(test_data,
                                    batch_size=64,
                                    shuffle=True)

# net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.fc = torch.nn.Linear(14*14*32,10)

    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

cnn = CNN()
cnn = cnn.to(device)

# loss
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# training

for epoch in range(10):
    for step, (images, labels) in enumerate(train_loader):
        image = images.to(device)
        label = labels.to(device)

        output = cnn(image)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Step {step}/{len(train_data)//64}, Loss {loss.item()}")


# eval
    loss_test = 0
    accuracy = 0
    for step, (images, labels) in enumerate(test_loader):
        image = images.to(device)
        label = labels.to(device)

        output = cnn(image)
        loss_test += loss_func(output, label)
        _, pred = torch.max(output, 1)
        accuracy += torch.sum(pred == label).item()

    accuracy /= len(test_loader)
    loss_test /= (len(test_loader) // 64)

    print(f"Epoch: {epoch+1}, Accuracy: {accuracy}, Test Loss: {loss_test}")


# save

model = torch.save(cnn, "src/models/mnist_model.pkl")

# load



# inference