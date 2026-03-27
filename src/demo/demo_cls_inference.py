import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.demo.CNN import CNN
import cv2
# device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


test_data = datasets.MNIST(root='./data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=False)


test_loader = data_utils.DataLoader(test_data,
                                    batch_size=64,
                                    shuffle=True)

cnn = torch.load('src/models/mnist_model.pkl',weights_only=False)
cnn = cnn.to(device)
cnn = cnn.to(device)

# loss
loss_func = torch.nn.CrossEntropyLoss()

loss_test = 0
accuracy = 0
for step, (images, labels) in enumerate(test_loader):
    image = images.to(device)
    label = labels.to(device)

    output = cnn(image)
    loss_test += loss_func(output, label)
    _, pred = torch.max(output, 1)
    accuracy += torch.sum(pred == label).item()

    image = images.cpu().numpy()
    label = labels.cpu().numpy()

    # batchsize * 1 * 28 * 28
    for idx in range(image.shape[0]):
        im_data = image[idx]
        im_label = label[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2,0)

        print(f"label: {im_label}")
        print(f"pred: {im_pred}")

        cv2.startWindowThread()
        cv2.imshow("imdata",im_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


accuracy /= len(test_loader)
loss_test /= (len(test_loader) // 64)

print(f"Accuracy: {accuracy}, Test Loss: {loss_test}")
