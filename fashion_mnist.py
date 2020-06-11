import torch
import pandas as pd
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

label_dict = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
              5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

data_path = r"C:\Users\91900\Documents\data\fashion_mnist"
train_csv_file = "fashion-mnist_train.csv"
test_csv_file = "fashion-mnist_test.csv"

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 5


def get_data(train=True):
    if train:
        csv_file_path = os.path.join(data_path, train_csv_file)
    else:
        csv_file_path = os.path.join(data_path, test_csv_file)

    df = pd.read_csv(csv_file_path)
    x = df.iloc[0:, 1:].values
    y = df.iloc[0:, 0].values
    x = x.reshape(x.shape[0], 1, 28, 28)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    ds = TensorDataset(x, y)
    if train:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    else:
        dl = DataLoader(ds, batch_size=batch_size)

    return dl


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':

    train_dl = get_data()
    test_dl = get_data(False)
    model = ConvNet(num_classes)
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_dl)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dl):
            images = images
            labels = labels

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dl:
            images = images
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))



