import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import keras

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FINAL_OUTPUT_SIZE = 17


def import_operator_data(x_train, y_train, x_test, y_test):
    operator_x_train = np.load("../data/processed_data/X_train.npy")
    operator_x_test = np.load("../data/processed_data/X_test.npy")
    operator_y_train = np.load("../data/processed_data/Y_train.npy")
    operator_y_test = np.load("../data/processed_data/Y_test.npy")

    x_train = np.concatenate((x_train, operator_x_train))
    x_test = np.concatenate((x_test, operator_x_test))

    y_train = np.concatenate((y_train, operator_y_train))
    y_test = np.concatenate((y_test, operator_y_test))

    train_perm = np.random.permutation(len(x_train))
    test_perm = np.random.permutation(len(x_test))

    x_train = x_train[train_perm]
    x_test = x_test[test_perm]

    y_train = y_train[train_perm]
    y_test = y_test[test_perm]

    return x_train, y_train, x_test, y_test

def preprocess_data(batch_size: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train, y_train, x_test, y_test = import_operator_data(x_train, y_train, x_test, y_test)

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_train = x_train.astype("float32") / 255

    # y_train = keras.utils.to_categorical(y_train)
    # y_train = y_train.reshape(len(y_train), FINAL_OUTPUT_SIZE, 1)

    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    x_test = x_test.astype("float32") / 255

    x_train = torch.as_tensor(x_train)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    x_test = torch.as_tensor(x_test)
    y_test = torch.as_tensor(y_test, dtype=torch.long)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

class ConvNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNetwork, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense_layer_1 = nn.Linear(in_features=16 * 13 * 13, out_features=64)
        self.relu_2 = nn.ReLU()

        self.dense_layer_2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.max_pool_1(out)

        # Keeps the batch the dimension the same, then flattens the rest.
        out = out.reshape(out.size(0), -1)

        out = self.dense_layer_1(out)
        out = F.relu(out)

        out = self.dense_layer_2(out)

        return out
    
model = ConvNeuralNetwork(num_classes=FINAL_OUTPUT_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.005
)

BATCH_SIZE = 64
NUM_EPOCH = 5

train_loader, test_loader = preprocess_data(BATCH_SIZE)

for epoch in range(NUM_EPOCH):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, NUM_EPOCH, loss.item())) 

print("checkin accuracy")

with torch.no_grad():
    correct, total = 0, 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the train images: {} %'.format(100 * correct / total))