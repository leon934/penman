import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import preprocess_data, save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FINAL_OUTPUT_SIZE = 17

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
    
def main():
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
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, NUM_EPOCH, loss.item())) 

    print("Now checking accuracy of model with both the testing data and the training data.")

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

    save_model(model=model, path="./model/model.pt")

if __name__ == "__main__":
    main()