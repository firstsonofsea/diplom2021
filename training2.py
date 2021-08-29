import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os


class ConvNet_ident(nn.Module):
    def __init__(self):
        super(ConvNet_ident, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(86528, 2048)
        self.fc2 = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return x


def get_dataset(data_dir, data_transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=98,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    classes = image_datasets['train'].classes

    return dataloaders["train"], dataloaders['test'], classes, dataset_sizes


if __name__ == '__main__':
    model = ConvNet_ident()

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }


    data_dir = "data_ident"
    trainloader, testloader, classes, dataset_sizes = get_dataset(data_dir, data_transforms)
    print(classes)
    print(len(classes), dataset_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0], data[1]
            #img = transforms.ToPILImage()(inputs[0])
            #img.show()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / len(trainloader)))


    print('Finished Training')
    dir = os.getcwd()
    PATH = os.path.join(dir, "my_model4.pt")
    torch.save(model, PATH)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the', dataset_sizes['test'], 'test images: %d %%' % (
            100 * correct / total))
    print(outputs)
