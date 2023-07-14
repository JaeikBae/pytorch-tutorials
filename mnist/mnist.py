import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

BATCH_SIZE = 4

transform = transforms.Compose(
    [transforms.ToTensor()]
)

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, trainloader, optimizer, criterion):
    model.train()
    for idx, data in enumerate(trainloader):
        img = data[0].to(DEVICE)
        label = data[1].to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        if idx % 2000 == 1999:
            print(f'[{idx+1}, {loss.item()}]')
    print('Finished Training')



def eval():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100*correct/total}%')

if __name__ == '__main__':
    model = Net().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    print(summary(model, (1, 28, 28)))
    train(model, trainloader, optimizer, criterion)
    eval()
    
