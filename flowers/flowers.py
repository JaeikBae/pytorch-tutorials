import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import os

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #conv net for flower 102
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, 1),
            torch.nn.ReLU()
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(12800, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 102)
        )


    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
    
BATCH_SIZE = 10
epochs = 3

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
        transforms.ToTensor()]
)

trainset = torchvision.datasets.Flowers102(root='./data', split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True)
testset = torchvision.datasets.Flowers102(root='./data', split='test',
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=True)
valset = torchvision.datasets.Flowers102(root='./data', split='val',
                                        download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                            shuffle=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_log = []
def train(model, loader, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        for idx, data in enumerate(loader):
            img = data[0].to(DEVICE)
            label = data[1].to(DEVICE)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if idx % 100 == 99:
                print(f'[{epoch+1} / {idx+1}, {loss.item()}]')
        train_log.append(loss.item())
    print(f'Finished Training {loss.item()}')


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(loader):
            img = data[0].to(DEVICE)
            label = data[1].to(DEVICE)
            out = model(img)
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(f'Accuracy(total : {total}) : {100*correct/total}')


if __name__ == "__main__":
    model = Net().to(DEVICE)
    summary(model, (3, 32, 32))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9)    
    train(model, testloader, optimizer, criterion)
    test(model, trainloader)
    test(model, valloader)
    plt.plot(train_log)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.show()
    torch.save(model.state_dict(), './flowers.pth')