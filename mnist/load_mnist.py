import torch
import torchvision
import torchvision.transforms as transforms
from mnist import Net, eval

if __name__ == '__main__':
    BATCH_SIZE = 4

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                shuffle=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(DEVICE)
    weight = torch.load('mnist.pth')
    model.load_state_dict(weight)
    eval(model, testloader)