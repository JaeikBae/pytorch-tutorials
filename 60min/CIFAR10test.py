import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
from CIFAR10 import Net, testloader

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show(block=False)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    start_time = time.time()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    BATCH_SIZE = 4
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()
    net.load_state_dict(torch.load('./cifar_net.pth'))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += BATCH_SIZE
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100*correct/total}%')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(BATCH_SIZE):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s} : {100*class_correct[i]/class_total[i]}%')
    
    print(f'Finished in {time.time() - start_time} seconds')
