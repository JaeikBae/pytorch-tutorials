import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
from Flowers import Net, testloader

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    BATCH_SIZE = 10
    net = Net()
    net.load_state_dict(torch.load('./flower.pth'))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(BATCH_SIZE)))
    #imshow(torchvision.utils.make_grid(images))
    print(len(testloader))
    # correct = 0
    # total = 0

    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         total += BATCH_SIZE
    #         correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the {total} test images: {100*correct/total}%')
    class_correct = list(0. for i in range(8189))
    class_total = list(0. for i in range(8189))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    import json
    with open('./flower_to_name.json', 'r') as f:
        flower_to_name = json.load(f)
    for i in range(8189):
        if  class_total[i] != 0 and 100*class_correct[i]/class_total[i] != 0.0:
            print(f'Acc of {str(i+1)}: {100*class_correct[i]/class_total[i]}%')
