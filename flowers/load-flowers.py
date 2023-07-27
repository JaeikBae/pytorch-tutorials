import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch import Tensor
from flowers import Net, test, transform
import matplotlib.pyplot as plt

#num_cols= choose the grid size you want
def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

if __name__ == '__main__':
    BATCH_SIZE = 4

    testset = torchvision.datasets.Flowers102(root='./data', split='test',
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=True, drop_last=True)

    valset = torchvision.datasets.Flowers102(root='./data', split='val',
                                                download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                            shuffle=True, drop_last=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(DEVICE)
    weight = torch.load('flower.pth')
    model.load_state_dict(weight)
    tensor = Tensor.cpu(model.net[0].weight.data).numpy()
    #(1/(2*(maximum negative value)))*filter+0.5 === you need to normalize the filter before plotting.
    tensor = (1/(2*(tensor.min())))*tensor+0.5
    plot_kernels(tensor)
    # print('Test')
    # test(model, testloader)
    # print('Validation')
    # test(model, valloader)