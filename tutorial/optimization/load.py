import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.FMnistNetwork import NeuralNetwork
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("model.pth", map_location=device)
print(model)

with torch.no_grad():
    model.eval()
    inputs = torch.randn(1, 1, 28, 28)
    inputs = inputs.to(device)
    outputs = model(inputs)
    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)