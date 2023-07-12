import torch
import numpy as np

data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

#slicing and indexing
print('\nslicing and indexing')
print(f'tensor runs on : ', tensor.device, '\n')
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])

#joining tensors
print('\njoining tensors')
t1 = torch.cat([tensor, tensor, tensor], dim=0)
t2 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t2)

#arithmetic operations
tensor = torch.tensor([[1, 2],
                       [3, 4]], dtype=torch.float32)

print("\narithmetic operations")
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
print(y2)
print(y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z2)
print(z3)

#single-element tensors
print("\nsingle-element tensors")
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#in-place operations
print("\nin-place operations")
print(tensor)
tensor.add_(5)
print(tensor)

#bridge with numpy
print("\nbridge with numpy")
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')
t.add_(1) #in-place operation reflected in numpy array as well
print(f't: {t}')
print(f'n: {n}')
print('\n')
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n) #in-place operation reflected in torch tensor as well
print(f't: {t}')
print(f'n: {n}')