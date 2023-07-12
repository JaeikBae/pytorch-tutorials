import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f'Tensor from data: \n {x_data} \n')

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'Numpy Array: \n {np_array} \n')

x_ones = torch.ones_like(x_data)    
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random Tensor: \n {rand_tensor} \n')
print(f'Ones Tensor: \n {ones_tensor} \n')
print(f'Zeros Tensor: \n {zeros_tensor} \n')

tensor = torch.rand(3,4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')

print('\nGPU available: ', torch.cuda.is_available())
print('GPU name: ', torch.cuda.get_device_name(0))
print('GPU count: ', torch.cuda.device_count())
print('GPU current device: ', torch.cuda.current_device())
print('GPU device: ', torch.cuda.device(0), '\n')

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

print(f'tensor runs on : ', tensor.device, '\n')
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)