import torch

tensor = torch.ones(4, 4)

print('\nGPU available: ', torch.cuda.is_available())
print('GPU name: ', torch.cuda.get_device_name(0))
print('GPU count: ', torch.cuda.device_count())
print('GPU current device: ', torch.cuda.current_device())
print('GPU device: ', torch.cuda.device(0), '\n')

if torch.cuda.is_available():
    tensor = tensor.to('cuda')

print(f'tensor runs on : ', tensor.device, '\n')
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)