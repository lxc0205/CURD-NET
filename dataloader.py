import torch
from torchvision import datasets, transforms

# 生成随机的合成数据集
def generate_data(input_size, num_samples):
    X = torch.rand(num_samples, input_size)
    y = torch.rand(num_samples, 1)
    return X, y

# mnist dataset classification dataloader
def mnist_dataloader(batch_size):
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    # 获取数据
    train_data = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_data, test_data