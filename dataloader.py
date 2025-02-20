import torch
from torchvision import datasets, transforms

# 生成随机的合成数据集
def generate_data(input_size, num_samples):
    X = torch.rand(num_samples, input_size)
    y = torch.rand(num_samples, 1)
    return X, y

def mnist_dataloader(train_batch_size, test_batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader