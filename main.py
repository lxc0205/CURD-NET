import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mlp import Base_Linear, Base_Nonlinear, Base_NonlinearB, MiniMLP, LargeMLP, MiniMLP_nonlinear, LargeMLP_nonlinear, MiniMLP_nonlinearB, LargeMLP_nonlinearB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# conditional_uncorrelation
def loss_function(outputs, y):
    pearson_matrix1 = torch.corrcoef(torch.cat((outputs, y), dim=1))
    pearson_matrix2 = torch.corrcoef(outputs)
    loss = pearson_matrix1 / pearson_matrix2
    return loss

# 训练函数
def train_model(model, train_loader, epochs, optimizer, tau = 0.5):
    model = model.to(device)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            image = image.view(-1, 1, image.shape[2]*image.shape[3])
            label = label.float()
            optimizer.zero_grad()
            outputs = model(image)
            outputs = outputs.squeeze(-1).squeeze(-1)
            loss = criterion(outputs, label)
            # loss = tau * criterion(outputs, label) + (1-tau) * loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            break

        if loss < 0.0001:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')
            break
        # if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')

# 测试函数
def test_model(model, test_loader, tau = 0.5):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            image = image.view(-1, 1, image.shape[2]*image.shape[3])
            optimizer.zero_grad()
            predictions = model(image)
            loss = criterion(predictions, label)
            # loss = tau * criterion(outputs, label) + (1-tau) * loss_function(outputs, label)
        print(f'Test Loss: {loss.item():.5f}')

# 主程序
if __name__ == "__main__":
    input_size, hidden_size, output_size = 784, 88, 1
    epochs = 1000
    lr = 0.001
    batch_size = 64

    model = MiniMLP(input_size, hidden_size, output_size)
    model_non = MiniMLP_nonlinear(input_size, hidden_size, output_size)

    train_data = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    # start_time = time.time()
    print("Training MLP...")
    train_model(model, train_data_loader, epochs, optimizer)
    test_model(model, test_data_loader)
    # end_time = time.time()
    # print(f"Training took {end_time - start_time:.2f} seconds.")

    # start_time = time.time()
    print("Training MLP_nonlinear...")
    train_model(model_non, train_data_loader, epochs, optimizer)
    test_model(model_non, test_data_loader)
    # end_time = time.time()
    # print(f"Training took {end_time - start_time:.2f} seconds.")

    
