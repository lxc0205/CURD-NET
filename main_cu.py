import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from mlp import Base_Linear, Base_Nonlinear, Base_NonlinearB, MiniMLP, LargeMLP, MiniMLP_nonlinear, LargeMLP_nonlinear, MiniMLP_nonlinearB, LargeMLP_nonlinearB

# 生成合成数据集
def generate_data(input_size, num_samples):
    X = torch.rand(num_samples, input_size)  # 随机生成输入数据
    y = torch.rand(num_samples, 1)  # 随机生成目标数据
    return X, y

# mnist dataset classification dataloader
def mnist_dataloader():
    train_data = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    return train_data, test_data

def loss_function(outputs, y, tau = 0.5):
    # conditional_uncorrelation
    assert outputs.shape == y.shape
    assert outputs.dim() == 2

    pearson_matrix1 = torch.corrcoef(torch.cat((outputs, y), dim=1))
    pearson_matrix2 = torch.corrcoef(outputs)

    conditional_correlation = pearson_matrix1 / pearson_matrix2
    
    loss = nn.MSELoss()(outputs, y) * (1 - tau) + conditional_correlation * tau
    return loss

# 训练函数
def train_model(model, X, y, epochs, learning_rate, tau=0.5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, y, tau)
        loss.backward()
        optimizer.step()

        if loss < 0.0001:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')
            break

        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')

# 测试函数
def test_model(model, X, y, tau=0.5):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        predictions = model(X)
        loss = loss_function(predictions, y, tau)
        print(f'Test Loss: {loss.item():.5f}')

# 主程序
def main(arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 10
    hidden_size = 50
    output_size = 1
    num_samples = 1000
    epochs = 40000
    learning_rate = 0.001
    model_scale_type = 'mini' # base, mini, or large
    tau = 0.5

    if model_scale_type == 'base':
        model = Base_Linear(input_size, output_size)
        model_non = Base_Nonlinear(input_size, output_size)
        model_nonB = Base_NonlinearB(input_size, output_size)
    if model_scale_type == 'mini':
        model = MiniMLP(input_size, hidden_size, output_size)
        model_non = MiniMLP_nonlinear(input_size, hidden_size, output_size)
        model_nonB = MiniMLP_nonlinearB(input_size, hidden_size, output_size)
    if model_scale_type == 'large':
        model = LargeMLP(input_size, hidden_size, output_size)
        model_non = LargeMLP_nonlinear(input_size, hidden_size, output_size)
        model_nonB = LargeMLP_nonlinearB(input_size, hidden_size, output_size)

    # 生成数据
    X, y = generate_data(input_size, num_samples)
    train_data, test_data = mnist_dataloader()


    # 训练
    start_time = time.time()
    print("Training MLP...")
    train_model(model, X, y, epochs, learning_rate, tau)
    test_model(model, X, y, tau)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Training MLP_nonlinear...")
    train_model(model_non, X, y, epochs, learning_rate, tau)
    test_model(model_non, X, y , tau)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Training MLP_nonlinearB...")
    train_model(model_nonB, X, y, epochs, learning_rate, tau)
    test_model(model_nonB, X, y, tau)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_scale_type', dest='model_scale_type', type=str, required=True, default='mini', help='Support methods: base|mini|large')
    parser.add_argument('--model_type', dest='model_type', type=str, required=True, default='linear', help='Support methods: linear|nonlinear|nonlinearB')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, required=True, default=50, help='Hidden layer size.')
    parser.add_argument('--output_size', dest='output_size', type=int, required=True, default=1, help='Output layer size.')
    parser.add_argument('--input_size', dest='input_size', type=int, required=True, default=10, help='Input layer size.')
    args = parser.parse_args()
    main(args)