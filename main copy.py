import time
import torch
import torch.nn as nn
import torch.optim as optim
from mlp import Base_Linear, Base_Nonlinear, Base_NonlinearB, MiniMLP, LargeMLP, MiniMLP_nonlinear, LargeMLP_nonlinear, MiniMLP_nonlinearB, LargeMLP_nonlinearB

# 合成数据集
def generate_data(input_size, num_samples):
    X = torch.rand(num_samples, input_size)  # 随机生成输入数据
    y = torch.rand(num_samples, 1)  # 随机生成目标数据
    return X, y


# 训练函数
def train_model(model, X, y, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if loss < 0.0001:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')
            break

        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')

# 测试函数
def test_model(model, X, y):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        predictions = model(X)
        loss = nn.MSELoss()(predictions, y)
        print(f'Test Loss: {loss.item():.5f}')

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 10
    hidden_size = 50
    output_size = 1
    num_samples = 1000
    epochs = 40000
    learning_rate = 0.001
    model_scale_type = 'base' # base, mini, or large


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

    # 训练
    start_time = time.time()
    print("Training MLP...")
    train_model(model, X, y, epochs, learning_rate)
    test_model(model, X, y)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Training MLP_nonlinear...")
    train_model(model_non, X, y, epochs, learning_rate)
    test_model(model_non, X, y)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Training MLP_nonlinearB...")
    train_model(model_nonB, X, y, epochs, learning_rate)
    test_model(model_nonB, X, y)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")
    
