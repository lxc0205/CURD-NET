import time
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from dataloader import generate_data, mnist_dataloader
from mlp import Base_Linear, Mini_MLP, Large_MLP
from mnlp import Base_Nonlinear, Base_Nonlinear_Broad, Mini_MNLP, Large_MNLP, Mini_MNLP_Broad, Large_MNLP_Broad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def loss_function(outputs, y, tau = 0.5):
    # MSE * (1 - tau) + conditional_uncorrelation * tau
    assert outputs.shape == y.shape and outputs.dim() == 2
    # conditional_uncorrelation = torch.corrcoef(torch.cat((outputs, y), dim=1)) / torch.corrcoef(outputs)
    # loss = nn.MSELoss()(outputs, y) * (1 - tau) + conditional_uncorrelation * tau
    
    loss = nn.MSELoss()(outputs, y)
    return loss

# 训练函数
def train_model(model, dataloader, epochs, learning_rate, tau=0.5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X, y in dataloader:
            X = X.view(X.size(0), -1)
            y = y.view(y.size(0), -1)
            X = X.float()
            y = y.float()

            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = loss_function(outputs, y, tau)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        
        # 提前停止条件
        if avg_loss < 0.0001:
            print(f'Training stopped early as loss reached {avg_loss:.5f}')
            break

# 测试函数
def test_model(model, dataloader, tau=0.5):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for X, y in  tqdm(dataloader):
            X = X.view(X.size(0), -1)
            y = y.view(y.size(0), -1)
            X = X.float()
            y = y.float()

            X = X.to(device)
            y = y.to(device)
            predictions = model(X)  # 获取模型预测
            batch_loss = loss_function(predictions, y, tau)  # 计算当前 batch 的损失
            total_loss += batch_loss.item()  # 累加当前 batch 的损失
    # 计算平均损失
    avg_loss = total_loss / num_batches
    # 打印整个测试集的平均损失
    print(f'Test Loss: {avg_loss:.5f}')

# 训练函数
def train_model1(model, X, y, epochs, learning_rate, tau=0.5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        outputs = model(X)
        loss = loss_function(outputs, y, tau)
        loss.backward()
        optimizer.step()

        if loss < 0.0001:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')
            break

# 测试函数
def test_model1(model, X, y, tau=0.5):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        predictions = model(X)
        loss = loss_function(predictions, y, tau)
        print(f'Test Loss: {loss.item():.5f}')

# 主程序
def main1(arg):
    input_size = 784
    hidden_size = 196
    output_size = 1
    epochs = 1000
    learning_rate = 0.01
    tau = 0
    model_scale_type = 'base' # base, mini, or large 
    
    if model_scale_type == 'base':
        model = Base_Linear(input_size, output_size)
        model_non = Base_Nonlinear(input_size, output_size)
        model_nonB = Base_Nonlinear_Broad(input_size, output_size)
    if model_scale_type == 'mini':
        model = Mini_MLP(input_size, hidden_size, output_size)
        model_non = Mini_MNLP(input_size, hidden_size, output_size)
        model_nonB = Mini_MNLP_Broad(input_size, hidden_size, output_size)
    if model_scale_type == 'large':
        model = Large_MLP(input_size, hidden_size, output_size)
        model_non = Large_MNLP(input_size, hidden_size, output_size)
        model_nonB = Large_MNLP_Broad(input_size, hidden_size, output_size)

    model.to(device)
    model_non.to(device)
    model_nonB.to(device)

    # 生成数据 mnist_dataloader()
    train_dataloader, test_dataloader = mnist_dataloader(batch_size=64)

    # 训练
    start_time = time.time()
    print("Training MLP...")
    train_model(model, train_dataloader, epochs, learning_rate, tau)
    test_model(model, test_dataloader, tau)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Training MLP_nonlinear...")
    train_model(model_non, train_dataloader, epochs, learning_rate, tau)
    test_model(model_non, test_dataloader, tau)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Training MLP_nonlinearB...")
    train_model(model_nonB, train_dataloader, epochs, learning_rate, tau)
    test_model(model_nonB, test_dataloader, tau)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

def main(arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 10
    hidden_size = 50
    output_size = 1
    num_samples = 1000
    epochs = 40000
    learning_rate = 0.001
    model_scale_type = 'base' # base, mini, or large
    tau = 0

    if model_scale_type == 'base':
        model = Base_Linear(input_size, output_size)
        model_non = Base_Nonlinear(input_size, output_size)
        model_nonB = Base_Nonlinear_Broad(input_size, output_size)
    if model_scale_type == 'mini':
        model = Mini_MLP(input_size, hidden_size, output_size)
        model_non = Mini_MNLP(input_size, hidden_size, output_size)
        model_nonB = Mini_MNLP_Broad(input_size, hidden_size, output_size)
    if model_scale_type == 'large':
        model = Large_MLP(input_size, hidden_size, output_size)
        model_non = Large_MNLP(input_size, hidden_size, output_size)
        model_nonB = Large_MNLP_Broad(input_size, hidden_size, output_size)

    # 生成数据
    X, y = generate_data(input_size, num_samples)

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
    parser.add_argument('--model_scale_type', dest='model_scale_type', type=str, default='mini', help='Support methods: base|mini|large')
    parser.add_argument('--model_type', dest='model_type', type=str, default='linear', help='Support methods: linear|nonlinear|nonlinearB')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=50, help='Hidden layer size.')
    parser.add_argument('--output_size', dest='output_size', type=int, default=1, help='Output layer size.')
    parser.add_argument('--input_size', dest='input_size', type=int, default=10, help='Input layer size.')
    args = parser.parse_args()
    # main(args)
    main1(args)