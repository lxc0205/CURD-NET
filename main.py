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

def loss_function(outputs, y, tau = 0):
    conditional_uncorrelation = 1 # torch.corrcoef(torch.cat((outputs, y), dim=1)) / torch.corrcoef(outputs)
    assert outputs.shape == y.shape and outputs.dim() == 2 and torch.abs(torch.tensor(conditional_uncorrelation)) <= 1
    return nn.MSELoss()(outputs, y) * (1 - tau) + conditional_uncorrelation * tau # tau is the strength of conditional uncorrelation loss

def train(model, dataloader, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in tqdm(range(epochs)):
        epoch_loss = 0
        for X, y in dataloader:
            X = X.view(X.size(0), -1).float().to(device)
            y = y.view(y.size(0), -1).float().to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        
        if avg_loss < 0.0001:
            print(f'Training stopped early as loss reached {avg_loss:.5f}')
            break
    return model

def test(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in  tqdm(dataloader):
            X = X.view(X.size(0), -1).float().to(device)
            y = y.view(y.size(0), -1).float().to(device)
            predictions = model(X)
            batch_loss = loss_function(predictions, y)
            total_loss += batch_loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Test Loss: {avg_loss:.5f}')

# 主程序
def main(args):
    input_size, hidden_size, output_size = 784, 196, 1
    epochs = 1000
    learning_rate = 0.01
    model_scale_type = args.model_scale_type # base, mini, or large
    
    if model_scale_type == 'base':
        mlp = Base_Linear(input_size, output_size).to(device)
        mnlp = Base_Nonlinear(input_size, output_size).to(device)
        mnlpb = Base_Nonlinear_Broad(input_size, output_size).to(device)
    if model_scale_type == 'mini':
        mlp = Mini_MLP(input_size, hidden_size, output_size).to(device)
        mnlp = Mini_MNLP(input_size, hidden_size, output_size).to(device)
        mnlpb = Mini_MNLP_Broad(input_size, hidden_size, output_size).to(device)
    if model_scale_type == 'large':
        mlp = Large_MLP(input_size, hidden_size, output_size).to(device)
        mnlp = Large_MNLP(input_size, hidden_size, output_size).to(device)
        mnlpb = Large_MNLP_Broad(input_size, hidden_size, output_size).to(device)

    # 数据集
    train_dataloader, test_dataloader = mnist_dataloader(train_batch_size=64, test_batch_size=1)

    # 训练
    start_time = time.time()
    # model = train(mlp, train_dataloader, epochs, learning_rate)
    # model = train(mnlp, train_dataloader, epochs, learning_rate)
    model = train(mnlpb, train_dataloader, epochs, learning_rate)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")
    test(model, test_dataloader)
  
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_scale_type', dest='model_scale_type', type=str, default='mini', help='Support methods: base|mini|large')
    args = parser.parse_args()
    main(args)