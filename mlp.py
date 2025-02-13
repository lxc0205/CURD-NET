import torch.nn as nn

class Base_Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Base_Linear, self).__init__()     
        self.layer1 = nn.Linear(input_size, output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_Sigmoid(x)
        return x

class Mini_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Mini_MLP, self).__init__()     
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_Sigmoid(x)
        x = self.layer2(x) 
        x = self.activation_Sigmoid(x)
        return x

class Large_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Large_MLP, self).__init__()
        self.mini_mlp1 = Mini_MLP(input_size, hidden_size, hidden_size)
        self.mini_mlp2 = Mini_MLP(hidden_size, hidden_size, hidden_size)
        self.mini_mlp3 = Mini_MLP(hidden_size, hidden_size, output_size)
    
    def forward(self, x):
        x = self.mini_mlp1(x)
        x = self.mini_mlp2(x)
        x = self.mini_mlp3(x)
        return x
    