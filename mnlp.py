import torch.nn as nn
from nonlinear import Nonlinear, Nonlinear_Broad
    
class Base_Nonlinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Base_Nonlinear, self).__init__()
        self.layer1 = Nonlinear(input_size, output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_Sigmoid(x)
        return x
    
class Base_Nonlinear_Broad(nn.Module):
    def __init__(self, input_size, output_size):
        super(Base_Nonlinear_Broad, self).__init__()
        self.layer1 = Nonlinear(input_size, output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_Sigmoid(x)
        return x

      
class Mini_MNLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Mini_MNLP, self).__init__()
        self.layer1 = Nonlinear(input_size, hidden_size)
        self.layer2 = Nonlinear(hidden_size, output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_Sigmoid(x)
        x = self.layer2(x)
        x = self.activation_Sigmoid(x)
        return x
    
class Large_MNLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Large_MNLP, self).__init__()
        self.mini_mlp1 = Mini_MNLP(input_size, hidden_size, hidden_size)
        self.mini_mlp2 = Mini_MNLP(hidden_size, hidden_size, hidden_size)
        self.mini_mlp3 = Mini_MNLP(hidden_size, hidden_size, output_size)
    
    def forward(self, x):
        x = self.mini_mlp1(x)
        x = self.mini_mlp2(x)
        x = self.mini_mlp3(x)
        return x
    
class Mini_MNLP_Broad(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Mini_MNLP_Broad, self).__init__()
        middle_size = 20
        self.layer1 = Nonlinear_Broad(input_size, middle_size, hidden_size)
        self.layer2 = Nonlinear_Broad(hidden_size, middle_size,  output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class Large_MNLP_Broad(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Large_MNLP_Broad, self).__init__()
        self.mini_mlp1 = Mini_MNLP_Broad(input_size, hidden_size, hidden_size)
        self.mini_mlp2 = Mini_MNLP_Broad(hidden_size, hidden_size, hidden_size)
        self.mini_mlp3 = Mini_MNLP_Broad(hidden_size, hidden_size, output_size)
    
    def forward(self, x):
        x = self.mini_mlp1(x)
        x = self.mini_mlp2(x)
        x = self.mini_mlp3(x)
        return x