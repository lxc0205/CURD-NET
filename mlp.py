import torch
import torch.nn as nn
from nonlinear import Nonlinear, Nonlinear_Broad
from torch.nn.utils import weight_norm

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
    
class Base_NonlinearB(nn.Module):
    def __init__(self, input_size, output_size):
        super(Base_NonlinearB, self).__init__()
        self.layer1 = Nonlinear(input_size, output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_Sigmoid(x)
        return x

class MiniMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MiniMLP, self).__init__()     
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

class LargeMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LargeMLP, self).__init__()
        self.mini_mlp1 = MiniMLP(input_size, hidden_size, hidden_size)
        self.mini_mlp2 = MiniMLP(hidden_size, hidden_size, hidden_size)
        self.mini_mlp3 = MiniMLP(hidden_size, hidden_size, output_size)
    
    def forward(self, x):
        x = self.mini_mlp1(x)
        x = self.mini_mlp2(x)
        x = self.mini_mlp3(x)
        return x
      
class MiniMLP_nonlinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MiniMLP_nonlinear, self).__init__()
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
    
class LargeMLP_nonlinear(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LargeMLP_nonlinear, self).__init__()
        self.mini_mlp1 = MiniMLP_nonlinear(input_size, hidden_size, hidden_size)
        self.mini_mlp2 = MiniMLP_nonlinear(hidden_size, hidden_size, hidden_size)
        self.mini_mlp3 = MiniMLP_nonlinear(hidden_size, hidden_size, output_size)
    
    def forward(self, x):
        x = self.mini_mlp1(x)
        x = self.mini_mlp2(x)
        x = self.mini_mlp3(x)
        return x
    
class MiniMLP_nonlinearB(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MiniMLP_nonlinearB, self).__init__()
        middle_size = 20
        self.layer1 = Nonlinear_Broad(input_size, middle_size, hidden_size)
        self.layer2 = Nonlinear_Broad(hidden_size, middle_size,  output_size)
        self.activation_ReLU = nn.ReLU()
        self.activation_Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class LargeMLP_nonlinearB(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LargeMLP_nonlinearB, self).__init__()
        self.mini_mlp1 = MiniMLP_nonlinearB(input_size, hidden_size, hidden_size)
        self.mini_mlp2 = MiniMLP_nonlinearB(hidden_size, hidden_size, hidden_size)
        self.mini_mlp3 = MiniMLP_nonlinearB(hidden_size, hidden_size, output_size)
    
    def forward(self, x):
        x = self.mini_mlp1(x)
        x = self.mini_mlp2(x)
        x = self.mini_mlp3(x)
        return x