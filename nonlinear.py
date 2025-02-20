import math
import torch
import random
from torch import Tensor
from torch.nn import init
from funcitions import origin_norm
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class Nonlinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, base_func=origin_norm, random_num = 8) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_func = base_func
        self.func_num = len(base_func)
        self.ramdom_num = random_num
        self.weight = Parameter(torch.empty((out_features, in_features*random_num), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()    

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def minMax_normalization(self, x):
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
        if (x_max - x_min).sum() == 0:
            return x
        else:
            return (x - x_min) / (x_max - x_min + 1e-8)

    def nonlinear_layer(self, x):
        x = self.minMax_normalization(x)
        assert (x >= 0).all() and (x <= 1).all(), "Input data must be in the range [0, 1]"
        funcs_index = random.sample(range(self.func_num), self.ramdom_num)
        funcs_index.sort()
        x_expand = torch.hstack([self.base_func[index](x) for index in funcs_index])
        assert not torch.isnan(x_expand).any(), "expand function warning: Array contains NaN."
        assert not torch.isinf(x_expand).any(), "expand function warning: Array contains Inf."
        return x_expand
    
    def forward(self, input: Tensor) -> Tensor: 
        x = F.linear(self.nonlinear_layer(input), self.weight, self.bias)
        return x
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, base_func={self.base_func}, func_num={self.func_num}'

class Nonlinear_Broad(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, base_func=origin_norm, random_num = 8) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_func = base_func
        self.func_num = len(base_func)
        self.ramdom_num = random_num
        self.weight1 = Parameter(torch.empty((hidden_features, in_features), **factory_kwargs))
        self.weight2 = Parameter(torch.empty((out_features, hidden_features*random_num), **factory_kwargs))
        if bias:
            self.bias1 = Parameter(torch.empty(hidden_features, **factory_kwargs))
            self.bias2 = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()    

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

        if self.bias1 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias1, -bound, bound)
            
        if self.bias2 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias2, -bound, bound)

    def minMax_normalization(self, x):
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
        if (x_max - x_min).sum() == 0:
            return x
        else:
            return (x - x_min) / (x_max - x_min + 1e-8)


    def nonlinear_layer(self, x):
        x = self.minMax_normalization(x)
        assert (x >= 0).all() and (x <= 1).all(), "Input data must be in the range [0, 1]"
        funcs_index = random.sample(range(self.func_num), self.ramdom_num)
        funcs_index.sort()
        x_expand = torch.hstack([self.base_func[index](x) for index in funcs_index])
        assert not torch.isnan(x_expand).any(), "expand function warning: Array contains NaN."
        assert not torch.isinf(x_expand).any(), "expand function warning: Array contains Inf."
        return x_expand
    
    def forward(self, input: Tensor) -> Tensor: 
        x = F.linear(input, self.weight1, self.bias1)
        x = torch.nn.Sigmoid()(x)
        x = F.linear(self.nonlinear_layer(x), self.weight2, self.bias2)     
        x = torch.nn.Sigmoid()(x)
        return x
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, base_func={self.base_func}, func_num={self.func_num}, random_num={self.ramdom_num}'


if __name__ == '__main__':
    # 测试对比
    input = torch.rand(128, 20)

    model0 = Nonlinear_Broad(20, 20, 30)
    output0 = model0(input)
    print(output0.size())

    # model1 = Nonlinear(20, 30)
    # output1 = model1(input)
    # print(output1.size())