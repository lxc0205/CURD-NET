import math
import torch
import random
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from funcitions import origin_norm

class Nonlinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = nonlinear(x)A^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        base_func: the set of base functions
        func_num: the number of base functions
        random_num: the number of selected base functions
        weight: the learnable weights.
       
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = Nonlinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
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

    def nonlinear_layer(self, x):
        # Input data x must be in the range [0, 1]，or对每个特征列进行归一化
        assert (x >= 0).all() and (x <= 1).all(), "Input data must be in the range [0, 1]"
        funcs_index = random.sample(range(self.func_num), self.ramdom_num)
        funcs_index.sort()
        x_expand = torch.hstack([self.base_func[index](x) for index in funcs_index])
        assert not torch.isnan(x_expand).any(), "expand function warning: Array contains NaN."
        assert not torch.isinf(x_expand).any(), "expand function warning: Array contains Inf."
        return x_expand
    
    def forward(self, input: Tensor) -> Tensor: 
        # 64*8*784=6272
        print(input.shape)
        print(self.weight.shape)
        x = F.linear(self.nonlinear_layer(input), self.weight, self.bias)
        return x
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, base_func={self.base_func}, func_num={self.func_num}'

class Nonlinear_Broad(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = nonlinear((x)A+x)^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        base_func: the set of base functions
        func_num: the number of base functions
        random_num: the number of selected base functions
        weight: the learnable weights.
       
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = Nonlinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
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
    def nonlinear_layer(self, x):
        # Input data x must be in the range [0, 1]，or对每个特征列进行归一化
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


    # model2 = torch.nn.Linear(20, 30)
    # output2 = model2(input)
    # print(output2.size())