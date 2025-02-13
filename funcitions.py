import torch
import pywt

origin_norm = [lambda x: x,
               lambda x: x**2, 
               lambda x: torch.sqrt(x), 
               lambda x: x**3, 
               lambda x: x**(1/3), 
               lambda x: torch.log(x+1) / torch.log(torch.tensor(2.)),
               lambda x: torch.pow(2, x) - 1,
               lambda x: (torch.exp(x)-1) / (torch.exp(torch.tensor(1.))-1)]

origin = [lambda x: x,
          lambda x: x**2, 
          lambda x: torch.sqrt(x), 
          lambda x: x**3, 
          lambda x: x**(1/3), 
          lambda x: torch.log(x),
          lambda x: torch.pow(2, x),
          lambda x: torch.exp(x)]

k = 1
fourier = [lambda x: torch.sin(k*x),
              lambda x: torch.cos(k*x),
              lambda x: torch.sin(2*k*x),
              lambda x: torch.cos(2*k*x),
              lambda x: torch.sin(3*k*x),
              lambda x: torch.cos(3*k*x),
              lambda x: torch.sin(4*k*x),
              lambda x: torch.cos(4*k*x)]


wavelet_names = ['db4', 'sym4', 'coif3', 'haar', 'bior2.2']
wavelet = pywt.Wavelet(wavelet_names[0])
level = 1
wavelet_function = [lambda x: pywt.wavedec(x, wavelet, level=1)[0],
                    lambda x: pywt.wavedec(x, wavelet, level=1)[1],
                    lambda x: pywt.wavedec(x, wavelet, level=1)[2],
                    lambda x: pywt.wavedec(x, wavelet, level=1)[3],
                    lambda x: pywt.wavedec(x, wavelet, level=1)[4],
                    lambda x: pywt.wavedec(x, wavelet, level=1)[5],
                    lambda x: pywt.wavedec(x, wavelet, level=1)[6],
                    lambda x: pywt.wavedec(x, wavelet, level=1)[7]]


def generate_legendre_basis(max_degree, x):
    legendre_basis = []
    for i in range(max_degree + 1):
        def legendre_func():
            # 使用 torch 计算勒让德多项式，这里假设输入 x 是一个张量
            # torch.polyval 可以计算多项式在 x 处的值
            coeffs = torch.zeros(i + 1)
            coeffs[-1] = 1  # 勒让德多项式的最高次项系数为 1
            value = torch.polyval(coeffs, x)
            return value
        legendre_basis.append(legendre_func)
    return legendre_basis


def generate_chebyshev_basis(max_degree, x):
    chebyshev_basis = []
    for i in range(max_degree + 1):
        def chebyshev_func():
            # 这里简单实现 Chebyshev 多项式的递推公式 Tn+1(x) = 2xTn(x) - Tn-1(x)
            if i == 0:
                return torch.ones_like(x)
            elif i == 1:
                return x
            else:
                Tn_minus_1 = torch.ones_like(x)
                Tn = x
                for n in range(2, i + 1):
                    Tn_plus_1 = 2 * x * Tn - Tn_minus_1
                    Tn_minus_1 = Tn
                    Tn = Tn_plus_1
                return Tn
        chebyshev_basis.append(chebyshev_func)
    return chebyshev_basis


# 示例输入张量 x
x = torch.tensor([1.0, 2.0, 3.0])
max_degree = 3


# 生成勒让德基底函数
legendre_basis = generate_legendre_basis(max_degree, x)
legendre_values = [func() for func in legendre_basis]
print("Legendre basis functions values:", legendre_values)


# 生成切比雪夫基底函数
chebyshev_basis = generate_chebyshev_basis(max_degree, x)
chebyshev_values = [func() for func in chebyshev_basis]
print("Chebyshev basis functions values:", chebyshev_values)