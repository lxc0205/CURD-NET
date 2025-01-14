import torch

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

fourier = [lambda x: torch.sin(x),
              lambda x: torch.cos(x),
              lambda x: torch.sin(2*x),
              lambda x: torch.cos(2*x),
              lambda x: torch.sin(3*x),
              lambda x: torch.cos(3*x),
              lambda x: torch.sin(4*x),
              lambda x: torch.cos(4*x)]