import torch
from numpy.ma.core import size

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.backends.mps.is_available())
# print(torch.backends.mps.is_built())
#
# a = torch.Tensor([[1,2],[3,4]])
# print(a)
# print(a.type())
#
# a = torch.Tensor(2,3)
# print(a)
# print(a.type())
#
# print(torch.ones(2,2))
# print(torch.eye(2,2))
# print(torch.zeros(2,2))
# print(torch.zeros_like(a))
# print(torch.ones_like(a))
# print(torch.rand(2,2))
# print(torch.normal(mean=0.0,std=torch.rand(5)))
# print(torch.normal(mean=torch.rand(5),std=torch.rand(5)))
# print(torch.Tensor(2,2).uniform_(-1,1).type())
# print(torch.arange(0.0,10.0,1).type())
# print(torch.linspace(2,10,2).type())
# print(torch.randperm(10).type())
#
# ######################################
# import numpy as np
#
# b = np.array([[1,2,3],[4,5,6]])
# print(b)
# print(b.dtype)
#
# ######################################
# a = torch.tensor([1,2,3],dtype=torch.float32,device=torch.device('cpu'))
# print(a.dtype)
# print(a.device)
#
# indices = torch.tensor([[0,1,2],[2,0,2]])
# values = torch.tensor([3,4,5],dtype=torch.float32)
# b = torch.sparse_coo_tensor(indices=indices, values=values, size=[2,4])
# print(b)

#######################################
# dev = torch.device('cpu')
# # dev = torch.device('cuda')
# a = torch.tensor([2,2],device=dev)
# print(a.device)

# i = torch.tensor([[0,1,2],[0,1,2]])
# v = torch.tensor([1,2,3],dtype=torch.float32)
# b = (torch.sparse_coo_tensor(i, v,(4,4),dtype=torch.float32,
#                             device=torch.device('mps'))
#      .to_dense())
# print(b)
#######################################
# a = torch.rand(2,3)
# b = torch.rand(2,3)
# print(a)
# print(b)
#
# print(a+b)
# print(a.add(b))
# print(torch.add(a,b))
# print(a.add_(b))
# print(a)
# print(a-b)
# print(torch.sub(a,b))
# print(a.sub(b))
# print(a.sub_(b))
# print(a)

# print(a*b)
# print(torch.mul(a,b))
# print(a.mul(b))
# print(a.mul_(b))
# print(a)
# print(a/b)
# print(torch.div(a,b))
# print(a.div(b))
# print(a.div_(b))
# print(a)

# a = torch.ones(2,1)
# b = torch.ones(1,2)

# print(a@b)
# print(torch.matmul(a,b))
# print(a.matmul(b))
# print(torch.mm(a,b))
# print(a.mm(b))
# print(a.mm(b))
# print(a)

# a = torch.ones(1,2,3,4)
# b = torch.ones(1,2,4,5)
# print(a.matmul(b))
# print(a.shape)
# print(b.shape)

# a = torch.tensor([1,2],dtype=torch.float32)
# print(torch.pow(a,2))
# print(a.pow(2))
# print(a**2)
# print(a.pow_(2))
# print(a)

# print(torch.exp(a))
# print(a.exp())
# print(a)
# print(torch.exp_(a))
# print(a.exp_())
# print(a)

# print(torch.log(a))
# print(a.log())
# print(a.log_())
# print(a)

# print(torch.sqrt(a))
# print(a.sqrt())
# print(a.sqrt_())
# print(a)
#######################################




