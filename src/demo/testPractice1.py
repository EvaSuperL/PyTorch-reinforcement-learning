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
# print(torch.rand(2,2,1))
# print(torch.rand(2,3))
# print(torch.rand(2,2,1)+torch.rand(2,3))


# a = torch.rand(2,2)
# a = a*10
#
# print(a)
#
# print(a.floor())
# print(a.ceil())
# print(a.abs())
# print(a.round())
# print(a.trunc())
# print(a.frac())
# print(a%2)

# print(torch.eq(torch.tensor([1,2]),torch.tensor(3)))
# print(torch.equal(torch.tensor([[2,3],[1,2]]),torch.tensor([2,3])))
# print(torch.ge(torch.tensor([1,3]),torch.tensor(3)))
# print(torch.gt(torch.tensor([1,3]),torch.tensor(3)))
# print(torch.le(torch.tensor([1,3]),torch.tensor(3)))
# print(torch.lt(torch.tensor([1,3]),torch.tensor(3)))
# print(torch.ne(torch.tensor([1,3]),torch.tensor(3)))
# a = torch.randn(10)
# print(torch.sort(a,descending=True,out=None)[0])
# print(torch.sort(a, dim=0))
# print(torch.topk(a, dim=0, k=2))
# print(torch.kthvalue(a, dim=0,k=2,out=None))

# print(torch.isfinite(torch.ones(5)))
# print(torch.isinf(torch.ones(5)))
# print(torch.isnan(torch.ones(5)))
# a = torch.rand([2,3])
# print(torch.isnan(a))
# print(torch.isnan(a).sum())

#######################################

# a = torch.zeros(2,3)
# b = torch.atan(a)
# print(a)
# print(b)
#
# a = torch.randn(5,5)
# print(a)
# print(torch.prod(a,dim=0))
#
# print(torch.histc(a, 6,0,1))
# b = torch.randint(0,10,[2,10])
# print(torch.bincount(b))

# torch.manual_seed(1)
# mean = torch.rand(1,2)
# std = torch.rand(1,2)
# print(torch.normal(mean,std))

# a = torch.randn(2,1)
# b = torch.randn(2,1)
# print(a, b)
# print(torch.dist(a, b,p=1)) # sum(|a-b|)
# print(torch.dist(a,b,p=2)) # ((a-b)^2)^(1/2)
# print(torch.dist(a,b,p=3)) # ((a-b)^3)^(1/3)
#
# print(torch.norm(a))
# print(torch.norm(a,p=1))

# a = torch.rand(2,2)*10
# print(a)
#
# b = a.clamp(2,6)
# print(b)
# a = torch.rand(4,4)
# b = torch.rand(4,4)
# print(a)
# print(b)

# out = torch.where(a>0.5,a,b)
# print(out)
# out = torch.index_select(a,0,index=torch.tensor([0,3,2]))
# print(out)

# a = torch.linspace(1,16,16).view(4,4)
# print(a)
#
# out = torch.gather(a,dim=-2,index=torch.tensor([[0,1,1,1],
#                                                [0,1,2,2],
#                                                [0,3,3,2]]))
# print(out)
# a = torch.linspace(1,16,16)
# a = torch.tensor([[0,1,2,0],[2,3,0,1]])
# print(a)
# mask = torch.gt(a,8)
# print(mask)
# out = torch.masked_select(a,mask)
# print(out)

# out = torch.take(a,index=torch.tensor([0,15,13,10]))
# out = torch.nonzero(a)
# print(out)

#######################################
# a = torch.zeros((2,4))
# b = torch.ones((2,4))
# a = torch.linspace(1,6,6).view(2,3)
# b = torch.linspace(7,12,6).view(2,3)
# print(a)
# print(b)
#
# # out = torch.cat((a, b),dim=1)
# out = torch.stack((a,b),dim=2)
#
# print(out, out.shape)
#
# a = torch.rand((3,4))
# print(a)
#
# # out = torch.chunk(a, 2, 1)
# out = torch.split(a,3,dim=0)
# print(out)

# a = torch.rand(1,2,3)
# print(a)
# print(a.shape)
# out = torch.reshape(a,(3,2))
# out = torch.transpose(a,0,1)
# out = torch.unsqueeze(a,-1)
# out = torch.unbind(a, dim=1)
# out = torch.flip(a, dims=[1,2])
# out = torch.rot90(a, 1, (2,1))

# print(out, len(out))

# a = torch.full((3,3), 5)
# print(a)

# import numpy as np
# a = torch.from_numpy(np.ones((3,3,3)))
# print(a)
#
# b = a.numpy()
# print(b)

#######################################
#
# class line(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, w, x, b):
#         ctx.save_for_backward(w, x, b)
#         print("w*x+b: \n")
#         print(w*x+b)
#         return w*x+b
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         w, x, b = ctx.saved_tensors
#
#         grad_w = grad_output *x
#         grad_x = grad_output *w
#         grad_b = grad_output
#         print("grad_output: \n")
#         print(grad_output)
#         print("grad_w, grad_x, grad_b: \n")
#         print(grad_w, grad_x, grad_b)
#         return grad_w, grad_x, grad_b
#
#         return grad_w, grad_x, grad_b
#
# w = torch.rand(2,2,requires_grad=True)
# x = torch.rand(2,2,requires_grad=True)
# b = torch.rand(2,2,requires_grad=True)
#
# out = line.apply(w, x, b)
# out.backward(torch.ones(2,2))
#
# print(out)
# print(w, x, b)
# print(w.grad,x.grad,b.grad)

#######################################
# from tensorboardX import SummaryWriter
#
# writer = SummaryWriter("log")
#
# for i in range(100):
#     writer.add_scalar("a",i,global_step=i)
#     writer.add_scalar("b",i**2,global_step=i)
#
# writer.close()

#######################################

