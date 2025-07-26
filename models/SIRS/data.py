import torch

nrtestsets = 3

gridsteps = 2

nrallsets = 13
#different seeds for different sets
torch.manual_seed(21)
y0_S = (0.99-0.8)*torch.rand(nrallsets)+0.8
y_0_I = torch.ones(nrallsets)-y0_S
y0 = torch.stack([y0_S, y_0_I, torch.zeros(nrallsets)])

kI = (0.9-0.7)*torch.rand(nrallsets)+0.7 #2*torch.rand(nrallsets)#torch.hstack((2*torch.rand(nrtestsets),torch.Tensor([0.52,0.88,0.52,0.68]),2*torch.rand(nrallsets-7)))#(0.9-0.5)*torch.rand(nrallsets)+0.5
kR = 0.2*torch.rand(nrallsets)+0.4#torch.hstack((0.3*torch.rand(nrtestsets),torch.Tensor([0.48,0.12,0.12,0.25]),0.3*torch.rand(nrallsets-7)))
kS = 0.02*torch.rand(nrallsets)+0.03#torch.hstack((0.5*torch.rand(nrtestsets),torch.Tensor([0.012,0.012,0.048,0.02]),0.5*torch.rand(nrallsets-7)))
trueparameters = dict(k_I=kI, k_R=kR, k_S=kS)

kI_min, kI_max = 0.5, 0.9
kR_min, kR_max = 0.1, 0.5
kS_min, kS_max = 0.01, 0.05

# Create 5 evenly spaced points for each range
kI_values = torch.linspace(kI_min, kI_max, steps=gridsteps)
kR_values = torch.linspace(kR_min, kR_max, steps=gridsteps)
kS_values = torch.linspace(kS_min, kS_max, steps=gridsteps)
"""
# Create the 5x5x5 grid
kI_grid, kR_grid, kS_grid = torch.meshgrid(kI_values, kR_values, kS_values, indexing='ij')

# Combine the grids into a single tensor of shape [125, 3] (flattened grid)
grid = torch.stack([kI_grid.flatten(), kR_grid.flatten(), kS_grid.flatten()], dim=1)

trueparameters = dict(k_I=torch.hstack(((0.9-0.5)*torch.rand(nrtestsets)+0.5,grid[:,0])), 
                      k_R= torch.hstack(((0.5-0.1)*torch.rand(nrtestsets)+0.1,grid[:,1])), 
                      k_S=torch.hstack(((0.05-0.01)*torch.rand(nrtestsets)+0.01,grid[:,2])))
"""


torch.save(y0,'y0_rand11.pt')
torch.save(trueparameters,f'trueparameters_rand11.pt')