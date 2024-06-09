import torch
import numpy as np
from torch.optim import Optimizer

class iKFAD(Optimizer):     # Inherits from class optimizer
    
# '''The following optimizer implements Friction Adaptive Descent. The equations are as follows

#                dx/dt  = p/m
#                dp/dt  = F(x) - ξ*p  - γp                       ---->     Hyperparameters: μ and α, γ, learning rate
#                dξ/dt  = p*p/μ - αξ

# Splitting scheme:
#
# ATTENTION: The symbol "*" below is used to denote element-wise vector multiplication
#         --   --     --   --    --     --      --                               --        --              --
#         |dx/dt|     | p/m |    |   0   |      |             0                   |       |        0        |
#         |dp/dt| =   |  0  | +  |  F(x) |  +   |         -  ξ*p                  |   +   |     - γp        |
#         |dξ/dt|     |  0  |    |   0   |      |        p*p/μ - α*ξ              |       |        0        |
#         --   --     --  --     --     --      --                              --        --              --
#                        A           B                       C                                    D

# Integrator : BACD          (where lr is the time step) 

#                 Step A:    x = x + lr p/m                # Here we take  m = 1
#                 Step B:    p = p +lr F(x)

#                 Step C:    dp/dt = - ξ*p           
#                            dξ/dt = p*p/μ - αξ        

#                            The above system of equations can be solved numerically as

#                            p_{n+1/2} = exp(-ξ_n lr/2) * p_n    where the exponentiation is element-wise (ξ_n is a vector)
#
#                            ξ_{n+1}   = exp(-α*lr)*ξ_n + (1/(α μ)) * (1-exp(-α*lr)) * p_{n+1/2} * p_{n+1/2}
#                            p_{n+1} = exp(-ξ_{n+1} lr/2) * p_{n+1/2}
#
#
#                 Step D:    p_{n+1} = exp(-γh)  p_n 
#
#
#'''


    def __init__(self, params, lr=0.001, alpha=1, mu=1, gamma=1, device='cpu'):
        # Checks on default parameters
        if lr < 0:
            raise ValueError("Invalid learning rate:{} - should be >=0.0".format(lr))
        if alpha < 0:
            raise ValueError("Invalid alph value:{} - should be >=0.0".format(alpha))
        if mu < 0:
            raise ValueError("Invalid mu value:{} - should be >=0.0".format(mu))
#         if gamma < 0:
#             raise ValueError("Invalid gamma value:{} - should be >=0.0".format(gamma))


        defaults = dict(lr=torch.tensor([lr], device=device), 
                        alph=torch.tensor([alpha], device=device), 
                        mu=torch.tensor([mu], device=device),
                        gamma=torch.tensor([gamma], device=device))


        super(iKFAD, self).__init__(params, defaults)  
        
        self.initialized = False



    @torch.no_grad()
    #=========================================================================================================
    def initialize_state(self):
    #=========================================================================================================
        print('Initializing momentum buffer')
        for group in self.param_groups:        
            for q in group["params"]:
                if q.grad is None:             
                    continue                 
                d_q = q.grad
                param_state = self.state[q]   

                # initializing p,ξ
                p_buf   = param_state["momentum_buffer"]   =    torch.zeros_like(q)  #torch.clone(d_q).detach().to(device)
                ksi_buf = param_state["ksi_buffer"]        =    torch.zeros_like(q) 
                
        self.initialized = True



    @torch.no_grad()
    #======================================================================
    def compute_momenta_l1_norm(self):
    #======================================================================
        l1_norm_mom = 0 
        for group in self.param_groups:
            for q in group["params"]:
                param_state = self.state[q]
                p_buf = param_state["momentum_buffer"]
                l1_norm_mom += torch.norm(p_buf, p=1).item()
        return l1_norm_mom


    @torch.no_grad()
    #======================================================================
    def compute_ksi_l1_norm(self):
    #======================================================================
        l1_norm_ksi = 0 
        for group in self.param_groups:
            for q in group["params"]:
                param_state = self.state[q]
                ksi_buf = param_state["ksi_buffer"]
                l1_norm_ksi += torch.norm(ksi_buf, p=1).item()
        return l1_norm_ksi



    @torch.no_grad()
    #=========================================================================================================
    def A_step(self):
    #=========================================================================================================
        '''This step updates q'''
        for group in self.param_groups:
            lr = group["lr"].item() #/ 2            
            for q in group["params"]:
                param_state = self.state[q]
                p_buf       = param_state["momentum_buffer"]
                q.add_(p_buf,alpha =  lr)


    @torch.no_grad()
    #=========================================================================================================
    def B_step(self):
    #=========================================================================================================
        '''This step updates p'''
        for group in self.param_groups:
            lr = group["lr"].item() #/ 2              
            for q in group["params"]:
                param_state = self.state[q]
                p_buf       = param_state["momentum_buffer"]
                d_q         = q.grad
                p_buf.add_(d_q,alpha = - lr)

    
    @torch.no_grad()
    #=========================================================================================================
    def C_step(self):
    #=========================================================================================================
        '''This step updates p,ξ'''
        for group in self.param_groups:
            lr       = group["lr"]                      
            alph    =  group["alph"]
            mu      = group["mu"]

            for q in group["params"]:
                param_state =  self.state[q]
                p_buf       =  param_state["momentum_buffer"]
                ksi_buf     =  param_state["ksi_buffer"]
                
                # updating p
                # p_{n+1/2} = exp(-ξ_n lr/2) * p_n , where "*" denotes element-wise multiplication
                
                p_buf.mul_(torch.exp(-lr * ksi_buf /2))
                
                # Now update ξ
                #ξ_{n+1}   = exp(-α*lr)*ξ_n + (1/(α μ)) * (1-exp(-α*lr)) * p_{n+1/2} * p_{n+1/2}
                ksi_buf.mul_(torch.exp(-alph*lr)).add_( (1-torch.exp(-alph * lr)) * p_buf.pow(2) /(mu*alph)  ) 
                

                # Updating p again
                #p_{n+1} = exp(-ξ_{n+1} lr/2) * p_{n+1/2}
                p_buf.mul_(torch.exp(-lr * ksi_buf /2))




    @torch.no_grad()
    #=========================================================================================================
    def D_step(self):
    #=========================================================================================================
        ''' This step updates p'''
        for group in self.param_groups:
            lr     = group["lr"]#/2                  
            gamma = group["gamma"]
            for q in group["params"]:
                param_state = self.state[q]
                p_buf       = param_state["momentum_buffer"]
                k           = torch.exp(-gamma * lr)                      #.to(device)
                p_buf.mul_(k)


    @torch.no_grad()
    #=========================================================================================================
    def step(self):
    #=========================================================================================================
        if not self.initialized:
            self.initialize_state()
            print("Initialized optimizer state")
            
        ## Remember to change step size within each step if you change integrator        
        self.B_step()
        self.A_step()
        self.C_step()
        self.D_step()

        
        





























