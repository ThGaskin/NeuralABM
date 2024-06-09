import torch
import numpy as np
from torch.optim import Optimizer

class cubic_damping_opt(Optimizer):     # Inherits from class optimizer
    
# '''Linearly dissipated Hamiltonian dynamics integrator
#
#                dx/dt  = p
#                dp/dt  = F(x) - γ p  - c [p]^3    

# Splitting scheme:

#         --   --     --   --       --     --        --                     --          
#         |dx/dt|     |  p  |       |   0   |        |          0            |                
#         |dp/dt| =   |  0  |   +   |  F(q) |   +    |     -c[p]^3 - γ p     |       
#         --   --     --  --        --     --        --                     --                
#                        A              B                      C                            

# Integrator : BACAB          (where lr is the time step)

#                 Step A:    q = q + lr*p/m                # Here we take  m = 1
#
#                 Step B:    p = p +lr*F(q)
#
#                 Step C:   p_{n+1} = p_n ( 1/sqrt(2c[p_n]^2 t + 1 ) )                or ... p_{n+1} = p_n e^(-γ lr) - (c/γ) p_{n+1/2}^3  (1-e^(-γh) )
#
#
#'''

    def __init__(self, params, lr=0.001, gamma=1, c=1, device='cpu'):
        # Checks on default parameters
        if lr < 0:
            raise ValueError("Invalid learning rate:{} - should be >=0.0".format(lr))
        if gamma < 0:
             raise ValueError("Invalid gamma value:{} - should be >=0.0".format(gamma))
        if c < 0:
             raise ValueError("Invalid c value:{} - should be >=0.0".format(c))

        
        defaults = dict(lr=torch.tensor([lr], device=device),
                        gamma=torch.tensor([gamma], device=device), 
                        c=torch.tensor([c], device=device)) 

        super(cubic_damping_opt, self).__init__(params, defaults)  
        
        self.initialized = False
        self.device      = device
        
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
                p_buf   = param_state["momentum_buffer"]   =    torch.clone(d_q).detach()
                ksi_buf = param_state["ksi_buffer"]        =    torch.tensor([0.0],device=self.device)
        
        self.initialized = True
        
        
    @torch.no_grad()
    #=========================================================================================================
    def p_norm_2(self):
    #=========================================================================================================
        squared_norm_p = 0
        for group in self.param_groups:
            for q in group["params"]:
                param_state      =  self.state[q]
                p_buf            =  param_state["momentum_buffer"]
                p_buf_n          =  torch.clone(p_buf).detach()
                squared_norm_p  +=  torch.norm(p_buf_n, p=2) ** 2
#                 squared_norm_p  +=  torch.inner(p_buf_n.ravel(), p_buf_n.ravel())

        return squared_norm_p
    

    @torch.no_grad()
    #======================================================================
    def compute_momenta_l1_norm(self):
    #======================================================================
        l1_norm = 0 
        for group in self.param_groups:
            for q in group["params"]:
                param_state = self.state[q]
                p_buf = param_state["momentum_buffer"]
                l1_norm += torch.norm(p_buf, p=1).item()
        #self.l1_norm_mom = l1_norm
        return(l1_norm)



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
        ''' This step updates p'''
        for group in self.param_groups:
            lr     = group["lr"]
            gamma = group["gamma"]
            c     = group["c"]
            for q in group["params"]:
                param_state     = self.state[q]
                p_buf           = param_state["momentum_buffer"]
                # squared_norm_p  = self.p_norm_2()
                # p_buf.mul_(1/torch.sqrt(2*c*squared_norm_p*lr+1))
                
                aux   = 2*c*lr*p_buf.pow(2) + 1
                denom = aux.sqrt()
                p_buf.div_(denom)

    @torch.no_grad()
    #=========================================================================================================
    def D_step(self):
    #=========================================================================================================
        ''' This step updates p'''
        for group in self.param_groups:
            lr     = group["lr"]    #/2
            gamma = group["gamma"]
            for q in group["params"]:
                param_state = self.state[q]
                p_buf       = param_state["momentum_buffer"]
                k           = torch.exp(-gamma * lr)
                p_buf.mul_(k)



    @torch.no_grad()
    #=========================================================================================================
    def step(self):
    #=========================================================================================================
        if not self.initialized:
            self.initialize_state()
            print("Initialized optimizer state")
            
        self.B_step()
        self.A_step()
        self.C_step()
        self.D_step()
#         A_step()
#         B_step()
        
