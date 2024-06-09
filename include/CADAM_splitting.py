import torch
from torch.optim import Optimizer
import numpy as np

class Cadam(Optimizer):
    
    # '''The dynamics are as follows:
    #
    #                dx/dt  = p/ sqrt(ζ + ε)
    #                dp/dt  = F(x) - γ p - c p^3                ---->     Hyperparameters: γ, c, α, ε
    #                dζ/dt  = F(x)^2 - α ζ

    # Splitting scheme:

    #         --   --     --                --      --     --      --             --     --       --     --    --
    #         |dx/dt|     |   p/ sqrt(ζ + ε) |     |   0    |     |      0        |     |     0      |     |   0   |
    #         |dp/dt| =   |         0        |  +  |  F(x)  |  +  |      0        |  +  | - c [p]^3  |  +  | - γp  |
    #         |dξ/dt|     |         0        |     |   0    |     |[F(x)]^2 - α ζ |     |     0      |     |   0   |
    #         --   --     --               --      --     --      --             --     --          --     --     --
    #                               A                   B                     C               D                E

    # Integrator : DABCBAD          (where lr is the time step) 

    #                 Step A:    x_{n+1} = x_n + lr p/sqrt(ζ + ε)               # Since p,ζ fixed during A step
    #                 Step B:    p_{n+1} = p_n + lr F(x)

    #                 Step C:    ζ_{n+1} = e^(-αh) ζ_n + F^2 (1-e^(-αh))/α
    #                            
    #                 Step D:    p_{n+1} = p_n / sqrt(2ch p_n^2 + 1)
    #                              
    #                 Step E:    p_{n+1} = e^(-γh) p_n
    #
    #'''

    #==================================================================================
    def __init__(self, params, lr=0.001, gamma=1, c=1, alpha=1, eps=1e-8, device='cpu'):
    #==================================================================================
        # Checks on default parameters
        if lr < 0:
            raise ValueError("Invalid learning rate:{} - should be >=0.0".format(lr))
        if c < 0:
            raise ValueError("Invalid c value:{} - should be >=0.0".format(c))
        if gamma < 0:
            raise ValueError("Invalid gamma value:{} - should be >=0.0".format(gamma))
        if alpha < 0:
            raise ValueError("Invalid alpha value:{} - should be >=0.0".format(alpha))
        if eps < 0:
            raise ValueError("Invalid eps value:{} - should be >=0.0".format(eps))

        defaults = dict(lr=torch.tensor([lr], device = device), 
                        gamma=torch.tensor([gamma], device = device), 
                        c=torch.tensor([c], device = device), 
                        alpha=torch.tensor([alpha], device = device), 
                        eps=torch.tensor([eps], device = device))   
        super(Cadam, self).__init__(params, defaults)  
        self.t = 0
        self.initialized = False
        self.device = device

    
    @torch.no_grad()
    #======================================================================
    def initialize_state(self):
    #======================================================================
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer']    = torch.zeros_like(p.data, device=self.device)
                state['exp_avg_sq']         = torch.zeros_like(p.data, device=self.device)
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
        '''This step updates x:
        
           x_{n+1} = x_n + lr p/sqrt(ζ + ε)               # Since p,ζ fixed during A step
           
        '''
        for group in self.param_groups:
            lr = group["lr"].item()   #/2  
            eps   = group["eps"]

            for q in group["params"]:
                param_state = self.state[q]
                p_buf       = param_state["momentum_buffer"]
                zeta_buf    = param_state["exp_avg_sq"]

                denom       = zeta_buf.sqrt().add_(eps)    # denom = sqrt(ζ) + ε
                q.addcdiv_(p_buf, denom, value=lr)         # x = x + lr p_buf/ denom



    @torch.no_grad()
    #=========================================================================================================
    def B_step(self):
    #=========================================================================================================
        '''This step updates p:
        
           p_{n+1} = p_n + lr F(x)  
           
        '''
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
        '''This step updates zeta:
        
           ζ_{n+1} = e^(-αh) ζ_n + F^2 (1-e^(-αh))/α 
           
        '''
        for group in self.param_groups:
            lr       = group["lr"].item()                      
            alpha    = group["alpha"].item()

            for q in group["params"]:
                param_state =  self.state[q]
                zeta_buf    =  param_state["exp_avg_sq"]
                F           = -torch.clone(q.grad).detach()   # do we need all this clone and detaching stuff?
                expnt       = np.exp(-alpha * lr)
                zeta_buf.mul_(expnt).addcmul_(F, F, value=(1-expnt)/alpha)

                
                
    @torch.no_grad()
    #=========================================================================================================
    def D_step(self):
    #=========================================================================================================
        ''' This step updates p:
            
            p_{n+1} = p_n / sqrt(2ch p_n^2 + 1)    # p_n^2 can either be the norm of p squared or! we could individually 
                                                   # square each component of p_n                         
        '''
        
        for group in self.param_groups:
            lr     = group["lr"]                   
            gamma = group["gamma"]
            c     = group["c"]
            for q in group["params"]:
                param_state     = self.state[q]
                p_buf           = param_state["momentum_buffer"]
                
##                 Uncomment if you want norm of p rather than individual squares
#                 squared_norm_p  = self.p_norm_2()
#                 p_buf.mul_(1/torch.sqrt(2*c*squared_norm_p*lr+1))

                aux   = 2*c*lr*p_buf.pow(2) + 1
                denom = aux.sqrt()
                p_buf.div_(denom)


                
    @torch.no_grad()
    #=========================================================================================================
    def E_step(self):
    #=========================================================================================================
        ''' This step updates p:
        
            p_{n+1} = e^(-γh) p_n

        '''
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        for group in self.param_groups:
            lr     = group["lr"]#/2            
            gamma = group["gamma"]
            for q in group["params"]:
                param_state = self.state[q]
                p_buf       = param_state["momentum_buffer"]
                k           = torch.exp(-gamma * lr)                     
                p_buf.mul_(k)
       
        
        
    @torch.no_grad()
    #==============================================================================
    def step(self):
    #==============================================================================    
        if not self.initialized:
            self.initialize_state()
            print("Initialized optimizer state")
            
        self.B_step()
        self.C_step()
        self.A_step()
        self.D_step()
        self.E_step()

    
    
