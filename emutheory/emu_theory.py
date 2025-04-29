import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, TRF
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict

class Supact(nn.Module):
    # New activation function, returns:
    # f(x)=(gamma+(1+exp(-beta*x))^(-1)*(1-gamma))*x
    # gamma and beta are trainable parameters.
    # I chose the initial value for gamma to be all 1, and beta to be all 0
    def __init__(self, in_size):
        super(Supact, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(in_size))
        self.beta = nn.Parameter(torch.zeros(in_size))
        self.m = nn.Sigmoid()
    def forward(self, x):
        inv = self.m(torch.mul(self.beta,x))
        fac = 1-self.gamma
        mult = self.gamma + torch.mul(inv,fac)
        return torch.mul(mult,x)

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        # This function is designed for the Neuro-network to learn how to normalize the data between
        # layers. we will initiate gains and bias both at 1 
        self.gain = nn.Parameter(torch.ones(1))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        return x * self.gain + self.bias

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = Supact(in_size)#nn.Tanh()#nn.ReLU()#
        self.norm         = Affine()#torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights = nn.Parameter(weights) # turn the weights tensor into trainable weights
        bias = torch.Tensor(in_size)
        self.bias = nn.Parameter(bias) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5)) # matrix weights init 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights) # fan_in in the input size, fan out is the output size but it is not use here
        bound = 1 / np.sqrt(fan_in) 
        nn.init.uniform_(self.bias, -bound, bound) # bias weights init

    def forward(self,x):
        mat = torch.block_diag(*self.weights) # how can I do this on init rather than on each forward pass?
        x_norm = self.norm(x)
        #_x = x_norm.reshape(x_norm.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        o = self.act(torch.matmul(x_norm,mat)+self.bias)
        return o+x

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = Supact(in_size)#nn.Tanh()#nn.ReLU()#
        self.act2 = Supact(in_size)#nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.layer1(self.norm1(x)))
        o2 = self.act2(self.layer2(self.norm2(o1))) + xskip

        return o2


class ResMLP(nn.Module):

    def __init__(self, input_dim, output_dim, int_dim, N_layer):

        super(ResMLP, self).__init__()

        modules=[]

        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim * 128

        # Def: we will only change the dimension of the datavector using linear transformations  
        modules.append(nn.Linear(input_dim, int_dim))
        
        # Def: by design, a pure block has the input and output dimension to be the same
        for n in range(N_layer):
            # Def: This is what we defined as a pure MLP block
            # Why the Affine function?
            #   R: this is for the Neuro-network to learn how to normalize the data between layer
            modules.append(ResBlock(int_dim, int_dim))
            modules.append(Supact(int_dim))
        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        
        modules.append(nn.Linear(int_dim, output_dim))
        modules.append(Affine())
        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
        self.resmlp =nn.Sequential(*modules)#

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.resmlp(x)

        return out



class TRF(nn.Module):

    def __init__(self, input_dim, output_dim, int_dim, int_trf, N_channels):

        super(TRF, self).__init__()

        modules=[]

        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim * 128

        # Def: we will only change the dimension of the datavector using linear transformations  
        
        
        n_channels = N_channels
        int_dim_trf = int_trf
        modules.append(nn.Linear(input_dim, int_dim))
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(Supact(int_dim))
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(Supact(int_dim))
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(Supact(int_dim))
        modules.append(nn.Linear(int_dim, int_dim_trf))
        modules.append(Supact(int_dim_trf))
        modules.append(Better_Attention(int_dim_trf, n_channels))
        modules.append(Better_Transformer(int_dim_trf, n_channels))
        modules.append(nn.Linear(int_dim_trf, output_dim))
        modules.append(Affine())

        self.trf =nn.Sequential(*modules)#
        

    def forward(self, x):

        out = self.trf(x)
        
        return out

class emutheory(BoltzmannBase):
    aliases: dict = {
        "omega_b" : [ "ombh2" ],
        "omega_cdm" : [ "omch2" ],
        "H_0" : [ "H0" ],
        "ln10^{10}A_s" : [ "logA" ],
        "n_s" : [ "ns" ],
        "tau_reio" : [ "tau" ],
    }

    extra_args: InfoDict = { }

    def initialize(self):
        super().initialize()
        PATH1 = "./cobaya/cobaya/theories/emutheory/chiTTAstautrfaxcamb16"
        PATH2 = "./cobaya/cobaya/theories/emutheory/chiTEAstautrfaxcamb16"
        PATH3 = "./cobaya/cobaya/theories/emutheory/chiEEAstautrfaxcamb16"
        intdim = 4
        nc = 16
        device = 'cpu'
        self.model1 = TRF(input_dim=6,output_dim=2999,int_dim=intdim, int_trf=3200,N_channels=nc)
        self.model2 = TRF(input_dim=6,output_dim=2999,int_dim=intdim, int_trf=3200,N_channels=nc)
        self.model3 = TRF(input_dim=6,output_dim=2999,int_dim=intdim, int_trf=3200,N_channels=nc)

        #model=model.module.to(device)
        self.model1 = self.model1.to(device)
        self.model2 = self.model2.to(device)
        self.model3 = self.model3.to(device)

        self.model1 = nn.DataParallel(self.model1)#.cpu()
        self.model2 = nn.DataParallel(self.model2)
        self.model3 = nn.DataParallel(self.model3)


        self.model1.load_state_dict(torch.load(PATH1+'.pt',map_location=device))
        self.model2.load_state_dict(torch.load(PATH2+'.pt',map_location=device))
        self.model3.load_state_dict(torch.load(PATH3+'.pt',map_location=device))

        self.model1 = self.model1.module.to(device)
        self.model2 = self.model2.module.to(device)
        self.model3 = self.model3.module.to(device)

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        self.ell = np.arange(2,3001,1)
        self.lmax_theory = 3001

    def predict(self,model,X, extrainfo):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
        Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)

        X_send = np.array([X["omega_b"],X["omega_cdm"],X["H_0"],X["ln10^{10}A_s"],X["n_s"],X["tau_reio"]])

        X = torch.Tensor(X_send).to(device)
        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)

            X_norm.to(device)

            
            pred=model(X_norm)
            
            
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std.float()+Y_mean.float()).cpu().numpy()
            
        return y_pred

    def pcainvtrans(self,y_pred,X):
        X = np.array([X["omega_b"],X["omega_cdm"],X["H_0"],X["ln10^{10}A_s"],X["n_s"],X["tau_reio"]])
        for i in range(len(y_pred)):
            y_pred[i]=y_pred[i]*(np.exp(X[3]))/(np.exp(2*X[5]))
        return y_pred

    def calculate(self, state, want_derived = True, **params):
        cmb_params = { }
        
        for par in self.aliases:
            if par in params:
                cmb_params[par] = [params[par]]
            else:
                for alias in self.aliases[par]:
                    if alias in params:
                        cmb_params[par] = [params[alias]]
                        break

        extrainfo_TT = np.load('./cobaya/cobaya/theories/emutheory/extraaxcambTT.npy', allow_pickle=True)
        extrainfo_TE = np.load('./cobaya/cobaya/theories/emutheory/extraaxcambTE.npy', allow_pickle=True)
        extrainfo_EE = np.load('./cobaya/cobaya/theories/emutheory/extraaxcambEE.npy', allow_pickle=True)

        TT_rescale = self.predict(self.model1, cmb_params, extrainfo_TT)
        TE_rescale = self.predict(self.model2, cmb_params, extrainfo_TE)
        EE_rescale = self.predict(self.model3, cmb_params, extrainfo_EE)

        state["tt"] = self.pcainvtrans(TT_rescale, cmb_params)[0,:]
        state["te"] = self.pcainvtrans(TE_rescale, cmb_params)[0,:]
        state["ee"] = self.pcainvtrans(EE_rescale, cmb_params)[0,:]

        return True

    def get_Cl(self, ell_factor = True, units = "FIRASmuK2"):
        cls_old = self.current_state.copy()
        #print(cls_old)
        
        cls_dict = { k : np.zeros(self.lmax_theory-2) for k in [ "tt", "te", "ee" ] }
        cls_dict["ell"] = self.ell
        
        ls = self.ell
        
        #cmb_fac = self._cmb_unit_factor(units, 2.726)
        
        if ell_factor:
            ls_fac = ls * (ls + 1.0) / (2.0 * np.pi)
        else:
            ls_fac = 1.0
        
        for k in [ "tt", "te", "ee" ]:
            cls_dict[k][ls] = cls_old[k] * ls_fac
        for k in [ "tt", "te", "ee" , "ell"]:
            cls_dict[k] = cls_dict[k][:2508]
        #print(cls_dict)
        
        return cls_dict

    def get_can_support_params(self):
        return [ "omega_b", "omega_cdm", "h", "logA", "ns", "tau_reio" ]
    
    def _cmb_unit_factor(self, units, T_cmb):
        return 1