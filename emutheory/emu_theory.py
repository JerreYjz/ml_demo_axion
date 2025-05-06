import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict
from scipy import interpolate
import joblib
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

PATH = os.environ.get("ML_MODEL_DIR")
sys.path.insert(0, PATH)
from emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF


class emutheory(BoltzmannBase):
    aliases: dict = {
        "omega_b" : [ "omegabh2" ],
        "omega_cdm" : [ "omegach2" ],
        "H_0" : [ "H0" ],
        "ln10^{10}A_s" : [ "logA" ],
        "n_s" : [ "ns" ],
        "tau_reio" : [ "tau" ],
        "theta_star": ["thetastar"],
    }

    extra_args: InfoDict = { }
    

    def initialize(self):
        super().initialize()
        
        PATH1 = "./"+PATH+"/"+self.extra_args.get('ttfilename')
        PATH2 = "./"+PATH+"/"+self.extra_args.get('tefilename')
        PATH3 = "./"+PATH+"/"+self.extra_args.get('eefilename')
        PATH4 = "./"+PATH+"/"+self.extra_args.get('dlfilename')
        PATH5 = "./"+PATH+"/"+self.extra_args.get('rsfilename')
        PATH6 = "./"+PATH+"/"+self.extra_args.get('Hfilename')
        PATH7 = "./"+PATH+"/"+self.extra_args.get('GPfilename')

        self.extrainfo_TT = np.load('./'+PATH+'/'+self.extra_args.get('ttextraname'), allow_pickle=True)
        self.extrainfo_TE = np.load('./'+PATH+'/'+self.extra_args.get('teextraname'), allow_pickle=True)
        self.extrainfo_EE = np.load('./'+PATH+'/'+self.extra_args.get('eeextraname'), allow_pickle=True)
        self.extrainfo_dl = np.load('./'+PATH+'/'+self.extra_args.get('dlextraname'), allow_pickle=True)
        self.transmat_dl  = np.load('./'+PATH+'/'+self.extra_args.get('dltransmat'), allow_pickle=True)
        self.extrainfo_rs = np.load('./'+PATH+'/'+self.extra_args.get('rsextraname'), allow_pickle=True)
        self.transmat_rs  = np.load('./'+PATH+'/'+self.extra_args.get('rstransmat'), allow_pickle=True)
        self.extrainfo_H = np.load('./'+PATH+'/'+self.extra_args.get('Hextraname'), allow_pickle=True)
        self.transmat_H  = np.load('./'+PATH+'/'+self.extra_args.get('Htransmat'), allow_pickle=True)
        self.extrainfo_GP = np.load('./'+PATH+'/'+self.extra_args.get('GPextraname'), allow_pickle=True)


        intdim = 4
        nlayer = 4
        nc = 16
        inttrf=5120
        device = 'cpu'
        intdim_simple = 1
        nlayer_simple = 1

        self.model1 = TRF(input_dim=9,output_dim=4998,int_dim=intdim, int_trf=inttrf,N_channels=nc)
        self.model2 = TRF(input_dim=9,output_dim=4998,int_dim=intdim, int_trf=inttrf,N_channels=nc)
        self.model3 = TRF(input_dim=9,output_dim=4998,int_dim=intdim, int_trf=inttrf,N_channels=nc)
        self.model4 = ResMLP(input_dim=2, output_dim=96, int_dim=intdim, N_layer=nlayer)
        self.model5 = ResMLP(input_dim=3, output_dim=96, int_dim=intdim_simple, N_layer=nlayer_simple)
        self.model6 = ResMLP(input_dim=2, output_dim=96, int_dim=intdim_simple, N_layer=nlayer_simple)
        self.model7 = joblib.load(PATH7)
        

        
        self.model1 = self.model1.to(device)
        self.model2 = self.model2.to(device)
        self.model3 = self.model3.to(device)
        self.model4 = self.model4.to(device)
        self.model5 = self.model5.to(device)
        self.model6 = self.model6.to(device)

        self.model1 = nn.DataParallel(self.model1)
        self.model2 = nn.DataParallel(self.model2)
        self.model3 = nn.DataParallel(self.model3)
        self.model4 = nn.DataParallel(self.model4)
        self.model5 = nn.DataParallel(self.model5)
        self.model6 = nn.DataParallel(self.model6)


        self.model1.load_state_dict(torch.load(PATH1+'.pt',map_location=device))
        self.model2.load_state_dict(torch.load(PATH2+'.pt',map_location=device))
        self.model3.load_state_dict(torch.load(PATH3+'.pt',map_location=device))
        self.model4.load_state_dict(torch.load(PATH4+'.pt',map_location=device))
        self.model5.load_state_dict(torch.load(PATH5+'.pt',map_location=device))
        self.model6.load_state_dict(torch.load(PATH6+'.pt',map_location=device))

        self.model1 = self.model1.module.to(device)
        self.model2 = self.model2.module.to(device)
        self.model3 = self.model3.module.to(device)
        self.model4 = self.model4.module.to(device)
        self.model5 = self.model5.module.to(device)
        self.model6 = self.model6.module.to(device)

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model4.eval()
        self.model5.eval()
        self.model6.eval()

        self.ell = np.arange(0,2551,1)
        self.lmax_theory = 2551



    def predict(self,model,X, extrainfo):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
        Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)

        X_send = np.array([X["omega_b"][0],X["omega_cdm"][0],X["H_0"][0],X["tau_reio"][0],X["n_s"][0],X["ln10^{10}A_s"][0],0.06,-1,0])
        

        X = torch.Tensor(X_send).to(device)

        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)
            X_norm[:,6:]=0


            X_norm.to(device)

            
            pred=model(X_norm)
            
            
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std.float()+Y_mean.float()).cpu().numpy()
            
        return y_pred

    def predict_dl(self,model,X, extrainfo,transform_matrix):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=extrainfo.item()['Y_mean']
        Y_std=extrainfo.item()['Y_std']
        Y_mean_2=torch.Tensor(extrainfo.item()['Y_mean_2']).to(device)
        Y_std_2=torch.Tensor(extrainfo.item()['Y_std_2']).to(device)
        

        X_send = np.array([(X["omega_b"][0]+X["omega_cdm"][0])/(X["H_0"][0]/100)**2,X["H_0"][0]])
        

        X = torch.Tensor(X_send).to(device)

        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)



            X_norm.to(device)

            
            pred=model(X_norm)
            
            
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred)-4400
        return y_pred[0]

    def predict_H(self,model,X, extrainfo,transform_matrix):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=extrainfo.item()['Y_mean']
        Y_std=extrainfo.item()['Y_std']
        Y_mean_2=torch.Tensor(extrainfo.item()['Y_mean_2']).to(device)
        Y_std_2=torch.Tensor(extrainfo.item()['Y_std_2']).to(device)
        

        X_send = np.array([(X["omega_b"][0]+X["omega_cdm"][0])/(X["H_0"][0]/100)**2,X["H_0"][0]])
        

        X = torch.Tensor(X_send).to(device)

        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)



            X_norm.to(device)

            
            pred=model(X_norm)
            
            
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred)
        return y_pred[0]

    def predict_rs(self,model,X, extrainfo,transform_matrix):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=extrainfo.item()['Y_mean']
        Y_std=extrainfo.item()['Y_std']
        Y_mean_2=torch.Tensor(extrainfo.item()['Y_mean_2']).to(device)
        Y_std_2=torch.Tensor(extrainfo.item()['Y_std_2']).to(device)
        

        X_send = np.array([X["omega_b"][0],X["omega_cdm"][0],X["H_0"][0]])
        

        X = torch.Tensor(X_send).to(device)

        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)



            X_norm.to(device)

            
            pred=model(X_norm)
            
            
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred)
        return y_pred[0]

    def scaletrans(self,y_pred,X):
        X = np.array([X["omega_b"][0],X["omega_cdm"][0],X["H_0"][0],X["tau_reio"][0],X["n_s"][0],X["ln10^{10}A_s"][0]])
        
        for i in range(len(y_pred)):
            y_pred[i]=y_pred[i]*(np.exp(X[5]))/(np.exp(2*X[3]))
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

        
        if 'H_0' not in cmb_params:
            theta_array = np.array([[cmb_params["omega_b"][0],cmb_params["omega_cdm"][0],cmb_params["theta_star"][0]]])
            theta_array = (theta_array-self.extrainfo_GP.item()['X_mean'])/self.extrainfo_GP.item()['X_std']
            H0 = self.model7.predict(theta_array)[0]*self.extrainfo_GP.item()['Y_std'][0]+self.extrainfo_GP.item()['Y_mean'][0]
            cmb_params["H_0"]=[H0]

        
        TT_rescale = self.predict(self.model1, cmb_params, self.extrainfo_TT)
        TE_rescale = self.predict(self.model2, cmb_params, self.extrainfo_TE)
        EE_rescale = self.predict(self.model3, cmb_params, self.extrainfo_EE)
        dl         = self.predict_dl(self.model4, cmb_params, self.extrainfo_dl, self.transmat_dl)
        rs         = self.predict_rs(self.model5, cmb_params, self.extrainfo_rs, self.transmat_rs)
        H          = self.predict_H(self.model6, cmb_params, self.extrainfo_H, self.transmat_H)

        factor=self.ell*(self.ell+1)/2/np.pi
        state["ell"] = self.ell.astype(int)
        state["tt"] = np.zeros(self.lmax_theory)
        state["te"] = np.zeros(self.lmax_theory)
        state["bb"] = np.zeros(self.lmax_theory)
        state["ee"] = np.zeros(self.lmax_theory)
        state["tt"][2:] = self.scaletrans(TT_rescale, cmb_params)[0,:self.lmax_theory-2]
        state["te"][2:] = self.scaletrans(TE_rescale, cmb_params)[0,:self.lmax_theory-2]
        state["ee"][2:] = self.scaletrans(EE_rescale, cmb_params)[0,:self.lmax_theory-2]
        state["et"] = state["te"]
        
        state["dl"] = dl
        state["rs"] = rs
        state["H"] = H

        return True

    def get_Cl(self, ell_factor = False, units = "1", unit_included = True, Tcmb=2.7255):
        cls_old = self.current_state.copy()
        
        cls_dict = { k : np.zeros(self.lmax_theory) for k in [ "tt", "te", "ee" , "et" , "bb" ] }
        cls_dict["ell"] = self.ell
        
        ls = self.ell
        
        if ell_factor:
            for k in [ "tt", "te", "ee" , "et" , "bb"]:
                ls_fac = self.ell_factor(ls,k)
                cls_dict[k] = cls_old[k] * ls_fac
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb"]:
                cls_dict[k] = cls_old[k]
        if unit_included:
            unit=1
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb" ]:
            
                unit = self.cmb_unit_factor(k, units, Tcmb)
            
                cls_dict[k] = cls_dict[k] * unit
        
       
        
        return cls_dict
        
    def ell_factor(self, ls: np.ndarray, spectra: str) -> np.ndarray:
        """
        Calculate the ell factor for a specific spectrum.
        These prefactors are used to convert from Cell to Dell and vice-versa.

        See also:
        cobaya.BoltzmannBase.get_Cl
        `camb.CAMBresults.get_cmb_power_spectra
        <https://camb.readthedocs.io/en/latest/results.html#camb.results.CAMBdata.get_cmb_power_spectra>`_

        Examples:

        ell_factor(l, "tt") -> :math:`\ell ( \ell + 1 )/(2 \pi)`

        ell_factor(l, "pp") -> :math:`\ell^2 ( \ell + 1 )^2/(2 \pi)`.

        :param ls: the range of ells.
        :param spectra: a two-character string with each character being one of [tebp].

        :return: an array filled with ell factors for the given spectrum.
        """
        ellfac = np.ones_like(ls).astype(float)

        if spectra in ["tt", "te", "tb", "ee", "et", "eb", "bb", "bt", "be"]:
            ellfac = ls * (ls + 1.0) / (2.0 * np.pi)
        elif spectra in ["pt", "pe", "pb", "tp", "ep", "bp"]:
            ellfac = (ls * (ls + 1.0)) ** (3. / 2.) / (2.0 * np.pi)
        elif spectra in ["pp"]:
            ellfac = (ls * (ls + 1.0)) ** 2.0 / (2.0 * np.pi)

        return ellfac

    def cmb_unit_factor(self, spectra: str,
                        units: str = "1",
                        Tcmb: float = 2.7255) -> float:
        """
        Calculate the CMB prefactor for going from dimensionless power spectra to
        CMB units.

        :param spectra: a length 2 string specifying the spectrum for which to
                        calculate the units.
        :param units: a string specifying which units to use.
        :param Tcmb: the used CMB temperature [units of K].
        :return: The CMB unit conversion factor.
        """
        res = 1.0
        
        x, y = spectra.lower()

        if x == "t" or x == "e" or x == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif x == "p":
            res *= 1. / np.sqrt(2.0 * np.pi)

        if y == "t" or y == "e" or y == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif y == "p":
            res *= 1. / np.sqrt(2.0 * np.pi)
        
        return res


    def get_angular_diameter_distance(self,z):
        d_l = self.current_state["dl"].copy()

        z_lin = np.linspace( -0.5, 3, num=2333, endpoint=True)

        d_a = d_l/(1+z_lin)**2

        D_A_interpolate = interpolate.interp1d(z_lin, d_a)

        print(D_A_interpolate(z))

        return D_A_interpolate(z)

    def get_rs(self,z):
        r_s = self.current_state["rs"].copy()

        z_lin = np.linspace( -0.5, 3, num=2333, endpoint=True)

        r_s_interpolate = interpolate.interp1d(z_lin, r_s)

        print(r_s_interpolate(z))

        return r_s_interpolate(z)

    def get_Hubble(self,z):
        H = self.current_state["H"].copy()

        z_lin = np.linspace( -0.5, 3, num=2333, endpoint=True)

        H_interpolate = interpolate.interp1d(z_lin, H)

        print(H_interpolate(z))

        return H_interpolate(z)


    def get_can_support_params(self):
        return [ "omega_b", "omega_cdm", "h", "logA", "ns", "tau_reio" ]
    
