#!/usr/bin/python
import subprocess
import numpy as np
from mpi4py import MPI



#if "-f" in sys.argv:
    #idx = sys.argv.index('-f')
#n= int(sys.argv[idx+1])

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_ranks = comm.Get_size()

file_name       = 'testingaxcamb'
cosmology_file  = file_name + '.npy'
output_file     = 'testingaxcamb_output.npy'

camb_accuracy_boost   = 1.5#test 2.5
camb_l_sampple_boost  = 10# test50    # 50 = every ell is computed
camb_ell_min          = 2#30
camb_ell_max          = 3001
camb_ell_range        = camb_ell_max  - camb_ell_min 
camb_num_spectra      = 4

total_num_dvs  = int(2000)

if rank == 0:
    #start=time.time()
    param_info_total = np.load(
        cosmology_file,
        allow_pickle = True
    )
    total_num_dvs = len(param_info_total)

    param_info = param_info_total[0:total_num_dvs:num_ranks]#reading for 0th rank input

    source_file = "../inifiles/params.ini"

    for i in range(1,num_ranks):#sending other ranks' data
        destination_file = "../inifiles/params"+str(i)+".ini"
        command = ["cp", source_file, destination_file]

        subprocess.run(command, capture_output=True, text=True, check=True)
        comm.send(
            param_info_total[i:total_num_dvs:num_ranks], 
            dest = i, 
            tag  = 1
        )
        comm.send(destination_file,
            dest = i,
            tag = 2)
else:
    
    param_info = comm.recv(source = 0, tag = 1)
    source_file = comm.recv(source = 0, tag = 2)
    
num_datavector = len(param_info)

total_cls = np.zeros(
        (num_datavector, camb_ell_range, camb_num_spectra), dtype = "float32"
    ) 

#camb_params = camb.CAMBparams()

for i in range(num_datavector):

    command = ["sed", "-i", "-e", "s/output_root = .*/output_root = test_"+str(rank+i*num_ranks)+"/g",
                            "-e", "s/ombh2 = .*/ombh2 = "+f"{param_info[i,0]:.4e}"+"/g", 
                            "-e", "s/omch2 = .*/omch2 = "+f"{param_info[i,1]:.4e}"+"/g",
                            "-e", "s/hubble = .*/hubble = "+f"{param_info[i,2]:.4e}"+"/g",
                            "-e", "s/scalar_amp(1)             = .*/scalar_amp(1)             = "+f"{np.exp(param_info[i,3])/(1e10):.4e}"+"/g",
                            "-e", "s/scalar_spectral_index(1)  = .*/scalar_spectral_index(1)  = "+f"{param_info[i,4]:.4e}"+"/g",
                            "-e", "s/re_optical_depth     = .*/re_optical_depth     =  "+f"{param_info[i,5]:.4e}"+"/g", 
                            source_file]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    try:
        
        command = ["./camb", source_file]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = np.genfromtxt('test_'+str(rank+i*num_ranks)+'_lensedCls.dat', delimiter='')
        #print(str(rank+i*num_ranks))
    except:
        
        print(str(rank+i*num_ranks)+"Unphysical") # put 1s for all   

    else:

        
        total_cls[i] = data[:,1:]
        


if rank == 0:
    result_cls = np.zeros((total_num_dvs, camb_ell_range, 3), dtype="float32")
    
    result_cls[0:total_num_dvs:num_ranks,:,0] = total_cls[:,:,0] ## TT

    result_cls[0:total_num_dvs:num_ranks,:,1] = total_cls[:,:,3] ## TE
        
    result_cls[0:total_num_dvs:num_ranks,:,2] = total_cls[:,:,1] ## EE
    
    for i in range(1,num_ranks):        
        result_cls[i:total_num_dvs:num_ranks,:,0] = comm.recv(source = i, tag = 10)
        
        result_cls[i:total_num_dvs:num_ranks,:,1] = comm.recv(source = i, tag = 11)
        
        result_cls[i:total_num_dvs:num_ranks,:,2] = comm.recv(source = i, tag = 12)
        

    np.save(output_file, result_cls)
    #command = ["rm", 'test_*.*']
    #subprocess.run(command, text=True, check=True)
    
else:    
    comm.send(total_cls[:,:,0], dest = 0, tag = 10)
    
    comm.send(total_cls[:,:,3], dest = 0, tag = 11)
    
    comm.send(total_cls[:,:,1], dest = 0, tag = 12)
