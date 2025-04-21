Guideline for using the codes above:

Step 1: Use axioncamb.py to run parallelized data vector generation for training, validation and testing sets. Plz make sure call the correct CAMB by specifying the dir for the code.
        After running it, to prevent having too many files saved, plz run the following lines in terminal
        rm test_*.*
        rm params_*.ini
Step 2: The code above saves D_ell. Convert D_ell to C_ell.
Step 3: We need to collect some pre-processing info before training. Specically, compute and save the mean and std of the input parameters, and the output power spectra rescaled as
        C'_ell=C_ell/A_s*exp(2tau). Save those in some 'extrainfo' numpy dictionary file.
Step 4: Collect all files including training/validation sets, extrainfo numpy file, the covariance matrix of your liking, and run cmbemuaxcambXY.py to get your model.
