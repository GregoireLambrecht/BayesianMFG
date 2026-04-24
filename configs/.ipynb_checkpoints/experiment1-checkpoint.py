TEST = {
    # Metadata & Paths
    'folder_name': 'exp1_test',
    
    # Environment Dimensions
    'NB_STATES': 7,
    'H': 20,
    'eta': 0.1,
    'theta_low': 0,
    'theta_high': 2.5,

    # Fictitious Play (Bayesian)
    'K_bays': 2,
    'epochs_fic_bays': 100,
    'batch_size_fic_bays': 16,
    'lr_fic_bays': 5e-4,

    'size_mc': 100,
    'nb_batch_mc': 10,

    # Nash Policy Learning (Expert Compression)
    'epochs_nash_bays': 100,
    'batch_size_nash_bays': 16,
    'lr_nash_bays': 1e-3,

    # Normalizing Flow (NLE)
    'epochs_flow': 100,
    'batch_size_flow': 16,
    'lr_flow': 1e-3,

    # Standard Fictitious Play (Deterministic Baselines)
    'K': 2,
    'epochs_fic': 100,
    'batch_size_fic': 16,
    'lr_fic': 5e-4,

    # Nash Policy Learning (Deterministic)
    'epochs_nash': 100,
    'batch_size_nash': 16,
    'lr_nash': 5e-4
}


# SMALL = {
#     # Metadata & Paths
#     'folder_name': 'exp1_small',
    
#     # Environment Dimensions
#     'NB_STATES': 7,
#     'H': 10,
#     'eta': 0.1,
#     'theta_high': 2.0,      # Max theta for uniform sampling
    
#     # Fictitious Play (Bayesian)
#     'K_bays': 15,            # Only 2 rounds of FP
#     'epochs_fic_bays': 30000,  # 10 gradient steps per round
#     'batch_size_fic_bays': 16,
#     'lr_fic_bays': 1e-4,

#     'size_mc' : 10000,
#     'nb_batch_mc': 100,
    
#     # Nash Policy Learning (Expert Compression)
#     'epochs_nash_bays': 100_000,
#     'batch_size_nash_bays': 200,
#     'lr_nash_bays': 5e-4,
    
#     # Normalizing Flow (NLE)
#     'epochs_flow': 100_000,
#     'batch_size_flow': 500,
#     'lr_flow': 5e-4,
    
#     # Standard Fictitious Play (Deterministic Baselines)
#     'K': 15,
#     'epochs_fic': 2000,
#     'batch_size_fic': 200,
#     'lr_fic': 1e-4,
    
#     # Nash Policy Learning (Deterministic)
#     'epochs_nash': 50_000,
#     'batch_size_nash': 200,
#     'lr_nash': 5e-3
# }

# SMALL = {
#     # Metadata & Paths
#     'folder_name': 'exp1_small',
    
#     # Environment Dimensions
#     'NB_STATES': 7,
#     'H': 10,
#     'eta': 0.1,
#     'theta_low':-0.5,
#     'theta_high': 2.5,      # Max theta for uniform sampling
    
#     # Fictitious Play (Bayesian)
#     'K_bays': 20,            
#     'epochs_fic_bays': 15000,  
#     'batch_size_fic_bays': 1024,
#     'lr_fic_bays': 5e-4,

#     'size_mc' : 10000,
#     'nb_batch_mc': 100,
    
#     # Nash Policy Learning (Expert Compression)
#     'epochs_nash_bays': 100_000,
#     'batch_size_nash_bays': 200,
#     'lr_nash_bays': 1e-3,
    
#     # Normalizing Flow (NLE)
#     'epochs_flow': 20_000,
#     'batch_size_flow': 500,
#     'lr_flow': 5e-4,
    
#     # Standard Fictitious Play (Deterministic Baselines)
#     'K': 20,
#     'epochs_fic': 10000,
#     'batch_size_fic': 1024,
#     'lr_fic': 5e-4,
    
#     # Nash Policy Learning (Deterministic)
#     'epochs_nash': 100_000,
#     'batch_size_nash': 200,
#     'lr_nash': 1e-4
# }


SMALL = {
    # Metadata & Paths
    'folder_name': 'exp1_small',
    
    # Environment Dimensions
    'NB_STATES': 7,
    'H': 20,
    'eta': 0.1,
    'theta_low':0,
    'theta_high': 2.5,      # Max theta for uniform sampling
    
    # Fictitious Play (Bayesian)
    'K_bays': 15,            
    'epochs_fic_bays': 3000,  
    'batch_size_fic_bays': 512,
    'lr_fic_bays': 5e-4,

    'size_mc' : 10000,
    'nb_batch_mc': 10,
    
    # Nash Policy Learning (Expert Compression)
    'epochs_nash_bays': 50_000,
    'batch_size_nash_bays': 128,
    'lr_nash_bays': 1e-3,
    
    # Normalizing Flow (NLE)
    'epochs_flow': 25_000,
    'batch_size_flow': 256,
    'lr_flow': 1e-5,
    
    # Standard Fictitious Play (Deterministic Baselines)
    'K': 10,
    'epochs_fic': 3000,
    'batch_size_fic': 128,
    'lr_fic': 5e-4,
    
    # Nash Policy Learning (Deterministic)
    'epochs_nash': 50_000,
    'batch_size_nash': 128,
    'lr_nash': 1e-4
}



LARGE = {
    # Metadata & Paths
    'folder_name': 'exp1_small',
    
    # Environment Dimensions
    'NB_STATES': 21,
    'H': 50,
    'eta': 0.1,
    'theta_high': 2.5,      # Max theta for uniform sampling
    'theta_low': -0.5,
    
    # Fictitious Play (Bayesian)
    'K_bays': 20,            # Only 2 rounds of FP
    'epochs_fic_bays': 30000,  # 10 gradient steps per round
    'batch_size_fic_bays': 500,
    'lr_fic_bays': 1e-4,

    'size_mc' : 10000,
    'nb_batch_mc': 100,
    
    # Nash Policy Learning (Expert Compression)
    'epochs_nash_bays': 100_000,
    'batch_size_nash_bays': 200,
    'lr_nash_bays': 5e-4,
    
    # Normalizing Flow (NLE)
    'epochs_flow': 100_000,
    'batch_size_flow': 500,
    'lr_flow': 1e-3,
    
    # Standard Fictitious Play (Deterministic Baselines)
    'K': 15,
    'epochs_fic': 8000,
    'batch_size_fic': 500,
    'lr_fic': 1e-4,
    
    # Nash Policy Learning (Deterministic)
    'epochs_nash': 100_000,
    'batch_size_nash': 200,
    'lr_nash': 1e-4
}