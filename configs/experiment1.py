TEST = {
    # Metadata & Paths
    'folder_name': 'test_experiment_fast',
    
    # Environment Dimensions
    'NB_STATES': 5,
    'H': 3,
    'eta': 0.05,
    'theta_high': 2.0,      # Max theta for uniform sampling
    
    # Fictitious Play (Bayesian)
    'K_bays': 2,            # Only 2 rounds of FP
    'epochs_fic_bays': 10,  # 10 gradient steps per round
    'batch_size_fic_bays': 8,
    'lr_fic_bays': 1e-3,

    'size_mc' : 100,
    'nb_batch_mc': 1,
    
    # Nash Policy Learning (Expert Compression)
    'epochs_nash_bays': 10,
    'batch_size_nash_bays': 8,
    'lr_nash_bays': 1e-3,
    
    # Normalizing Flow (NLE)
    'epochs_flow': 10,
    'batch_size_flow': 8,
    'lr_flow': 1e-3,
    
    # Standard Fictitious Play (Deterministic Baselines)
    'K': 2,
    'epochs_fic': 10,
    'batch_size_fic': 8,
    'lr_fic': 1e-3,
    
    # Nash Policy Learning (Deterministic)
    'epochs_nash': 10,
    'batch_size_nash': 8,
    'lr_nash': 1e-3
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

SMALL = {
    # Metadata & Paths
    'folder_name': 'exp1_small',
    
    # Environment Dimensions
    'NB_STATES': 7,
    'H': 10,
    'eta': 0.1,
    'theta_low':-0.5,
    'theta_high': 2.5,      # Max theta for uniform sampling
    
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
    'lr_nash_bays': 1e-3,
    
    # Normalizing Flow (NLE)
    'epochs_flow': 20_000,
    'batch_size_flow': 500,
    'lr_flow': 5e-4,
    
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