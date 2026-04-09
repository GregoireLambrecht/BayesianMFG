import argparse
import jax
from utils_bayesian import *
from configs.experiment1 import *


# Assuming your experiment and config definitions are imported or in the same file
# from experiment_module import first_experiment

def main():
    parser = argparse.ArgumentParser(description="Run MFG Bayesian Experiments")
    parser.add_argument('config', type=str, help="config of experiment: 'test', 'small', or 'large'")
    parser.add_argument('eta', type = float)
    parser.add_argument('seed', type=int, default=0, help="Starting seed k")
    
    args = parser.parse_args()
    scale_name = args.config.lower()

    # 1. Select the base configuration
    if scale_name == 'test':
        base_config = TEST.copy()
    elif scale_name == 'small':
        base_config = SMALL.copy()
    elif scale_name == 'large':
        base_config = LARGE.copy()
    else:
        print(f"Error: Scale '{scale_name}' not recognized. Choose from 'test', 'small', 'large'.")
        return

    # 2. Update folder name based on the chosen scale
    base_config['folder_name'] = f"results_exp1_{scale_name}"
    
    eta = args.eta

    print(f"\n--- Starting Sweep for eta = {eta} ---")
    base_config['eta'] = eta
        
    seed = args.seed        
    print(f"Running Seed: {seed}")
        
    # Ensure the config passed has the correct seed
    current_config = base_config.copy()
    seed = args.seed
    current_config['seed'] = seed
        
    first_experiment(current_config, seed)

if __name__ == "__main__":
    main()