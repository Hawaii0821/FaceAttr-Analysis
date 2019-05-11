from solver import Solver
import os
import random
import numpy as np 
import torch
import pandas as pd 

# ------------------------------------------------ # 
# make sure the same results with same params in different running time.
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#--------------- exe ----------------------------- # 
if __name__ == "__main__":
    seed_everything()

    # too more params to send.... not a good way....use the config.py to improve it
    solver = Solver()
    solver.fit()
