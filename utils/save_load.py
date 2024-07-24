import os
import torch
import pickle
from config import cnfg

def load_model(model, name):
    path = os.path.join(cnfg.WEGHTS_DIR, name)
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, name: str) -> None:
    path = os.path.join(cnfg.WEGHTS_DIR, name)
    torch.save(model.state_dict(), path)

def load_pickle(name):
    path = os.path.join(cnfg.WEGHTS_DIR, name)
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(data, name) -> None:
    path = os.path.join(cnfg.WEGHTS_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
