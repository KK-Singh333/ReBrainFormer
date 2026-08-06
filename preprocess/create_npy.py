import numpy as np
from preprocess.load_graph import _fmri_extractor,_fmri_extractor_metadata
import torch
import os
from configs.fmri_extractor import fMRIConfig

DATA_DIR=r"E:\ReBrainFormer\ReBrainFormer\test_data"


def _prepare_raw_data(TARGET_DIR:str)->list[str]:
    '''The function takes the path to raw niftii file, converts them to .npy files and returns the list of filenames of npy file'''
    if(not os.path.exists(TARGET_DIR)):
        os.mkdir(TARGET_DIR)
    npyList=[]
    for file in os.listdir(DATA_DIR):
        if file.endswith('.npy'):
            continue
        filepath=os.path.join(DATA_DIR,file)
        try:
            event = _fmri_extractor_metadata(
                    filepath,
                    fMRIConfig,
                )
            
            if event is None:
                raise RuntimeError("Failed to create Fmri event.")
        
            fmri = _fmri_extractor(event)
            npy_timeseries=fmri.numpy()
            print(npy_timeseries)
            target_file = os.path.join(
                    TARGET_DIR,
                    os.path.splitext(os.path.basename(filepath))[0] + ".npy"
                )
            np.save(target_file,npy_timeseries)
            npyList.append(target_file)
        except Exception as e:
            print(f"Error Log:{e}| Skipping file {filepath}")
    return npyList
if __name__=='__main__':
    TARGET_DIR=r"E:\ReBrainFormer\ReBrainFormer\test_data_npy"
    a=_prepare_raw_data(TARGET_DIR)
    print(a)

