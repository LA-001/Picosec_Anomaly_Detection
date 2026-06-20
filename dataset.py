import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class WaveformDataset(Dataset):
    def __init__(self, file_list, std_norm):
        self.std_norm  = std_norm
        
        all_wfs = []
        print(f"Rilevati {len(file_list)} file HDF5. Inizio pre-caricamento del ritaglio in RAM...")
        
        for filepath in file_list:
            with h5py.File(filepath, 'r') as f:
                # Leggiamo SEQUENZIALMENTE solo la finestra 3500:6500 di tutto il file.
                # L'HDD esegue questa operazione alla massima velocità possibile.
                wf_cropped = f['voltages'][:, 3500:6500].astype(np.float32)
                all_wfs.append(wf_cropped)
                print(f"-> Caricato in RAM: {filepath.name} ({wf_cropped.shape[0]:,} tracce)")
                
        # Uniamo tutti i blocchi in un unico grande array NumPy residente in RAM (~10 GB)
        self.data = np.concatenate(all_wfs, axis=0)
        print(f"Pre-caricamento completato! Dataset pronto: {self.data.shape[0]:,} tracce totali in RAM.")
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        # Lettura istantanea dalla RAM
        wf = self.data[idx]
        
        #Baseline subtraction:
        baseline = wf[:500].mean()
        wf = wf - baseline
        
        #Z-score normalization: 
        wf = wf / (self.std_norm + 1e-8)
        
        return torch.tensor(wf, dtype=torch.float32)