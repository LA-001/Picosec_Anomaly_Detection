"""
Use: python train.py --file_rumore led_off.h5 --latent_dim 32 --epoche 100
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import WaveformDataset
from model import Autoencoder
from tqdm import tqdm
import numpy as np


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso: {device}")

    std_norm = np.load('checkpoints/std_norm.npy')      # Global std used for Z-score normalization

    # --- Load the background (LED off) data: 90% train / 10% validation ---
    file_rumore = [Path(p) for p in args.file_rumore.split(",")]
    dataset = WaveformDataset(file_rumore, std_norm)         # Dataset object; __getitem__ returns a normalized torch.tensor for each event
    print(f"Waveform totali (LED off): {len(dataset):,}")

    n_val   = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val

    ds_train, ds_val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))     # random_split only creates two index subsets (no data is copied)     
    print(f"Training: {n_train:,}  |  Validazione: {n_val:,}")

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)      # shuffle=True only for training, validation order doesn't matter
    loader_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Build the model ---
    modello = Autoencoder(latent_dim=args.latent_dim).to(device)         # Move all model weights to GPU if cuda is avaible
    n_param = sum(p.numel() for p in modello.parameters() if p.requires_grad)       # Count trainable parameters
    print(f"Parametri del modello: {n_param:,}  |  latent_dim={args.latent_dim}")

    patience_counter = 0
    patience = 10
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ottimizzatore, patience=3, factor=0.5, min_lr=1e-6, verbose=True)

    criterio = nn.L1Loss()

    # --- Training loop ---
    cartella = Path(args.cartella_output)
    cartella.mkdir(parents=True, exist_ok=True)
    storia = {"train": [], "val": []}
    best_val = float("inf")

    for epoca in range(1, args.epoche + 1):

        # --- Training step ---
        modello.train()
        loss_train = 0.0
        for wf in tqdm(loader_train, desc=f"Epoca {epoca}/{args.epoche} train"):         # It's like a for loop over loader_train, but with a visual progress bar
            wf = wf.to(device)
            ricostruita, _ = modello(wf)        # Forward pass; discard the latent vector here
            loss = criterio(ricostruita, wf)    # Already averaged over samples and over the batch
            ottimizzatore.zero_grad()           # Reset gradients from the previous step
            loss.backward()                     # Backpropagate
            ottimizzatore.step()                # Update the model weights
            loss_train += loss.item() * len(wf)
        loss_train /= n_train

        # --- Validation step ---
        modello.eval()
        loss_val = 0.0
        with torch.no_grad():
            for wf in tqdm(loader_val, desc="val"):
                wf = wf.to(device)
                ricostruita, _ = modello(wf)
                loss = criterio(ricostruita, wf)
                loss_val += loss.item() * len(wf)
        loss_val /= n_val

        scheduler.step(loss_val)        # Step for LR reducer
        storia["train"].append(loss_train)
        storia["val"].append(loss_val)

        print(f"Epoca {epoca:3d}/{args.epoche}  "
              f"train={loss_train:.6e}  val={loss_val:.6e}  "
              f"lr={ottimizzatore.param_groups[0]['lr']:.1e}")

        # --- Save the best model so far (lowest validation loss), and reset the early-stopping counter ---
        if loss_val < best_val:
            best_val = loss_val
            patience_counter = 0
            torch.save({
                "epoca": epoca,
                "latent_dim": args.latent_dim,
                "pesi": modello.state_dict(),
                "val_loss": loss_val,
            }, cartella / f"modello_migliore_latent{args.latent_dim}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping a epoca {epoca}")
                break

        # --- Periodic checkpoint every 10 epochs ---
        if epoca % 10 == 0:
            torch.save({
                "epoca": epoca,
                "latent_dim": args.latent_dim,
                "pesi": modello.state_dict(),
                "val_loss": loss_val,
            }, cartella / f"checkpoint_epoca{epoca}_latent{args.latent_dim}.pt")
            print(f"  Checkpoint salvato a epoca {epoca}")

        # --- Save the loss history after every epoch ---
        with open(cartella / f"storia_latent{args.latent_dim}.json", "w") as f:
            json.dump(storia, f)

    print(f"\nFine training. Miglior val loss: {best_val:.5e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_rumore",      type=str, required=True, help="File .h5 LED off, separati da virgola se più di uno")
    parser.add_argument("--latent_dim",       type=int, default=32)
    parser.add_argument("--epoche",           type=int, default=100)
    parser.add_argument("--batch_size",       type=int, default=256)
    parser.add_argument("--cartella_output",  type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)