# PICOSEC Anomaly Detection

Unsupervised anomaly detection on PICOSEC-Micromegas detector waveforms, using a 1D convolutional autoencoder trained on background-only data (LED off) to isolate single-photoelectron signals (LED on) without event-level labels.

## Project overview

The PICOSEC-Micromegas detector records fast waveforms (10 GS/s, 10002 samples) from single-photoelectron (SPE) measurements, used to characterize the detector's gain and avalanche dynamics. Two datasets are available:

- **LED off**: pure background/noise, ~800k waveforms
- **LED on**: background + a minority of genuine single-photoelectron signals, ~800k waveforms

Since individual LED on events are not labeled as signal or background, a standard supervised approach is not possible. Instead, a convolutional autoencoder is trained **only on the LED off (background) dataset**. The model therefore learns to reconstruct noise waveforms accurately. When applied to the LED on dataset, it fails to reconstruct the genuine photoelectron signals it has never seen during training, producing a measurably higher reconstruction error. This error is used as an **anomaly score** to separate signal-like events from pure background, without requiring any event-level label.

## Repository structure

```
.
├── dataset.py                                    # WaveformDataset class: HDF5 loading + preprocessing
├── model.py                                      # 1D convolutional autoencoder
├── train.py                                      # Training script (CLI)
├── LED_OFF.ipynb                                 # Preliminary exploration of the LED off dataset
├── LED_ON.ipynb                                  # Preliminary exploration of the LED on dataset
├── Valutazione_train_validation_riordinato.ipynb # Model evaluation on the validation set (LED off)
├── Valutazione_train_test_riordinato.ipynb       # Model evaluation on the test set (LED on)
└── Confronto.ipynb                               # Final comparison between the two datasets
```

## Pipeline

**1. Preliminary exploration** (`LED_OFF.ipynb`, `LED_ON.ipynb`)
Raw HDF5 inspection, waveform previews, amplitude/integral/RMS distributions, average waveform per dataset. Also computes and saves the physical features (minimum amplitude, integral) used later for evaluation, and the global mean/std (`mean_norm.npy`, `std_norm.npy`) used for Z-score normalization.

**2. Dataset and preprocessing** (`dataset.py`)
`WaveformDataset` loads the specified HDF5 files, crops each waveform to a 3000-sample window (300 ns) around the trigger, and pre-loads everything into RAM as a single NumPy array. For each requested event, `__getitem__` applies baseline subtraction (mean of the first 500 pre-trigger samples) and Z-score normalization (division by the global background standard deviation), returning a ready-to-use `torch.Tensor`.

**3. Model** (`model.py`)
A 1D convolutional autoencoder (`Autoencoder` class):
- **Encoder**: 3 conv blocks (kernel sizes 15, 7, 3; channels 8, 16, 32), each with BatchNorm, LeakyReLU, average pooling (downsampling by 2) and dropout (p=0.1), followed by a fully connected bottleneck into a `latent_dim`-sized latent space (32 in the final model).
- **Decoder**: mirrors the encoder, using nearest-neighbour upsampling + convolution blocks to reconstruct the original 3000-sample waveform.
- Several `reconstruction_error_*` methods are provided (MAE, MSE, Huber, and a signal-weighted variant) to compute a per-event anomaly score, restricted to a narrow window around the electron peak and the start of the ion tail ([400:1100] samples of the crop), where the difference between background and genuine signal is most pronounced.

**4. Training** (`train.py`)
```bash
python train.py --file_rumore led_off_part01.h5,led_off_part02.h5,... --latent_dim 32 --epoche 100 --batch_size 1024
```
Trains the autoencoder on the LED off dataset only, using:
- MAE (L1) loss over the full 3000-sample waveform
- Adam optimizer (initial lr = 1e-3)
- `ReduceLROnPlateau` scheduler (patience 3, factor 0.5, min lr 1e-6)
- Early stopping (patience 10 epochs)
- A random 90/10 train/validation split (fixed seed), to avoid bias from temporal drifts in the detector conditions during data taking

The best model (lowest validation loss), periodic checkpoints, and the loss history are saved under `checkpoints/`.

**5. Evaluation**
- `Valutazione_train_validation_riordinato.ipynb`: loads the trained model, plots the training/validation loss curve, computes the reconstruction error on the validation set (LED off), and visualizes the latent space via PCA.
- `Valutazione_train_test_riordinato.ipynb`: same evaluation on the test set (LED on). Selects an anomaly score threshold (scanning mean amplitude, selected fraction, and low-charge contamination as a function of the threshold), compares the amplitude/charge distributions of normal vs anomalous events, and fits the resulting spectra with a Polya distribution, expected for single-photoelectron avalanche multiplication.
- `Confronto.ipynb`: final side-by-side comparison of amplitude and charge distributions between the two datasets.

## Requirements

```
torch
h5py
numpy
scipy
matplotlib
scikit-learn
lmfit
tqdm
```

## Notes

- One out of the available LED off files was excluded from training, due to an anomalously high noise level (likely a temporary front-end electronics instability) that the model consistently flagged as anomalous across its entire duration.
- The anomaly score is evaluated on a restricted time window rather than the full waveform: this was found to significantly improve the separation between signal and background populations, since it focuses on the region where the slowly-decaying ion tail — the main feature distinguishing a genuine photoelectron signal from pure noise — is most prominent.
