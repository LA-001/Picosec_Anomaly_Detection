
"""
1D convolutional autoencoder for waveform anomaly detection.
The encoder compresses a 3000-sample waveform into a latent vector,
the decoder reconstructs the waveform from that latent representation.
 """

import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, latent_dim=16):
        super().__init__()      # Gives access to the parent class methods (nn.Module)

        # --- ENCODER ---
        # Three conv blocks: decreasing kernel size, increasing channel depth.
        # Each block: Conv1d -> BatchNorm -> LeakyReLU -> AvgPool (downsample by 2) -> Dropout
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AvgPool1d(2),          # (B, 8, 1500)
            nn.Dropout(p=0.1),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AvgPool1d(2),           # (B, 16, 750)
            nn.Dropout(p=0.1),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AvgPool1d(2),           # (B, 32, 375)
            nn.Dropout(p=0.1),
        )

        # Bottleneck: flatten the (32, 375) feature map and project to the latent space
        self.encoder_fc = nn.Linear(32 * 375, latent_dim)

        # --- DECODER ---
        # Mirror of the encoder bottleneck: latent vector -> (32, 375) feature map
        self.decoder_fc = nn.Linear(latent_dim, 32 * 375)

        # Three upsampling blocks (nearest upsampling + Conv1d) mirroring the encoder.
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),          # (B, 32, 750)
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Upsample(scale_factor=2),          # (B, 16, 1500)
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Upsample(scale_factor=2),          # (B, 8, 3000)
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=15, stride=1, padding=7),
        )

        self.latent_dim = latent_dim

    def encode(self, x):
        # Add the channel dimension if missing: (B, 3000) -> (B, 1, 3000) = (Batch size, channel, samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_conv(x)       # (B, 32, 375)
        h = h.flatten(1)               # (B, 32*375)
        z = self.encoder_fc(h)         # (B, latent_dim)
        return z

    def decode(self, z):
        h = self.decoder_fc(z)         # (B, 32*375)
        h = h.view(-1, 32, 375)        # reshape back to (B, 32, 375)
        x_hat = self.decoder_conv(h)   # (B, 1, 3000)
        return x_hat.squeeze(1)        # drop the channel dim: (B, 1, 3000) -> (B, 3000)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_error_mae(self, x):
        x_hat, _ = self.forward(x)
        return torch.abs(x[:, 400:1100] - x_hat[:, 400:1100]).mean(dim=1)

    def reconstruction_error_mse(self, x):
        x_hat, _ = self.forward(x)
        return torch.square(x[:, 400:1100] - x_hat[:, 400:1100]).mean(dim=1)

    def reconstruction_error_huber(self, x, soglia):
        x_hat, _ = self.forward(x)
        loss = torch.nn.HuberLoss(reduction='none', delta=soglia)
        return loss(x[:, 400:1100], x_hat[:, 400:1100]).mean(dim=1)

    def reconstruction_error_pesata(self, x, std_norm):
        # Weighted reconstruction error: up-weights events with a large negative
        # integrated signal (i.e. likely real photoelectron events), so that the
        # anomaly score becomes even more sensitive to signal-like residuals
        x_hat, _ = self.forward(x)
        alpha = 1

        residuo = (x[:, 400:1100] - x_hat[:, 400:1100]) * float(std_norm)
        segnale = (x[:, 400:1100] * float(std_norm)).sum(dim=1)

        peso = torch.exp(torch.clamp(-alpha * segnale, max=10.0))

        return peso * residuo.abs().mean(dim=1)


if __name__ == "__main__":
    # Quick sanity check: verify input/output shapes and count model parameters
    modello = Autoencoder(latent_dim=16)
    wf_finte = torch.randn(8, 3000)
    ricostruite, z = modello(wf_finte)
    print(f"Input:          {tuple(wf_finte.shape)}")
    print(f"Ricostruzione:  {tuple(ricostruite.shape)}")
    print(f"Spazio latente: {tuple(z.shape)}")
    n_params = sum(p.numel() for p in modello.parameters() if p.requires_grad)
    print(f"Parametri totali: {n_params:,}")