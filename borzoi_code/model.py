import torch
import torch.nn as nn
from borzoi_pytorch import Borzoi
from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks

class BorzoiTransferModel(nn.Module):
    def __init__(self, n_output_tracks: int, device: torch.device, n_folds: int = 4):
        super().__init__()
        self.device = device
        self.n_folds = n_folds

        # Download pretrained Borzoi from HuggingFace (cached after first download)
        # "johahi/borzoi-replicate-0" is the HF repo ID — owner/repo format
        self.backbones = []
        for fold in range(n_folds):
            repo_id = f"johahi/borzoi-replicate-{fold}"
            print(f"Loading fold {fold} from HuggingFace: {repo_id}")
            b = Borzoi.from_pretrained(repo_id)
            b.to(device)
            b.eval()
            # Freeze all backbone weights — we never train these
            for param in b.parameters():
                param.requires_grad = False
            self.backbones.append(b)

        # The only trainable part: a linear layer mapping 7611 → n_output_tracks
        self.head = nn.Sequential(
            nn.Linear(7611, n_output_tracks),
            nn.Softplus(),   # coverage is non-negative, Softplus is smooth unlike ReLU
        )
        self.head.to(device)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # sequences: (batch, seq_len, 4)
        outputs = []
        for i in range(sequences.shape[0]):
            seq = sequences[i].permute(1, 0)   # (4, seq_len) — borzoi-pytorch expects this

            # predict_tracks returns numpy: (1, n_folds, n_bins, 7611)
            pred = predict_tracks(self.backbones, seq, list(range(7611)))

            avg = pred.mean(axis=1).squeeze(0)  # average folds → (n_bins, 7611)

            # Extract the center bins (Borzoi outputs ~6144, we want center 4096)
            n_bins = avg.shape[0]
            center_bins = 4096
            offset = (n_bins - center_bins) // 2
            avg = avg[offset: offset + center_bins]

            out = self.head(torch.from_numpy(avg).to(self.device))  # (center_bins, n_tracks)
            outputs.append(out)

        return torch.stack(outputs, dim=0)   # (batch, center_bins, n_tracks)
