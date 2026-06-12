import torch
import torch.nn as nn
from borzoi_pytorch import Borzoi

# Borzoi's trunk produces a 1920-channel representation per 32 bp bin, which its
# own human/mouse output convolutions consume. We tap that representation as the
# feature for transfer learning instead of the lossy 7611-track human prediction.
EMBED_DIM = 1920


class BorzoiTransferModel(nn.Module):
    def __init__(
        self,
        n_output_tracks: int,
        device: torch.device,
        n_folds: int = 4,
        hidden: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.device = device
        self.n_folds = n_folds

        # Per-fold buffer for the embedding captured by the forward pre-hook below.
        self._embeddings: list = [None] * n_folds

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
            # Capture the 1920-dim trunk embedding that feeds the final human head.
            # The pre-hook fires with the head's input, i.e. the trunk representation,
            # so we never depend on Borzoi's internal method names.
            b.human_head.register_forward_pre_hook(self._make_hook(fold))
            self.backbones.append(b)

        # The only trainable part: an MLP mapping the 1920-dim trunk embedding to
        # the target tracks. Dropout regularizes the head (the backbone is frozen).
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_output_tracks),
            nn.Softplus(),   # coverage is non-negative, Softplus is smooth unlike ReLU
        )
        self.head.to(device)

    def _make_hook(self, fold: int):
        def hook(module, inputs):
            # inputs[0]: (1, EMBED_DIM, n_bins) — the trunk representation
            self._embeddings[fold] = inputs[0]
        return hook

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # sequences: (batch, seq_len, 4)
        outputs = []
        for i in range(sequences.shape[0]):
            # borzoi-pytorch expects (batch, 4, seq_len)
            seq = sequences[i].permute(1, 0).unsqueeze(0).to(self.device)

            embs = []
            with torch.no_grad():
                for fold, b in enumerate(self.backbones):
                    self._embeddings[fold] = None
                    b(seq)  # output discarded; pre-hook captures the embedding
                    e = self._embeddings[fold]      # (1, EMBED_DIM, n_bins)
                    embs.append(e.squeeze(0).permute(1, 0))  # (n_bins, EMBED_DIM)

            avg = torch.stack(embs, dim=0).mean(dim=0)  # average folds → (n_bins, EMBED_DIM)

            # Extract the center bins (Borzoi outputs ~6144, we want center 4096)
            n_bins = avg.shape[0]
            center_bins = 4096
            offset = (n_bins - center_bins) // 2
            avg = avg[offset: offset + center_bins]

            out = self.head(avg)  # (center_bins, n_tracks)
            outputs.append(out)

        return torch.stack(outputs, dim=0)   # (batch, center_bins, n_tracks)
