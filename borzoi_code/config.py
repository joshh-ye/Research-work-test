import os
import glob
from pathlib import Path

DATA_ROOT = os.environ.get("BORZOI_DATA_ROOT", str(Path(__file__).parent.parent / "borzoi_data"))

FASTA    = os.path.join(DATA_ROOT, "hg38", "hg38.ml.fa")
BW_FILES = sorted(glob.glob(os.path.join(DATA_ROOT, "CRC_TFs_bw", "*.bw")))
assert BW_FILES, f"No BigWigs found under {DATA_ROOT}/CRC_TFs_bw/"
