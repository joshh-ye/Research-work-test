import numpy as np
import pyBigWig


class BigWigLoader:
    def __init__(self, bw_paths: list, bin_size: int = 32):
        self.paths = bw_paths
        self.bin_size = bin_size
        self.handles = None   # opened lazily — important for multiprocessing

    def _open(self):
        handles = []
        for p in self.paths:
            try:
                handles.append(pyBigWig.open(p))
            except Exception:
                handles.append(None)  # broken file — skipped in load()
        self.handles = handles

    def load(self, chrom: str, start: int, end: int) -> np.ndarray:
        if self.handles is None:
            self._open()
        n_bins = (end - start) // self.bin_size
        result = np.zeros((n_bins, len(self.paths)), dtype=np.float32)
        for t, bw in enumerate(self.handles):
            if bw is None:
                continue
            try:
                vals = bw.stats(chrom, start, end, type="mean", nBins=n_bins)
                if vals:
                    result[:, t] = [v if v is not None else 0.0 for v in vals]
            except Exception:
                pass   # chrom absent or file error — leave track as zeros
        return result   # shape: (n_bins, n_tracks)

    def close(self):
        if self.handles:
            for bw in self.handles:
                if bw is not None:
                    try:
                        bw.close()
                    except Exception:
                        pass
            self.handles = None
