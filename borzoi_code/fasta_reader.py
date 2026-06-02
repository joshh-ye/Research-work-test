import pyfaidx

class FastaReader:
    def __init__(self, fasta_path: str):
        self._path = fasta_path
        self._fasta = None          # opened lazily — fork/pickle safe

    def _open(self):
        self._fasta = pyfaidx.Fasta(self._path, as_raw=True)

    @property
    def _handle(self):
        if self._fasta is None:
            self._open()
        return self._fasta

    def fetch(self, chrom: str, start: int, end: int) -> str:
        fasta = self._handle
        chrom_len = len(fasta[chrom])
        pad_left   = max(0, -start)
        pad_right  = max(0, end - chrom_len)
        clamped_start = max(0, start)
        clamped_end   = min(chrom_len, end)
        seq = str(fasta[chrom][clamped_start:clamped_end]).upper()
        return ("N" * pad_left) + seq + ("N" * pad_right)

    def chrom_sizes(self) -> dict:
        return {k: len(self._handle[k]) for k in self._handle.keys()}

    def close(self):
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
