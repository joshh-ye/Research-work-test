import pyfaidx

class FastaReader:                                                                           
    def __init__(self, fasta_path: str):
        self.fasta = pyfaidx.Fasta(fasta_path, as_raw=True)                                  
                                                                                            
    def fetch(self, chrom: str, start: int, end: int) -> str:                                
        chrom_len = len(self.fasta[chrom])                                                   
        # Clamp to chromosome boundaries, pad with N if window goes over                     
        pad_left  = max(0, -start)                                                           
        pad_right = max(0, end - chrom_len)                                                  
        clamped_start = max(0, start)                                                        
        clamped_end   = min(chrom_len, end)                                                  
        seq = str(self.fasta[chrom][clamped_start:clamped_end]).upper()
        return ("N" * pad_left) + seq + ("N" * pad_right)                                    
                                                                
    def chrom_sizes(self) -> dict:                                                           
        return {k: len(self.fasta[k]) for k in self.fasta.keys()}
                                                                                            
    def close(self):                                            
        self.fasta.close()