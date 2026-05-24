import numpy as np                                              

# DNA has 4 bases. We encode each as a position in a length-4 vector.                        
# This is called "one-hot" because exactly one slot is "hot" (=1).
#   A → [1, 0, 0, 0]                                                                         
#   C → [0, 1, 0, 0]                                                                         
#   G → [0, 0, 1, 0]                                                                         
#   T → [0, 0, 0, 1]                                                                         
#   N → [0, 0, 0, 0]  (unknown base)                                                         
                                                                                            
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}                                                
                                                                                            
def one_hot_encode(sequence: str) -> np.ndarray:                                             
    seq = sequence.upper()                                      
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):                                                           
        idx = BASE_INDEX.get(base, -1)
        if idx >= 0:                                                                         
            arr[i, idx] = 1.0                                   
    return arr                                                                               
                                                                
def reverse_complement(sequence: str) -> str:                                                
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return sequence.translate(table)[::-1]