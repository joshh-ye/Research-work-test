from bigwig_loader import bigWigLoader
from pathlib import Path

curr_dir = Path("./")
bw_paths = list(curr_dir.glob("*.bw"))
bw_paths += list(curr_dir.glob("*.bigwig"))

loader = bigWigLoader(bw_paths)

print(loader.load("chr1", 89294, 91629))