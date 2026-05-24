import time
import pyBigWig
from numpy import mean
import requests
from dataclasses import dataclass

@dataclass
class car:
    hp: int
    make: str='.'


honda = car(50, 'CIVIC')
print(honda)


# --- .bw data download ---
# url = 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeCrgMapabilityAlign75mer.bigWig'

# with requests.get(url, stream=True) as r:
#     r.raise_for_status()

#     with open("test.bw", "wb") as f:
#         for chunk in r.iter_content(chunk_size=8192):
#             f.write(chunk)