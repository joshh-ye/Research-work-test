"""Build /tmp/yejz1_ft/pooled128 from the trunk cache (read-only on the cache)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import poolcache as pc
info = pc.build(); print(json.dumps(info))
out = Path("/home/yejz1/experimental_research/results_ft/cache_exps"); out.mkdir(parents=True, exist_ok=True)
(out / "pooled128_build_info.json").write_text(json.dumps(info, indent=1))
