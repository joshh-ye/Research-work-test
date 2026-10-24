"""Model-side tests: staged forward == Borzoi forward (fp32, exact), LoRA identity at init + module compatibility, trainable accounting."""
import sys
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path("/home/yejz1/experimental_research")
sys.path.insert(0, str(ROOT / "analysis/ft"))
import model_ft as M
import data_ft as D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
tiles = D.load_tiles()


def codes(gid=120):
    return D.TileJobs(tiles, []).seq_codes(gid, False, 0)[None]


def cfg(**k):
    return dict(mode="frozen", **k)


@pytest.fixture(scope="module")
def ref():
    t = M.load_trunk().cuda().eval()
    x = torch.nn.functional.one_hot(torch.from_numpy(codes()).long().cuda(), 5)[..., :4].permute(0, 2, 1).float().contiguous()
    with torch.no_grad():
        # reference = pre-hook tap of the original forward (borzoi_code/model.py): re-create with human_head
        from borzoi_pytorch import Borzoi
        full = Borzoi.from_pretrained("johahi/borzoi-replicate-0").cuda().eval()
        cap = {}
        full.human_head.register_forward_pre_hook(lambda m, i: cap.__setitem__("e", i[0].detach().clone()))
        full(x)
    del full
    return cap["e"]


def test_staged_forward_equals_original_tap_fp32(ref):
    m = M.FTModel(cfg()).cuda().eval()
    x = torch.nn.functional.one_hot(torch.from_numpy(codes()).long().cuda(), 5)[..., :4].permute(0, 2, 1).float().contiguous()
    with torch.no_grad():
        e = m.trunk_embedding(x)
    assert e.shape == ref.shape == (1, 1920, 6144)
    assert torch.equal(e, ref), float((e - ref).abs().max())


def test_bf16_close_to_fp32(ref):
    m = M.FTModel(cfg()).cuda().eval()
    with torch.no_grad():
        p32 = m(torch.from_numpy(codes()).cuda(), autocast=False)
        p16 = m(torch.from_numpy(codes()).cuda(), autocast=True)
    a, b = p32.flatten().cpu().numpy(), p16.flatten().cpu().numpy()
    assert np.corrcoef(a, b)[0, 1] > 0.99


def test_lora_identity_and_param_count():
    base = M.FTModel(cfg()).cuda().eval()
    torch.manual_seed(0)
    lo = M.FTModel(dict(mode="lora", lora_targets="attn,mlp,conv1x1", lora_r=4, lora_blocks=8)).cuda().eval()
    lo.head.load_state_dict(base.head.state_dict())
    c = codes()
    with torch.no_grad():
        a, b = base(torch.from_numpy(c).cuda(), autocast=False), lo(torch.from_numpy(c).cuda(), autocast=False)
    assert torch.allclose(a, b, atol=1e-6)
    names = [n for n, p in lo.named_parameters() if p.requires_grad and not n.startswith("head.")]
    assert names and all("lora_" in n for n in names)
    assert len(lo.lora_modules) == 8 * 6 + 5          # 8 blocks x (q,k,v,out,ffn1,ffn2) + 5 1x1 convs
    assert lo.first_train_stage == 0 or lo.first_train_stage == 1


def test_partial_unfreeze_sets():
    m = M.FTModel(dict(mode="partial", unfreeze="blocks:2,final"))
    tr = {n.split(".")[2] if n.startswith("trunk.transformer") else n.split(".")[1] for n, p in m.named_parameters() if p.requires_grad and n.startswith("trunk")}
    assert tr == {"6", "7", "final_joined_convs"}
    assert m.first_train_stage == 8
    assert not m.trunk.training


def test_full_trains_everything_and_trunk_eval():
    m = M.FTModel(dict(mode="full")).train()
    assert m.first_train_stage == 0 and not m.trunk.training and m.head.training
    assert sum(p.numel() for p in m.trunk.parameters()) == 171_271_968
