"""
Helpers to run the pre-trained ParT_full model on JetClass jets and pull out its
attention weights -- with no weaver dependency.

  build_part_inputs(root_file, indices)  -> features (17), vectors (4), mask (1)
  load_part_full(weights_path)           -> model in eval mode (ParticleTransformerWrapper)
  run_attention(model, ...)              -> softmax outputs + per-layer attention

The 17 input features and their preprocessing follow particle_transformer's
data/JetClass/JetClass_full.yaml exactly.
"""
import numpy as np
import torch
import uproot

from part_model import ParticleTransformer


# ---- The 10 JetClass classes, in label order --------------------------------
LABELS = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']


class ParticleTransformerWrapper(torch.nn.Module):
    """Matches the key layout of the saved ParT_*.pt state dicts (`mod.` prefix)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        out = self.mod(features, v=lorentz_vectors, mask=mask)
        return out

    def get_attention(self):
        # list (len = num_layers) of (batch, num_heads, seq_len, seq_len)
        return self.mod.getAttention()

    def get_interaction(self):
        # the physics-informed pair bias U added to the attention logits
        return self.mod.getInteraction()


def get_part_full(**overrides):
    """Build the ParT 'full' configuration (17 input features, 10 classes)."""
    cfg = dict(
        input_dim=17,
        num_classes=10,
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=False,          # keep all 128 slots so particle i stays at index i
        for_inference=False,
    )
    cfg.update(overrides)
    return ParticleTransformerWrapper(**cfg)


def load_part_full(weights_path):
    """Instantiate ParT_full and load the pre-trained weights. Returns eval-mode model."""
    model = get_part_full()
    state = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def _pad(arr, maxlen=128):
    """Zero-pad/truncate a list of variable-length jets to (n_jets, maxlen)."""
    out = np.zeros((len(arr), maxlen), dtype='float32')
    for i, row in enumerate(arr):
        row = np.asarray(row, dtype='float32')[:maxlen]
        out[i, :len(row)] = row
    return out


def build_part_inputs(root_file, indices, maxlen=128):
    """Build the (features, vectors, mask) tensors ParT_full expects, for the
    given jet indices. Mirrors JetClass_full.yaml feature order + preprocessing."""
    t = uproot.open(root_file)['tree']
    branches = [
        'part_px', 'part_py', 'part_pz', 'part_energy', 'part_deta', 'part_dphi',
        'part_charge', 'part_isChargedHadron', 'part_isNeutralHadron',
        'part_isPhoton', 'part_isElectron', 'part_isMuon',
        'part_d0val', 'part_dzval', 'part_d0err', 'part_dzerr',
        'jet_pt', 'jet_energy',
    ]
    a = t.arrays(branches, entry_start=int(min(indices)), entry_stop=int(max(indices)) + 1)
    sel = [int(i) - int(min(indices)) for i in indices]  # local offsets into the slice

    def col(name):
        return [a[name][s] for s in sel]

    px, py, pz, energy = col('part_px'), col('part_py'), col('part_pz'), col('part_energy')
    deta, dphi = col('part_deta'), col('part_dphi')
    jet_pt = np.asarray(a['jet_pt'])[sel]
    jet_energy = np.asarray(a['jet_energy'])[sel]

    feats = []  # one (17, maxlen) block per jet
    vecs = []   # one (4, maxlen) block per jet
    masks = []
    for k in range(len(sel)):
        pxk, pyk = np.asarray(px[k]), np.asarray(py[k])
        ek = np.asarray(energy[k])
        pt = np.hypot(pxk, pyk)
        with np.errstate(divide='ignore', invalid='ignore'):
            pt_log = (np.log(pt) - 1.7) * 0.7
            e_log = (np.log(ek) - 2.0) * 0.7
            logptrel = (np.log(pt / jet_pt[k]) + 4.7) * 0.7
            logerel = (np.log(ek / jet_energy[k]) + 4.7) * 0.7
        detak, dphik = np.asarray(deta[k]), np.asarray(dphi[k])
        deltaR = (np.hypot(detak, dphik) - 0.2) * 4.0
        d0 = np.tanh(np.asarray(col('part_d0val')[k]))
        dz = np.tanh(np.asarray(col('part_dzval')[k]))
        d0err = np.clip(np.asarray(col('part_d0err')[k]), 0, 1)
        dzerr = np.clip(np.asarray(col('part_dzerr')[k]), 0, 1)

        per_jet = [
            pt_log, e_log, logptrel, logerel, deltaR,
            np.asarray(col('part_charge')[k], dtype='float32'),
            np.asarray(col('part_isChargedHadron')[k], dtype='float32'),
            np.asarray(col('part_isNeutralHadron')[k], dtype='float32'),
            np.asarray(col('part_isPhoton')[k], dtype='float32'),
            np.asarray(col('part_isElectron')[k], dtype='float32'),
            np.asarray(col('part_isMuon')[k], dtype='float32'),
            d0, d0err, dz, dzerr, detak, dphik,
        ]
        per_jet = [np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0) for f in per_jet]
        feats.append(_pad(per_jet, maxlen))                       # (17, maxlen)
        vecs.append(_pad([pxk, pyk, pz[k], ek], maxlen))          # (4, maxlen)
        m = np.zeros(maxlen, dtype='float32')
        m[:len(pt)] = 1.0
        masks.append(m)

    features = torch.from_numpy(np.stack(feats, axis=0))          # (N, 17, maxlen)
    vectors = torch.from_numpy(np.stack(vecs, axis=0))            # (N, 4, maxlen)
    mask = torch.from_numpy(np.stack(masks, axis=0))[:, None, :]  # (N, 1, maxlen)
    return features, vectors, mask


def run_attention(model, features, vectors, mask):
    """Run a forward pass and return (probs, attention_list). attention_list is
    one tensor per transformer layer, shape (N, num_heads, seq_len, seq_len)."""
    points = torch.zeros(features.size(0), 2, features.size(-1))  # unused by ParT
    model.mod.attention_matrix = []  # reset (it accumulates across forward calls)
    with torch.no_grad():
        logits = model(points, features, vectors, mask)
        probs = torch.softmax(logits, dim=1)
    attn = [a.detach().clone() for a in model.get_attention()]
    return probs, attn
