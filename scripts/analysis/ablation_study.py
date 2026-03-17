#!/usr/bin/env python3
"""
Ablation Study: Cross-Validated Comparison of Model Variants.

Compares the full VAE + uncertainty-aware framework against ablated variants
to quantify the contribution of each architectural component.

Model Variants:
    1. FULL        — VAE latent space + uncertainty-aware loss + temporal context
    2. DIRECT      — MLP from raw inputs to emissions (no latent space)
    3. PCA         — PCA dimensionality reduction + predictor
    4. AE          — Deterministic autoencoder (no KL, no sampling)
    5. VAE_MSE     — VAE latent space + vanilla MSE loss (no uncertainty)
    6. NO_TEMPORAL — VAE + uncertainty but predictor sees only z_t (no z_{t-1})

All variants predict ABSOLUTE (scaled) emissions y_t, not deltas.

Usage:
    python -m scripts.analysis.ablation_study

Outputs:
    - outputs/tables/ablation_results.csv
    - outputs/tables/ablation_summary.csv
"""

from __future__ import annotations

import pickle
import random
import traceback
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
VAE_CONFIG_PATH = Path("config/models/vae_config.yaml")
PRED_CONFIG_PATH = Path("config/models/co2_predictor_config.yaml")
OUTPUT_TABLE_DIR = Path("outputs/tables")

ABLATION_EPOCHS = 1500
BATCH_SIZE = 128
PATIENCE = 200

CV_TEST_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]
VARIANT_NAMES = ["FULL", "DIRECT", "PCA", "AE", "VAE_MSE", "NO_TEMPORAL"]


# =============================================================================
# Utility
# =============================================================================


def load_config(yaml_path):
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return SimpleNamespace(
        **{
            k: v["value"]
            for k, v in raw.items()
            if not k.startswith("_") and k != "wandb_version"
        }
    )


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_cv_splits(dataset, test_year):
    train_idx, test_idx = [], []
    for i in range(len(dataset)):
        (test_idx if dataset.keys.iloc[i, 1] == test_year else train_idx).append(i)
    return train_idx, test_idx


# =============================================================================
# Building Blocks
# =============================================================================


class ResBlock(nn.Module):
    def __init__(self, w, d, act, drop=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            *[l for _ in range(d - 1) for l in (nn.Linear(w, w), act, nn.Dropout(drop))]
        )

    def forward(self, x):
        return x + self.layers(x)


# =============================================================================
# PCA
# =============================================================================


class PCAReducer:
    def __init__(self, n=10):
        self.n = n

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: self.n]
        return self


# =============================================================================
# Deterministic Autoencoder
# =============================================================================


class DetEncoder(nn.Module):
    def __init__(self, idim, ldim, nb=5, db=6, drop=0.33):
        super().__init__()
        self.latent_dim = ldim
        step = max(1, (idim - ldim) // nb)
        ws = [idim - i * step for i in range(nb)] + [ldim]
        act = nn.GELU()
        blocks = [nn.Dropout(drop)]
        for i in range(len(ws) - 1):
            blocks += [nn.Linear(ws[i], ws[i + 1]), ResBlock(ws[i + 1], db, act, drop)]
        self.net = nn.Sequential(*blocks)
        self.out = nn.Linear(ldim, ldim)

    def forward(self, x):
        return self.out(self.net(x))


class DetDecoder(nn.Module):
    def __init__(self, idim, ldim, nb=5, db=6, drop=0.33):
        super().__init__()
        step = max(1, (idim - ldim) // nb)
        ws = ([idim - i * step for i in range(nb)] + [ldim])[::-1]
        act = nn.GELU()
        blocks = []
        for i in range(len(ws) - 1):
            blocks += [nn.Linear(ws[i], ws[i + 1]), ResBlock(ws[i + 1], db, act, drop)]
        self.net = nn.Sequential(*blocks)
        self.out = nn.Linear(idim, idim)

    def forward(self, z):
        return self.out(self.net(z))


class AEModel(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder, self.decoder = enc, dec

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# =============================================================================
# VAE
# =============================================================================


class VAEEnc(nn.Module):
    def __init__(self, idim, ldim, nb=5, db=6, drop=0.33, idrop=0.4):
        super().__init__()
        self.latent_dim = ldim
        step = max(1, (idim - ldim) // nb)
        ws = [idim - i * step for i in range(nb)] + [ldim]
        act = nn.GELU()
        blocks = [nn.Dropout(idrop)]
        for i in range(len(ws) - 1):
            blocks += [nn.Linear(ws[i], ws[i + 1]), ResBlock(ws[i + 1], db, act, drop)]
        self.net = nn.Sequential(*blocks)
        self.mean = nn.Linear(ldim, ldim)
        self.log_var = nn.Linear(ldim, ldim)

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.log_var(h)


class VAEDec(nn.Module):
    def __init__(self, idim, ldim, nb=5, db=6, drop=0.33):
        super().__init__()
        step = max(1, (idim - ldim) // nb)
        ws = ([idim - i * step for i in range(nb)] + [ldim])[::-1]
        act = nn.GELU()
        blocks = []
        for i in range(len(ws) - 1):
            blocks += [nn.Linear(ws[i], ws[i + 1]), ResBlock(ws[i + 1], db, act, drop)]
        self.net = nn.Sequential(*blocks)
        self.out = nn.Linear(idim, idim)

    def forward(self, z):
        return self.out(self.net(z))


class VAEModel(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder, self.decoder = enc, dec

    def forward(self, x):
        m, lv = self.encoder(x)
        z = m + torch.exp(0.5 * lv) * torch.randn_like(m)
        return self.decoder(z), m, lv, z


# =============================================================================
# Predictors
# =============================================================================


class PredUnc(nn.Module):
    def __init__(self, idim, odim=6, h=128, nb=2, db=2, drop=0.09):
        super().__init__()
        act = nn.SiLU()
        blocks = [nn.Linear(idim, h)]
        for _ in range(nb):
            blocks.append(ResBlock(h, db, act, drop))
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(h, odim)
        self.unc = nn.Linear(h, odim)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h), self.unc(h)


class PredPlain(nn.Module):
    def __init__(self, idim, odim=6, h=128, nb=2, db=2, drop=0.09):
        super().__init__()
        act = nn.SiLU()
        blocks = [nn.Linear(idim, h)]
        for _ in range(nb):
            blocks.append(ResBlock(h, db, act, drop))
        blocks.append(nn.Linear(h, odim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Losses
# =============================================================================


def unc_mse(tgt, pred, log_u):
    sq = torch.clamp((tgt - pred).pow(2), 0, 2.0)
    return (0.5 * torch.exp(-log_u) * sq + 0.005 * log_u).mean()


def vae_elbo(x, xh, m, lv):
    rec = torch.clamp(nn.functional.l1_loss(xh, x, reduction="none"), 0, 5.0).mean()
    kl = -0.5 * torch.mean(1 + lv - m.pow(2) - lv.exp())
    return 0.95 * rec + 0.05 * kl


# =============================================================================
# Shared Training Helpers
# =============================================================================


def _train_vae(ds, tr, te, cfg):
    idim = ds.input_df.shape[1]
    ld = cfg.vae_latent_dim
    enc = VAEEnc(
        idim,
        ld,
        cfg.vae_num_blocks,
        cfg.vae_dim_blocks,
        cfg.vae_dropouts,
        cfg.vae_input_dropouts,
    ).to(DEVICE)
    dec = VAEDec(idim, ld, cfg.vae_num_blocks, cfg.vae_dim_blocks, cfg.vae_dropouts).to(
        DEVICE
    )
    vae = VAEModel(enc, dec).to(DEVICE)
    opt = torch.optim.AdamW(
        vae.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_weight_decay, eps=1e-6
    )
    tl = DataLoader(Subset(ds, tr), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(Subset(ds, te), batch_size=BATCH_SIZE)
    best, bst, pat = float("inf"), None, 0
    for _ in range(ABLATION_EPOCHS):
        vae.train()
        for b in tl:
            x = b[0].to(DEVICE)
            opt.zero_grad()
            xh, m, lv, z = vae(x)
            vae_elbo(x, xh, m, lv).backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
        vae.eval()
        vv = 0.0
        n = 0
        with torch.no_grad():
            for b in vl:
                x = b[0].to(DEVICE)
                xh, m, lv, z = vae(x)
                vv += vae_elbo(x, xh, m, lv).item()
                n += 1
        vv /= max(n, 1)
        if vv < best:
            best, bst, pat = vv, deepcopy(vae.state_dict()), 0
        else:
            pat += 1
            if pat > PATIENCE:
                break
    vae.load_state_dict(bst)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def _train_pred(enc_fn, ds, tr, te, pidim, pc, use_unc=True):
    """Train predictor to predict absolute emissions y_t."""
    if use_unc:
        pred = PredUnc(
            pidim,
            6,
            pc.pred_width_block,
            pc.pred_num_blocks,
            pc.pred_dim_block,
            pc.pred_dropouts,
        ).to(DEVICE)
    else:
        pred = PredPlain(
            pidim,
            6,
            pc.pred_width_block,
            pc.pred_num_blocks,
            pc.pred_dim_block,
            pc.pred_dropouts,
        ).to(DEVICE)
    opt = torch.optim.Adam(
        pred.parameters(), lr=pc.pred_lr, weight_decay=pc.pred_wd, eps=1e-6
    )
    tl = DataLoader(Subset(ds, tr), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(Subset(ds, te), batch_size=BATCH_SIZE)
    best, bst, pat = float("inf"), None, 0
    for _ in range(ABLATION_EPOCHS):
        pred.train()
        for batch in tl:
            bd = [b.to(DEVICE) for b in batch]
            y = bd[2]
            opt.zero_grad()
            with torch.no_grad():
                inp = enc_fn(bd)
            out = pred(inp)
            if use_unc:
                p, u = out
                loss = unc_mse(y, p, u)
            else:
                loss = nn.functional.mse_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            opt.step()
        pred.eval()
        vv = 0.0
        n = 0
        with torch.no_grad():
            for batch in vl:
                bd = [b.to(DEVICE) for b in batch]
                y = bd[2]
                inp = enc_fn(bd)
                out = pred(inp)
                p = out[0] if isinstance(out, tuple) else out
                vv += (y - p).pow(2).mean().item()
                n += 1
        vv /= max(n, 1)
        if vv < best:
            best, bst, pat = vv, deepcopy(pred.state_dict()), 0
        else:
            pat += 1
            if pat > PATIENCE:
                break
    pred.load_state_dict(bst)
    pred.eval()
    return pred


# =============================================================================
# Variant Trainers
# =============================================================================


def train_FULL(ds, tr, te, vc, pc):
    ld, cd = vc.vae_latent_dim, ds.context_df.shape[1]
    vae = _train_vae(ds, tr, te, vc)

    def enc(bd):
        x, c, y, x1, c1 = bd
        m, lv = vae.encoder(x)
        z = m + torch.exp(0.5 * lv) * torch.randn_like(m)
        m1, lv1 = vae.encoder(x1)
        z1 = m1 + torch.exp(0.5 * lv1) * torch.randn_like(m1)
        return torch.cat([z, c, z1, c1], 1)

    return vae, _train_pred(enc, ds, tr, te, 2 * (ld + cd), pc, True), "full"


def train_DIRECT(ds, tr, te, vc, pc):
    idim, cd = ds.input_df.shape[1], ds.context_df.shape[1]

    def enc(bd):
        x, c, y, x1, c1 = bd
        return torch.cat([x, c, x1, c1], 1)

    return None, _train_pred(enc, ds, tr, te, 2 * (idim + cd), pc, True), "direct"


def train_PCA(ds, tr, te, vc, pc):
    ld, cd = vc.vae_latent_dim, ds.context_df.shape[1]
    pca = PCAReducer(ld).fit(ds.input_df[tr].cpu().numpy())
    pm = torch.tensor(pca.mean_, dtype=torch.float32).to(DEVICE)
    pcomp = torch.tensor(pca.components_.T, dtype=torch.float32).to(DEVICE)

    def enc(bd):
        x, c, y, x1, c1 = bd
        return torch.cat([(x - pm) @ pcomp, c, (x1 - pm) @ pcomp, c1], 1)

    return pca, _train_pred(enc, ds, tr, te, 2 * (ld + cd), pc, True), "pca"


def train_AE(ds, tr, te, vc, pc):
    idim, ld, cd = ds.input_df.shape[1], vc.vae_latent_dim, ds.context_df.shape[1]
    ae_e = DetEncoder(
        idim, ld, vc.vae_num_blocks, vc.vae_dim_blocks, vc.vae_dropouts
    ).to(DEVICE)
    ae_d = DetDecoder(
        idim, ld, vc.vae_num_blocks, vc.vae_dim_blocks, vc.vae_dropouts
    ).to(DEVICE)
    ae = AEModel(ae_e, ae_d).to(DEVICE)
    opt = torch.optim.AdamW(
        ae.parameters(), lr=vc.vae_lr, weight_decay=vc.vae_weight_decay, eps=1e-6
    )
    tl = DataLoader(Subset(ds, tr), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(Subset(ds, te), batch_size=BATCH_SIZE)
    best, bst, pat = float("inf"), None, 0
    for _ in range(ABLATION_EPOCHS):
        ae.train()
        for b in tl:
            x = b[0].to(DEVICE)
            opt.zero_grad()
            xh, z = ae(x)
            torch.clamp(
                nn.functional.l1_loss(xh, x, reduction="none"), 0, 5.0
            ).mean().backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
        ae.eval()
        vv = 0.0
        n = 0
        with torch.no_grad():
            for b in vl:
                x = b[0].to(DEVICE)
                xh, z = ae(x)
                vv += nn.functional.l1_loss(xh, x).item()
                n += 1
        vv /= max(n, 1)
        if vv < best:
            best, bst, pat = vv, deepcopy(ae.state_dict()), 0
        else:
            pat += 1
            if pat > PATIENCE:
                break
    ae.load_state_dict(bst)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    def enc(bd):
        x, c, y, x1, c1 = bd
        return torch.cat([ae.encoder(x), c, ae.encoder(x1), c1], 1)

    return ae, _train_pred(enc, ds, tr, te, 2 * (ld + cd), pc, True), "ae"


def train_VAE_MSE(ds, tr, te, vc, pc):
    ld, cd = vc.vae_latent_dim, ds.context_df.shape[1]
    vae = _train_vae(ds, tr, te, vc)

    def enc(bd):
        x, c, y, x1, c1 = bd
        m, _ = vae.encoder(x)
        m1, _ = vae.encoder(x1)
        return torch.cat([m, c, m1, c1], 1)

    return vae, _train_pred(enc, ds, tr, te, 2 * (ld + cd), pc, False), "vae_mse"


def train_NO_TEMPORAL(ds, tr, te, vc, pc):
    ld, cd = vc.vae_latent_dim, ds.context_df.shape[1]
    vae = _train_vae(ds, tr, te, vc)

    def enc(bd):
        x, c, y, x1, c1 = bd
        m, lv = vae.encoder(x)
        z = m + torch.exp(0.5 * lv) * torch.randn_like(m)
        return torch.cat([z, c], 1)

    return vae, _train_pred(enc, ds, tr, te, ld + cd, pc, True), "no_temporal"


TRAINERS = {
    "FULL": train_FULL,
    "DIRECT": train_DIRECT,
    "PCA": train_PCA,
    "AE": train_AE,
    "VAE_MSE": train_VAE_MSE,
    "NO_TEMPORAL": train_NO_TEMPORAL,
}


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(tag, enc_red, pred, ds, te_idx, vc):
    """Evaluate on held-out year. Everything moved to CPU for numpy at the end."""
    ld = vc.vae_latent_dim
    # PCA GPU tensors
    pm, pcomp = None, None
    if tag == "pca" and enc_red is not None:
        pm = torch.tensor(enc_red.mean_, dtype=torch.float32).to(DEVICE)
        pcomp = torch.tensor(enc_red.components_.T, dtype=torch.float32).to(DEVICE)

    all_p, all_t = [], []
    with torch.no_grad():
        for idx in te_idx:
            x = ds.input_df[idx].unsqueeze(0).to(DEVICE)
            c = ds.context_df[idx].unsqueeze(0).to(DEVICE)
            y = ds.emi_df[idx].cpu().numpy()  # target always on CPU

            geo, yr = ds.keys.iloc[idx, 0], ds.keys.iloc[idx, 1]
            pi = ds.index_map.get((geo, yr - 1))
            if pi is not None:
                x1 = ds.input_df[pi].unsqueeze(0).to(DEVICE)
                c1 = ds.context_df[pi].unsqueeze(0).to(DEVICE)
            else:
                x1, c1 = x, c

            if tag == "direct":
                inp = torch.cat([x, c, x1, c1], 1)
            elif tag == "pca":
                inp = torch.cat([(x - pm) @ pcomp, c, (x1 - pm) @ pcomp, c1], 1)
            elif tag == "ae":
                inp = torch.cat([enc_red.encoder(x), c, enc_red.encoder(x1), c1], 1)
            elif tag == "no_temporal":
                m, _ = enc_red.encoder(x)
                inp = torch.cat([m, c], 1)
            elif tag in ("full", "vae_mse"):
                m, _ = enc_red.encoder(x)
                m1, _ = enc_red.encoder(x1)
                inp = torch.cat([m, c, m1, c1], 1)
            else:
                raise ValueError(tag)

            out = pred(inp)
            p = (out[0] if isinstance(out, tuple) else out).squeeze(0).cpu().numpy()
            all_p.append(p)
            all_t.append(y)

    preds, tgts = np.array(all_p), np.array(all_t)
    res = {}
    for s, sec in enumerate(SECTORS):
        p, t = preds[:, s], tgts[:, s]
        res[f"{sec}_MSE"] = float(np.mean((p - t) ** 2))
        res[f"{sec}_MAE"] = float(np.mean(np.abs(p - t)))
        ssr = np.sum((t - p) ** 2)
        sst = np.sum((t - t.mean()) ** 2)
        res[f"{sec}_R2"] = float(1 - ssr / (sst + 1e-10))
    res["Overall_MSE"] = float(np.mean((preds - tgts) ** 2))
    res["Overall_MAE"] = float(np.mean(np.abs(preds - tgts)))
    ssr = np.sum((tgts - preds) ** 2)
    sst = np.sum((tgts - tgts.mean(0)) ** 2)
    res["Overall_R2"] = float(1 - ssr / (sst + 1e-10))
    return res


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("ABLATION STUDY: CROSS-VALIDATED MODEL COMPARISON")
    print("=" * 70)
    print(f"Device: {DEVICE}, Variants: {VARIANT_NAMES}")
    print(
        f"CV folds: {CV_TEST_YEARS}, Epochs: {ABLATION_EPOCHS}, Patience: {PATIENCE}\n"
    )

    ds = load_dataset(DATASET_PATH)
    ds.input_df = ds.input_df.to(DEVICE)
    ds.context_df = ds.context_df.to(DEVICE)
    ds.emi_df = ds.emi_df.to(DEVICE)
    vc = load_config(VAE_CONFIG_PATH)
    pc = load_config(PRED_CONFIG_PATH)
    print(
        f"Samples: {len(ds)}, Input: {ds.input_df.shape[1]}, "
        f"Context: {ds.context_df.shape[1]}, Latent: {vc.vae_latent_dim}\n"
    )

    results = []
    for fi, ty in enumerate(CV_TEST_YEARS):
        print(
            f"\n{'=' * 60}\nFOLD {fi + 1}/{len(CV_TEST_YEARS)}: year={ty}\n{'=' * 60}"
        )
        tr, te = get_cv_splits(ds, ty)
        print(f"  Train: {len(tr)}, Test: {len(te)}")
        if not te:
            continue

        for vn in VARIANT_NAMES:
            print(f"\n  --- {vn} ---")
            torch.manual_seed(SEED + fi * 100)
            random.seed(SEED + fi * 100)
            np.random.seed(SEED + fi * 100)
            try:
                er, pr, tag = TRAINERS[vn](ds, tr, te, vc, pc)
                m = evaluate(tag, er, pr, ds, te, vc)
                results.append({"variant": vn, "test_year": ty, "fold": fi, **m})
                print(
                    f"    MSE={m['Overall_MSE']:.6f} MAE={m['Overall_MAE']:.6f} R²={m['Overall_R2']:.4f}"
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
            try:
                del er, pr
            except:
                pass
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if not results:
        print("\nNo results.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_TABLE_DIR / "ablation_results.csv", index=False)

    rows = []
    for v in VARIANT_NAMES:
        vd = df[df["variant"] == v]
        if len(vd) == 0:
            continue
        r = {"variant": v, "n_folds": len(vd)}
        for m in ["Overall_MSE", "Overall_MAE", "Overall_R2"]:
            r[f"{m}_mean"], r[f"{m}_std"] = vd[m].mean(), vd[m].std()
        for s in SECTORS:
            for sf in ["MSE", "MAE", "R2"]:
                c = f"{s}_{sf}"
                r[f"{c}_mean"], r[f"{c}_std"] = vd[c].mean(), vd[c].std()
        rows.append(r)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(OUTPUT_TABLE_DIR / "ablation_summary.csv", index=False)

    print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
    print(f"{'Variant':<15} {'MSE':>20} {'MAE':>20} {'R²':>16}")
    print("-" * 75)
    for _, r in sdf.iterrows():
        print(
            f"{r['variant']:<15} {r['Overall_MSE_mean']:.6f}±{r['Overall_MSE_std']:.4f}  "
            f"{r['Overall_MAE_mean']:.6f}±{r['Overall_MAE_std']:.4f}  "
            f"{r['Overall_R2_mean']:.4f}±{r['Overall_R2_std']:.4f}"
        )
    print(f"\nSaved: {OUTPUT_TABLE_DIR}/ablation_results.csv")
    print(f"Saved: {OUTPUT_TABLE_DIR}/ablation_summary.csv\nDone!")


if __name__ == "__main__":
    main()
