#!/usr/bin/env python3
"""
VAE CONDICIONAL - PIXEL ART v2.0

Melhorias sobre v1.0:
- Gradientes KL reescritos com beta_kl como parametro
- Split treino/validacao e best-checkpoint
- Edge loss (preserva outlines) + palette loss
- KL warm-up linear
- Versionamento de arquitetura (checkpoints antigos invalidados)
- Seed deterministico e mensagens de erro claras
"""
import os, json, time, argparse, gc
from pathlib import Path
import numpy as np
from PIL import Image

import config as C

MODEL_DIR = Path("models")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
SAMPLES_DIR = MODEL_DIR / "samples"
MANIFEST_FILE = MODEL_DIR / "manifest.json"
TRAINING_LOG = MODEL_DIR / "training_log.json"


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0; self.m, self.v = {}, {}
    def update(self, name, param, grad):
        if name not in self.m:
            self.m[name] = np.zeros_like(param); self.v[name] = np.zeros_like(param)
        self.t += 1
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def tanh(x): return np.tanh(x)
def tanh_grad(x): return 1.0 - np.tanh(x) ** 2
def relu(x): return np.maximum(0.0, x)
def relu_grad(x): return (x > 0).astype(np.float32)


SOBEL_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
SOBEL_Y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)


def sobel_edges(patch_flat):
    bs = patch_flat.shape[0]
    img = patch_flat.reshape(bs, C.PATCH_SIZE, C.PATCH_SIZE, 3)
    pad = np.pad(img, ((0,0),(1,1),(1,1),(0,0)), mode='reflect')
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    for c in range(3):
        gx[:,:,:,c] = (pad[:,:-2,:-2,c]*SOBEL_X[0,0] + pad[:,:-2,1:-1,c]*SOBEL_X[0,1] + pad[:,:-2,2:,c]*SOBEL_X[0,2] +
                       pad[:,1:-1,:-2,c]*SOBEL_X[1,0] + pad[:,1:-1,1:-1,c]*SOBEL_X[1,1] + pad[:,1:-1,2:,c]*SOBEL_X[1,2] +
                       pad[:,2:,:-2,c]*SOBEL_X[2,0] + pad[:,2:,1:-1,c]*SOBEL_X[2,1] + pad[:,2:,2:,c]*SOBEL_X[2,2])
        gy[:,:,:,c] = (pad[:,:-2,:-2,c]*SOBEL_Y[0,0] + pad[:,:-2,1:-1,c]*SOBEL_Y[0,1] + pad[:,:-2,2:,c]*SOBEL_Y[0,2] +
                       pad[:,1:-1,:-2,c]*SOBEL_Y[1,0] + pad[:,1:-1,1:-1,c]*SOBEL_Y[1,1] + pad[:,1:-1,2:,c]*SOBEL_Y[1,2] +
                       pad[:,2:,:-2,c]*SOBEL_Y[2,0] + pad[:,2:,1:-1,c]*SOBEL_Y[2,1] + pad[:,2:,2:,c]*SOBEL_Y[2,2])
    mag = np.sqrt(gx**2 + gy**2 + 1e-8)
    return mag.reshape(bs, -1)


def edge_loss(pred, target):
    e_pred = sobel_edges(pred)
    e_targ = sobel_edges(target)
    return float(np.mean(np.abs(e_pred - e_targ)))


def palette_loss(pred, palette_rgb):
    bs = pred.shape[0]
    img = pred.reshape(bs, -1, 3)
    pal = palette_rgb[None, None, :, :].astype(np.float32)
    dist = ((img[:, :, None, :] - pal) ** 2).sum(axis=-1)
    min_dist = dist.min(axis=-1)
    return float(np.mean(min_dist))


def kl_weight_schedule(epoch):
    return C.BETA_KL * min(1.0, epoch / max(1, C.BETA_KL_WARMUP))


class ConditionalVAE:
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        s_enc = np.sqrt(2.0 / (C.INPUT_DIM + C.INPUT_DIM + C.HIDDEN_DIM))
        s_lat = np.sqrt(2.0 / (C.HIDDEN_DIM + C.LATENT_DIM))
        s_dec = np.sqrt(2.0 / (C.LATENT_DIM + C.HIDDEN_DIM))

        self.W_enc1 = (rng.randn(C.INPUT_DIM + C.INPUT_DIM, C.HIDDEN_DIM) * s_enc).astype(np.float32)
        self.b_enc1 = np.zeros(C.HIDDEN_DIM, dtype=np.float32)
        self.W_mu = (rng.randn(C.HIDDEN_DIM, C.LATENT_DIM) * s_lat).astype(np.float32)
        self.b_mu = np.zeros(C.LATENT_DIM, dtype=np.float32)
        self.W_logvar = (rng.randn(C.HIDDEN_DIM, C.LATENT_DIM) * s_lat).astype(np.float32)
        self.b_logvar = np.zeros(C.LATENT_DIM, dtype=np.float32)
        self.W_dec1 = (rng.randn(C.LATENT_DIM, C.HIDDEN_DIM) * s_dec).astype(np.float32)
        self.b_dec1 = np.zeros(C.HIDDEN_DIM, dtype=np.float32)
        self.W_dec2 = (rng.randn(C.HIDDEN_DIM, C.OUTPUT_DIM) * s_dec).astype(np.float32)
        self.b_dec2 = np.zeros(C.OUTPUT_DIM, dtype=np.float32)
        self.W_cond = (rng.randn(C.CONDITION_DIM, C.INPUT_DIM) * 0.01).astype(np.float32)
        self.opt = Adam(lr=C.LEARNING_RATE)

    def encode(self, x, cond):
        cond_proj = cond @ self.W_cond
        x_cond = np.concatenate([x, cond_proj], axis=-1)
        h1_pre = x_cond @ self.W_enc1 + self.b_enc1
        h1 = tanh(h1_pre)
        mu = h1 @ self.W_mu + self.b_mu
        log_var = h1 @ self.W_logvar + self.b_logvar
        log_var = np.clip(log_var, -8.0, 8.0)
        cache = {'x_cond': x_cond, 'h1_pre': h1_pre, 'h1': h1, 'mu': mu, 'log_var': log_var}
        return mu, log_var, cache

    def reparameterize(self, mu, log_var, rng=None):
        std = np.exp(0.5 * log_var)
        eps = (np.random.randn(*mu.shape) if rng is None
               else rng.randn(*mu.shape)).astype(np.float32)
        z = mu + std * eps
        return z, eps, std

    def decode(self, z):
        h1_pre = z @ self.W_dec1 + self.b_dec1
        h1 = relu(h1_pre)
        out_pre = h1 @ self.W_dec2 + self.b_dec2
        out = tanh(out_pre)
        cache = {'z': z, 'h1_pre': h1_pre, 'h1': h1, 'out_pre': out_pre, 'out': out}
        return out, cache

    def forward(self, x, cond):
        mu, log_var, enc_cache = self.encode(x, cond)
        z, eps, std = self.reparameterize(mu, log_var)
        out, dec_cache = self.decode(z)
        return out, mu, log_var, enc_cache, dec_cache, z, eps, std

    def backward(self, x, cond, target, enc_cache, dec_cache, z, eps, std, beta_kl):
        bs = x.shape[0]
        d_out = 2.0 * (dec_cache['out'] - target) / bs
        d_out_pre = d_out * tanh_grad(dec_cache['out_pre'])
        d_W_dec2 = dec_cache['h1'].T @ d_out_pre
        d_b_dec2 = d_out_pre.sum(axis=0)
        d_h1 = d_out_pre @ self.W_dec2.T
        d_h1_pre = d_h1 * relu_grad(dec_cache['h1_pre'])
        d_W_dec1 = dec_cache['z'].T @ d_h1_pre
        d_b_dec1 = d_h1_pre.sum(axis=0)
        d_z = d_h1_pre @ self.W_dec1.T

        d_mu_recon = d_z
        d_logvar_recon = d_z * 0.5 * std * eps

        mu = enc_cache['mu']
        log_var = enc_cache['log_var']
        d_mu_kl = beta_kl * mu / bs
        d_logvar_kl = beta_kl * 0.5 * (np.exp(log_var) - 1.0) / bs

        d_mu = d_mu_recon + d_mu_kl
        d_logvar = d_logvar_recon + d_logvar_kl
        d_logvar = np.clip(d_logvar, -1.0, 1.0)

        d_W_mu = enc_cache['h1'].T @ d_mu
        d_b_mu = d_mu.sum(axis=0)
        d_W_logvar = enc_cache['h1'].T @ d_logvar
        d_b_logvar = d_logvar.sum(axis=0)
        d_h1_from_latent = d_mu @ self.W_mu.T + d_logvar @ self.W_logvar.T
        d_h1_pre_e = d_h1_from_latent * tanh_grad(enc_cache['h1_pre'])
        d_x_cond = d_h1_pre_e @ self.W_enc1.T
        d_W_enc1 = enc_cache['x_cond'].T @ d_h1_pre_e
        d_b_enc1 = d_h1_pre_e.sum(axis=0)
        d_cond_proj = d_x_cond[:, C.INPUT_DIM:]
        d_W_cond = cond.T @ d_cond_proj

        return {
            'W_enc1': d_W_enc1, 'b_enc1': d_b_enc1,
            'W_mu': d_W_mu, 'b_mu': d_b_mu,
            'W_logvar': d_W_logvar, 'b_logvar': d_b_logvar,
            'W_dec1': d_W_dec1, 'b_dec1': d_b_dec1,
            'W_dec2': d_W_dec2, 'b_dec2': d_b_dec2,
            'W_cond': d_W_cond,
        }

    def train_step(self, x, cond, target, beta_kl, palette_rgb):
        out, mu, log_var, enc, dec, z, eps, std = self.forward(x, cond)
        recon = float(np.mean((out - target) ** 2))
        edge = edge_loss(out, target)
        pl = palette_loss(out, palette_rgb)
        kl = float(-0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var)))
        total = recon + C.BETA_EDGE * edge + C.BETA_PALETTE * pl + beta_kl * kl
        grads = self.backward(x, cond, target, enc, dec, z, eps, std, beta_kl)
        for name, g in grads.items():
            setattr(self, name, self.opt.update(name, getattr(self, name), g))
        return total, recon, kl, edge, pl

    def eval_loss(self, x, cond, target, beta_kl, palette_rgb):
        out, mu, log_var, *_ = self.forward(x, cond)
        recon = float(np.mean((out - target) ** 2))
        edge = edge_loss(out, target)
        pl = palette_loss(out, palette_rgb)
        kl = float(-0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var)))
        total = recon + C.BETA_EDGE * edge + C.BETA_PALETTE * pl + beta_kl * kl
        return total, recon, kl

    def save(self, path, epoch, loss, val_loss=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = dict(
            W_enc1=self.W_enc1, b_enc1=self.b_enc1,
            W_mu=self.W_mu, b_mu=self.b_mu,
            W_logvar=self.W_logvar, b_logvar=self.b_logvar,
            W_dec1=self.W_dec1, b_dec1=self.b_dec1,
            W_dec2=self.W_dec2, b_dec2=self.b_dec2,
            W_cond=self.W_cond,
            epoch=np.array(epoch), loss=np.array(loss),
            timestamp=np.array(time.time()),
            arch_version=np.array(C.ARCH_VERSION, dtype='S'),
        )
        if val_loss is not None:
            data['val_loss'] = np.array(val_loss)
        np.savez_compressed(path, **data)

    def load(self, path):
        d = np.load(path, allow_pickle=False)
        required = ['W_enc1','b_enc1','W_mu','b_mu','W_logvar','b_logvar',
                    'W_dec1','b_dec1','W_dec2','b_dec2','W_cond']
        missing = [k for k in required if k not in d.files]
        if missing:
            raise RuntimeError(
                f"Checkpoint {path.name} corrompido. Faltando: {missing}\n"
                "Reinicie o treino sem --resume."
            )
        arch = bytes(d['arch_version']).decode('utf-8') if 'arch_version' in d.files else '1.0'
        if arch != C.ARCH_VERSION:
            raise RuntimeError(
                f"Checkpoint tem arch_version={arch}, codigo atual e {C.ARCH_VERSION}.\n"
                "Arquitetura mudou - necessario treinar novamente.\n"
                "Remova models/checkpoints/ ou rode sem --resume."
            )
        for k in required:
            setattr(self, k, d[k])
        return int(d['epoch']), float(d['loss'])


def build_condition(pal_name, ptype_idx, rng):
    cond = np.zeros(C.CONDITION_DIM, dtype=np.float32)
    pal_idx = C.PALETTE_NAMES.index(pal_name)
    cond[C.COND_SLICE_PAL][pal_idx] = 1.0
    cond[C.COND_SLICE_TYPE][ptype_idx] = 1.0
    mood_map = {"dark": "dark", "neon": "neon", "nature": "nature",
                "nes": "retro", "gameboy": "retro", "cga": "retro"}
    mood = mood_map.get(pal_name, "retro")
    cond[C.COND_SLICE_MOOD][C.MOOD_TO_IDX[mood]] = 1.0
    if ptype_idx in (0, 1):
        cond[C.COND_SLICE_TRAIT][C.TRAIT_TO_IDX["symmetric"]] = 1.0
        cond[C.COND_SLICE_TRAIT][C.TRAIT_TO_IDX["outline"]] = 1.0
    elif ptype_idx == 9:
        cond[C.COND_SLICE_TRAIT][C.TRAIT_TO_IDX["outline"]] = 1.0
    elif ptype_idx == 3:
        cond[C.COND_SLICE_TRAIT][C.TRAIT_TO_IDX["wet"]] = 1.0
    elif ptype_idx == 6:
        cond[C.COND_SLICE_TRAIT][C.TRAIT_TO_IDX["glow"]] = 1.0
    feat_rng = np.random.RandomState(rng.randint(0, 2**31))
    cond[C.COND_SLICE_FEAT] = feat_rng.randn(C.COND_SLICE_FEAT.stop - C.COND_SLICE_FEAT.start).astype(np.float32) * 0.1
    return cond


def generate_realistic_patches(n_samples=8000, seed=123):
    print("📦 Gerando dataset REALISTA de pixel art...")
    rng = np.random.RandomState(seed)
    X, Y, Cond = [], [], []
    for i in range(n_samples):
        pal_name = C.PALETTE_NAMES[i % len(C.PALETTE_NAMES)]
        pal = C.TRAIN_PALETTES_RGB[pal_name]
        patch = np.full((C.PATCH_SIZE, C.PATCH_SIZE, 3), pal[0], dtype=np.float32)
        ptype_idx = rng.randint(0, len(C.PIXEL_ART_TYPES))
        bg = pal[0]; fg = pal[rng.randint(1, len(pal))]

        if ptype_idx == 0:
            half_w = C.PATCH_SIZE // 2
            for y in range(C.PATCH_SIZE):
                for x in range(half_w):
                    if rng.random() < 0.45:
                        patch[y, x] = fg
                        patch[y, C.PATCH_SIZE - 1 - x] = fg
        elif ptype_idx == 1:
            head_y, head_x = 1, C.PATCH_SIZE // 2
            hs = 3
            for dy in range(hs):
                for dx in range(-hs//2, hs//2+1):
                    y, x = head_y + dy, head_x + dx
                    if 0 <= y < C.PATCH_SIZE and 0 <= x < C.PATCH_SIZE:
                        patch[y, x] = pal[rng.randint(1, min(4, len(pal)))]
            bc = pal[rng.randint(1, min(4, len(pal)))]
            for dy in range(hs, C.PATCH_SIZE - 1):
                for dx in range(-1, 2):
                    y, x = head_y + dy, head_x + dx
                    if 0 <= y < C.PATCH_SIZE and 0 <= x < C.PATCH_SIZE:
                        patch[y, x] = bc
        elif ptype_idx == 2:
            for y in range(C.PATCH_SIZE):
                for x in range(C.PATCH_SIZE):
                    patch[y, x] = pal[rng.choice([0,1,2]) if len(pal) > 2 else 0] if rng.random() < 0.7 else bg
        elif ptype_idx == 3:
            for y in range(C.PATCH_SIZE):
                for x in range(C.PATCH_SIZE):
                    wave = int((np.sin(x*0.8 + y*0.3)+1)*1.5) % min(3, len(pal))
                    patch[y, x] = pal[wave]
        elif ptype_idx == 4:
            bh = 3
            for y in range(C.PATCH_SIZE):
                for x in range(C.PATCH_SIZE):
                    row = y // bh
                    offset = (row % 2) * (C.PATCH_SIZE // 2)
                    col = (x + offset) % (C.PATCH_SIZE // 2)
                    if y % bh == 0 or col == 0:
                        patch[y, x] = pal[min(3, len(pal)-1)]
                    else:
                        patch[y, x] = pal[rng.randint(0, min(3, len(pal)))]
        elif ptype_idx == 5:
            cx = C.PATCH_SIZE // 2
            for y in range(1, C.PATCH_SIZE - 3):
                patch[y, cx] = pal[min(5, len(pal)-1)]
            patch[C.PATCH_SIZE-3, cx-1:cx+2] = pal[min(6, len(pal)-1)]
            patch[C.PATCH_SIZE-2:C.PATCH_SIZE, cx] = pal[min(3, len(pal)-1)]
        elif ptype_idx == 6:
            cx, cy = C.PATCH_SIZE // 2, C.PATCH_SIZE // 2
            for y in range(C.PATCH_SIZE):
                for x in range(C.PATCH_SIZE):
                    d = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if d < 2.5: patch[y, x] = pal[rng.randint(1, min(4, len(pal)))]
                    elif d < 3.2: patch[y, x] = pal[min(5, len(pal)-1)]
        elif ptype_idx == 7:
            cx = C.PATCH_SIZE // 2
            for y in range(C.PATCH_SIZE//2, C.PATCH_SIZE):
                patch[y, cx] = pal[min(3, len(pal)-1)]
            cy = C.PATCH_SIZE // 3
            for y in range(C.PATCH_SIZE//2):
                for x in range(C.PATCH_SIZE):
                    d = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if d < 3 and rng.random() < 0.7:
                        patch[y, x] = pal[rng.choice([0,1,2]) if len(pal) > 2 else 0]
        elif ptype_idx == 8:
            patch[:] = pal[0]
            cx = rng.randint(2, C.PATCH_SIZE-2); cy = rng.randint(1, C.PATCH_SIZE-2)
            cloud = pal[min(5, len(pal)-1)]
            for _ in range(15):
                y, x = cy + rng.randint(-1, 2), cx + rng.randint(-2, 3)
                if 0 <= y < C.PATCH_SIZE and 0 <= x < C.PATCH_SIZE:
                    patch[y, x] = cloud
        elif ptype_idx == 9:
            mask = np.zeros((C.PATCH_SIZE, C.PATCH_SIZE), dtype=bool)
            cy, cx = C.PATCH_SIZE//2, C.PATCH_SIZE//2
            for _ in range(20):
                if 0 <= cy < C.PATCH_SIZE and 0 <= cx < C.PATCH_SIZE:
                    mask[cy, cx] = True
                cy += rng.randint(-1, 2); cx += rng.randint(-1, 2)
            for y in range(C.PATCH_SIZE):
                for x in range(C.PATCH_SIZE):
                    if mask[max(0,y-1):y+2, max(0,x-1):x+2].sum() >= 2:
                        mask[y, x] = True
            outline_color = pal[0]
            inner_color = pal[rng.randint(1, min(4, len(pal)))]
            for y in range(C.PATCH_SIZE):
                for x in range(C.PATCH_SIZE):
                    if mask[y, x]:
                        is_border = False
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < C.PATCH_SIZE and 0 <= nx < C.PATCH_SIZE:
                                if not mask[ny, nx]:
                                    is_border = True; break
                        patch[y, x] = outline_color if is_border else inner_color
        elif ptype_idx == 10:
            for y in range(C.PATCH_SIZE):
                t = y / (C.PATCH_SIZE - 1)
                c1 = pal[rng.randint(0, min(3, len(pal)))]
                c2 = pal[rng.randint(0, min(3, len(pal)))]
                patch[y, :] = c1 * (1 - t) + c2 * t
        else:
            cx, cy = C.PATCH_SIZE/2, C.PATCH_SIZE/2
            shape = rng.choice(['circle', 'diamond', 'triangle'])
            for y in range(C.PATCH_SIZE):
                for x in range(C.PATCH_SIZE):
                    if shape == 'circle':
                        d = np.sqrt((x-cx)**2 + (y-cy)**2)
                        if d < 3: patch[y, x] = fg
                    elif shape == 'diamond':
                        if abs(x-cx) + abs(y-cy) < 3: patch[y, x] = fg
                    else:
                        if y > cy and abs(x-cx) < (y-cy)*0.8:
                            patch[y, x] = fg

        patch_norm = patch * 2.0 - 1.0
        hist = np.histogram(patch.reshape(-1, 3), bins=16, range=(0, 1))[0]
        hist = hist / (hist.sum() + 1e-8)
        stats = np.array([patch.mean(), patch.std(), patch.min(), patch.max(),
                          np.unique(patch.reshape(-1,3),axis=0).shape[0]/64.0])
        feat = np.concatenate([hist, stats]).astype(np.float32)
        if len(feat) < C.INPUT_DIM: feat = np.pad(feat, (0, C.INPUT_DIM - len(feat)))
        cond = build_condition(pal_name, ptype_idx, rng)
        X.append(feat[:C.INPUT_DIM])
        Y.append(patch_norm.reshape(-1))
        Cond.append(cond)
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            np.array(Cond, dtype=np.float32))


def render_sample(model, out_path):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    z = np.random.randn(1, C.LATENT_DIM).astype(np.float32)
    out, _ = model.decode(z)
    patch = ((out.reshape(C.PATCH_SIZE, C.PATCH_SIZE, 3) + 1.0) / 2.0 * 255).astype(np.uint8)
    Image.fromarray(patch).save(out_path)


def train(epochs=5000, batch_size=64, save_every=500, seed=42, resume=False):
    print("=" * 70)
    print("🧠 TREINAMENTO VAE CONDICIONAL v" + C.ARCH_VERSION + " - PIXEL ART")
    print("=" * 70)
    print(f"   LATENT: {C.LATENT_DIM}d | BETA_KL max: {C.BETA_KL}")
    print(f"   Warm-up: {C.BETA_KL_WARMUP} epocas | Edge b: {C.BETA_EDGE} | Pal b: {C.BETA_PALETTE}")
    print(f"   Val split: {C.VAL_SPLIT*100:.0f}%")
    print("=" * 70)

    np.random.seed(seed)
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    X, Y, Cond = generate_realistic_patches(8000, seed=123)
    print(f"   ✓ Dataset: X{X.shape} Y{Y.shape} Cond{Cond.shape}")
    idx = np.random.permutation(len(X))
    n_val = max(1, int(len(X) * C.VAL_SPLIT))
    val_idx = idx[:n_val]; train_idx = idx[n_val:]
    X_tr, Y_tr, C_tr = X[train_idx], Y[train_idx], Cond[train_idx]
    X_va, Y_va, C_va = X[val_idx], Y[val_idx], Cond[val_idx]
    print(f"   ✓ Train: {len(X_tr)} | Val: {len(X_va)}")

    all_pal = np.concatenate([v for v in C.TRAIN_PALETTES_RGB.values()], axis=0)

    model = ConditionalVAE(seed=seed)
    total_params = sum(p.size for p in [
        model.W_enc1, model.b_enc1, model.W_mu, model.b_mu,
        model.W_logvar, model.b_logvar,
        model.W_dec1, model.b_dec1, model.W_dec2, model.b_dec2, model.W_cond])
    print(f"🧮 Parametros: {total_params:,}")
    print(f"🎯 Epocas: {epochs} | Batch: {batch_size}")

    start_epoch = 1
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [], 'train_edge': [], 'train_pal': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [], 'beta_kl': [],
    }

    if resume:
        latest = CHECKPOINT_DIR / "latest.npz"
        if latest.exists():
            try:
                start_epoch, _ = model.load(latest)
                start_epoch += 1
                print(f"📂 Retomando do epoch {start_epoch - 1}")
            except RuntimeError as e:
                print(f"❌ {e}")
                print("   Abortando. Rode sem --resume para treinar do zero.")
                return

    for epoch in range(start_epoch, epochs + 1):
        beta_kl = kl_weight_schedule(epoch)
        perm = np.random.permutation(len(X_tr))
        ep_l, ep_r, ep_k, ep_e, ep_p, n_b = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for i in range(0, len(X_tr), batch_size):
            b = perm[i:i + batch_size]
            loss, r, k, e, p = model.train_step(X_tr[b], C_tr[b], Y_tr[b], beta_kl, all_pal)
            ep_l += loss; ep_r += r; ep_k += k; ep_e += e; ep_p += p; n_b += 1
        tr_l = ep_l/max(1,n_b); tr_r = ep_r/max(1,n_b); tr_k = ep_k/max(1,n_b)
        tr_e = ep_e/max(1,n_b); tr_p = ep_p/max(1,n_b)

        va_l, va_r_sum, va_k_sum, va_n = 0.0, 0.0, 0.0, 0
        for i in range(0, len(X_va), batch_size):
            b = slice(i, i + batch_size)
            vl, vr, vk = model.eval_loss(X_va[b], C_va[b], Y_va[b], beta_kl, all_pal)
            va_l += vl; va_r_sum += vr; va_k_sum += vk; va_n += 1
        va_l_avg = va_l / max(1, va_n)
        va_r_avg = va_r_sum / max(1, va_n)
        va_k_avg = va_k_sum / max(1, va_n)

        history['train_loss'].append(tr_l); history['train_recon'].append(tr_r)
        history['train_kl'].append(tr_k); history['train_edge'].append(tr_e)
        history['train_pal'].append(tr_p); history['beta_kl'].append(beta_kl)
        history['val_loss'].append(va_l_avg); history['val_recon'].append(va_r_avg)
        history['val_kl'].append(va_k_avg)

        is_best = va_l_avg < best_val_loss
        if is_best: best_val_loss = va_l_avg

        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            eta = dt / epoch * (epochs - epoch)
            star = " ★" if is_best else ""
            print(f"   📊 Ep {epoch:5d}/{epochs} | tr={tr_l:.4f}(R{tr_r:.4f}+KL{tr_k:.2f}+E{tr_e:.4f}+P{tr_p:.4f}) | va={va_l_avg:.4f}{star} | bKL={beta_kl:.4f} | {dt:.0f}s ETA {eta:.0f}s")

        if (epoch % save_every == 0 or epoch == epochs or is_best):
            if is_best:
                model.save(CHECKPOINT_DIR / "best.npz", epoch, tr_l, va_l_avg)
            model.save(CHECKPOINT_DIR / f"model_epoch_{epoch:05d}.npz", epoch, tr_l, va_l_avg)
            model.save(CHECKPOINT_DIR / "latest.npz", epoch, tr_l, va_l_avg)
            render_sample(model, SAMPLES_DIR / f"sample_epoch_{epoch:05d}.png")
            gc.collect()

    manifest = {
        "arch_version": C.ARCH_VERSION,
        "model_type": "conditional_vae",
        "architecture": {
            "input_dim": C.INPUT_DIM, "latent_dim": C.LATENT_DIM,
            "output_dim": C.OUTPUT_DIM, "hidden_dim": C.HIDDEN_DIM,
            "condition_dim": C.CONDITION_DIM, "patch_size": C.PATCH_SIZE,
            "beta_kl_max": C.BETA_KL, "beta_kl_warmup": C.BETA_KL_WARMUP,
            "beta_edge": C.BETA_EDGE, "beta_palette": C.BETA_PALETTE,
        },
        "total_params": total_params, "epochs": epochs,
        "final_train_loss": float(history['train_loss'][-1]) if history['train_loss'] else None,
        "final_val_loss": float(history['val_loss'][-1]) if history['val_loss'] else None,
        "best_val_loss": float(best_val_loss),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_time_seconds": int(time.time() - t0),
        "palette_names": C.PALETTE_NAMES,
        "pixel_art_types": C.PIXEL_ART_TYPES,
        "moods": C.MOODS, "traits": C.TRAITS,
        "checkpoint_best": "best.npz",
        "checkpoint_latest": "latest.npz",
    }
    with open(MANIFEST_FILE, "w") as f: json.dump(manifest, f, indent=2)

    log_data = {
        "train_loss": history['train_loss'][::5],
        "val_loss": history['val_loss'][::5],
        "train_recon": history['train_recon'][::5],
        "train_kl": history['train_kl'][::5],
        "train_edge": history['train_edge'][::5],
        "train_pal": history['train_pal'][::5],
        "val_recon": history['val_recon'][::5],
        "val_kl": history['val_kl'][::5],
        "beta_kl": history['beta_kl'][::5],
        "best_val_loss": float(best_val_loss),
        "epochs": epochs, "training_time_seconds": int(time.time() - t0),
    }
    with open(TRAINING_LOG, "w") as f: json.dump(log_data, f, indent=2)

    print(f"\n✅ VAE completo em {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    if history['train_loss']:
        print(f"   Ultimo train_loss: {history['train_loss'][-1]:.5f}")
        print(f"   Ultimo val_loss:   {history['val_loss'][-1]:.5f}")
    print(f"   ★ Melhor val_loss: {best_val_loss:.5f} (em best.npz)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="🧠 Treinar VAE Pixel Art v" + C.ARCH_VERSION)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(a.epochs, a.batch_size, a.save_every, a.seed, a.resume)
