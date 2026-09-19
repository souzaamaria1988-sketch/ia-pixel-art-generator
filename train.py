#!/usr/bin/env python3
"""
🧠 TREINAMENTO AUTOENCODER CONDICIONAL - PIXEL ART
Backpropagation real em numpy puro
"""
import os, json, time, argparse, gc
from pathlib import Path
import numpy as np

MODEL_DIR = Path("models")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
MANIFEST_FILE = MODEL_DIR / "manifest.json"
TRAINING_LOG = MODEL_DIR / "training_log.json"

INPUT_DIM = 256
LATENT_DIM = 128
OUTPUT_DIM = 192
HIDDEN_DIM = 512
CONDITION_DIM = 64
PATCH_SIZE = 8

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m, self.v = {}, {}
    def update(self, name, param, grad):
        if name not in self.m:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
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

class ConditionalAutoencoder:
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        s_enc = np.sqrt(2.0 / (INPUT_DIM + CONDITION_DIM + HIDDEN_DIM))
        s_dec = np.sqrt(2.0 / (LATENT_DIM + HIDDEN_DIM))
        self.W_enc1 = (rng.randn(INPUT_DIM + CONDITION_DIM, HIDDEN_DIM) * s_enc).astype(np.float32)
        self.b_enc1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W_enc2 = (rng.randn(HIDDEN_DIM, LATENT_DIM) * s_enc).astype(np.float32)
        self.b_enc2 = np.zeros(LATENT_DIM, dtype=np.float32)
        self.W_dec1 = (rng.randn(LATENT_DIM, HIDDEN_DIM) * s_dec).astype(np.float32)
        self.b_dec1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W_dec2 = (rng.randn(HIDDEN_DIM, OUTPUT_DIM) * s_dec).astype(np.float32)
        self.b_dec2 = np.zeros(OUTPUT_DIM, dtype=np.float32)
        self.W_cond = (rng.randn(CONDITION_DIM, INPUT_DIM) * 0.01).astype(np.float32)
        self.opt = Adam(lr=0.001)

    def encode(self, x, cond):
        cond_proj = cond @ self.W_cond
        x_cond = np.concatenate([x, cond_proj], axis=-1)
        h1_pre = x_cond @ self.W_enc1 + self.b_enc1
        h1 = tanh(h1_pre)
        z_pre = h1 @ self.W_enc2 + self.b_enc2
        z = tanh(z_pre)
        cache = {'x_cond': x_cond, 'h1_pre': h1_pre, 'h1': h1, 'z_pre': z_pre, 'z': z}
        return z, cache

    def decode(self, z):
        h1_pre = z @ self.W_dec1 + self.b_dec1
        h1 = relu(h1_pre)
        out_pre = h1 @ self.W_dec2 + self.b_dec2
        out = tanh(out_pre)
        cache = {'z': z, 'h1_pre': h1_pre, 'h1': h1, 'out_pre': out_pre, 'out': out}
        return out, cache

    def forward(self, x, cond):
        z, enc = self.encode(x, cond)
        out, dec = self.decode(z)
        return out, enc, dec

    def backward(self, x, cond, target, enc, dec):
        bs = x.shape[0]
        d_out = 2.0 * (dec['out'] - target) / bs
        d_out_pre = d_out * tanh_grad(dec['out_pre'])
        d_W_dec2 = dec['h1'].T @ d_out_pre
        d_b_dec2 = d_out_pre.sum(axis=0)
        d_h1 = d_out_pre @ self.W_dec2.T
        d_h1_pre = d_h1 * relu_grad(dec['h1_pre'])
        d_W_dec1 = dec['z'].T @ d_h1_pre
        d_b_dec1 = d_h1_pre.sum(axis=0)
        d_z = d_h1_pre @ self.W_dec1.T
        d_z_pre = d_z * tanh_grad(enc['z_pre'])
        d_W_enc2 = enc['h1'].T @ d_z_pre
        d_b_enc2 = d_z_pre.sum(axis=0)
        d_h1_e = d_z_pre @ self.W_enc2.T
        d_h1_pre_e = d_h1_e * tanh_grad(enc['h1_pre'])
        d_x_cond = d_h1_pre_e @ self.W_enc1.T
        d_W_enc1 = enc['x_cond'].T @ d_h1_pre_e
        d_b_enc1 = d_h1_pre_e.sum(axis=0)
        d_cond_proj = d_x_cond[:, INPUT_DIM:]
        d_W_cond = cond.T @ d_cond_proj
        return {'W_enc1': d_W_enc1, 'b_enc1': d_b_enc1, 'W_enc2': d_W_enc2, 'b_enc2': d_b_enc2,
                'W_dec1': d_W_dec1, 'b_dec1': d_b_dec1, 'W_dec2': d_W_dec2, 'b_dec2': d_b_dec2,
                'W_cond': d_W_cond}

    def train_step(self, x, cond, target):
        out, enc, dec = self.forward(x, cond)
        loss = float(np.mean((out - target) ** 2))
        grads = self.backward(x, cond, target, enc, dec)
        for name, g in grads.items():
            setattr(self, name, self.opt.update(name, getattr(self, name), g))
        return loss

    def save(self, path, epoch, loss):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, W_enc1=self.W_enc1, b_enc1=self.b_enc1,
            W_enc2=self.W_enc2, b_enc2=self.b_enc2, W_dec1=self.W_dec1, b_dec1=self.b_dec1,
            W_dec2=self.W_dec2, b_dec2=self.b_dec2, W_cond=self.W_cond,
            epoch=np.array(epoch), loss=np.array(loss), timestamp=np.array(time.time()))

    def load(self, path):
        d = np.load(path)
        for k in ['W_enc1','b_enc1','W_enc2','b_enc2','W_dec1','b_dec1','W_dec2','b_dec2','W_cond']:
            setattr(self, k, d[k])
        return int(d['epoch']), float(d['loss'])

def generate_dataset(n_samples=5000, seed=123):
    print("📦 Gerando dataset sintético...")
    rng = np.random.RandomState(seed)
    palettes = [
        np.array([[0x7C,0x7C,0x7C],[0x00,0x00,0xFC],[0xA8,0x10,0x00],[0x00,0xB8,0x00],[0xF8,0x38,0x00],[0xFC,0xFC,0xFC]],dtype=np.float32)/255.,
        np.array([[0x0f,0x38,0x0f],[0x30,0x62,0x30],[0x8b,0xac,0x0f],[0x9b,0xbc,0x0e]],dtype=np.float32)/255.,
        np.array([[0x0A,0x0A,0x0A],[0x1A,0x1A,0x1A],[0x2C,0x2C,0x2C],[0x3D,0x3D,0x3D],[0x4F,0x4F,0x4F]],dtype=np.float32)/255.,
        np.array([[0xFF,0x00,0x6E],[0x00,0xF0,0xFF],[0x39,0xFF,0x14],[0xFF,0xF2,0x00],[0xBC,0x13,0xFE]],dtype=np.float32)/255.,
        np.array([[0,0,0],[0x55,0xFF,0xFF],[0xFF,0x55,0xFF],[0xFF,0xFF,0xFF]],dtype=np.float32)/255.,
        np.array([[0xFF,0xD1,0xDC],[0xFF,0xB7,0xC5],[0xFD,0xFD,0x96],[0xB5,0xEA,0xD7],[0xC7,0xCE,0xEA]],dtype=np.float32)/255.,
    ]
    n_styles = len(palettes)
    X, Y, C = [], [], []
    for i in range(n_samples):
        style_idx = i % n_styles
        pal = palettes[style_idx]
        patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
        ptype = rng.randint(0, 8)
        if ptype == 0:
            patch[:] = pal[rng.randint(0, len(pal))]
        elif ptype == 1:
            for x in range(PATCH_SIZE): patch[:, x] = pal[x % len(pal)]
        elif ptype == 2:
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE): patch[y, x] = pal[(x + y) % len(pal)]
        elif ptype == 3:
            y, x = PATCH_SIZE // 2, PATCH_SIZE // 2
            c = pal[0]
            for _ in range(PATCH_SIZE * PATCH_SIZE):
                if 0 <= y < PATCH_SIZE and 0 <= x < PATCH_SIZE: patch[y, x] = c
                dy, dx = rng.randint(-1, 2, 2)
                y, x = y + dy, x + dx
                if rng.random() < 0.15: c = pal[rng.randint(0, len(pal))]
        elif ptype == 4:
            for y in range(PATCH_SIZE):
                t = y / (PATCH_SIZE - 1)
                patch[y, :] = pal[0] * (1 - t) + pal[-1] * t
        elif ptype == 5:
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    patch[y, x] = pal[rng.randint(0, len(pal))] if rng.random() < 0.5 else pal[0]
        elif ptype == 6:
            cx, cy, r = PATCH_SIZE/2, PATCH_SIZE/2, PATCH_SIZE/3
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    d = np.sqrt((x-cx)**2 + (y-cy)**2)
                    patch[y, x] = pal[1] if d < r else pal[0]
        else:
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE): patch[y, x] = pal[rng.randint(0, len(pal))]
        patch_norm = patch * 2.0 - 1.0
        hist = np.histogram(patch.reshape(-1, 3), bins=16, range=(0, 1))[0]
        hist = hist / (hist.sum() + 1e-8)
        stats = np.array([patch.mean(), patch.std(), patch.min(), patch.max(), np.unique(patch.reshape(-1,3),axis=0).shape[0]/64.0])
        feat = np.concatenate([hist, stats]).astype(np.float32)
        if len(feat) < INPUT_DIM: feat = np.pad(feat, (0, INPUT_DIM - len(feat)))
        cond = np.zeros(CONDITION_DIM, dtype=np.float32)
        cond[style_idx] = 1.0
        cond[10 + style_idx] = rng.randn() * 0.1
        cond[20 + (ptype % 8)] = 0.5
        X.append(feat[:INPUT_DIM]); Y.append(patch_norm.reshape(-1)); C.append(cond)
    X = np.array(X, dtype=np.float32); Y = np.array(Y, dtype=np.float32); C = np.array(C, dtype=np.float32)
    print(f"   ✓ Dataset: X{X.shape} Y{Y.shape} C{C.shape}")
    return X, Y, C

def train(epochs=10000, batch_size=64, save_every=1000, seed=42, resume=False):
    print("="*70); print("🧠 TREINAMENTO AUTOENCODER - PIXEL ART"); print("="*70)
    np.random.seed(seed)
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    X, Y, C = generate_dataset(5000, seed=123)
    model = ConditionalAutoencoder(seed=seed)
    total_params = sum(p.size for p in [model.W_enc1, model.b_enc1, model.W_enc2, model.b_enc2,
                                         model.W_dec1, model.b_dec1, model.W_dec2, model.b_dec2, model.W_cond])
    print(f"🧮 Parâmetros: {total_params:,}")
    print(f"🎯 Épocas: {epochs} | Batch: {batch_size}")
    start_epoch = 1
    losses = []
    best_loss = float('inf')
    if resume:
        latest = CHECKPOINT_DIR / "latest.npz"
        if latest.exists():
            start_epoch, prev_loss = model.load(latest)
            start_epoch += 1
            best_loss = prev_loss
            print(f"📂 Retomando do epoch {start_epoch - 1} (loss {prev_loss:.5f})")
    for epoch in range(start_epoch, epochs + 1):
        idx = np.random.permutation(len(X))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X), batch_size):
            b_idx = idx[i:i + batch_size]
            xb, yb, cb = X[b_idx], Y[b_idx], C[b_idx]
            loss = model.train_step(xb, cb, yb)
            epoch_loss += loss
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)
        if avg_loss < best_loss: best_loss = avg_loss
        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            eta = dt / epoch * (epochs - epoch)
            print(f"   📊 Epoch {epoch:5d}/{epochs} | loss={avg_loss:.5f} | best={best_loss:.5f} | {dt:.0f}s (ETA {eta:.0f}s)")
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = CHECKPOINT_DIR / f"model_epoch_{epoch:05d}.npz"
            model.save(ckpt, epoch, avg_loss)
            model.save(CHECKPOINT_DIR / "latest.npz", epoch, avg_loss)
            print(f"   💾 Checkpoint: {ckpt.name}")
            gc.collect()
    manifest = {"model_type": "conditional_autoencoder",
                "architecture": {"input_dim": INPUT_DIM, "latent_dim": LATENT_DIM, "output_dim": OUTPUT_DIM,
                                 "hidden_dim": HIDDEN_DIM, "condition_dim": CONDITION_DIM, "patch_size": PATCH_SIZE},
                "total_params": total_params, "epochs": epochs,
                "final_loss": float(losses[-1]) if losses else None, "best_loss": float(best_loss),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "training_time_seconds": int(time.time() - t0),
                "checkpoint": "latest.npz", "runner": "ubuntu-latest"}
    with open(MANIFEST_FILE, "w") as f: json.dump(manifest, f, indent=2)
    log_data = {"losses": losses[::10], "best_loss": float(best_loss),
                "final_loss": float(losses[-1]) if losses else None,
                "epochs": epochs, "training_time_seconds": int(time.time() - t0)}
    with open(TRAINING_LOG, "w") as f: json.dump(log_data, f, indent=2)
    print(f"\n✅ Treino completo em {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"   Loss final: {losses[-1]:.5f}" if losses else "   Sem losses")
    print(f"   Melhor loss: {best_loss:.5f}")
    print(f"   Modelo: {CHECKPOINT_DIR/'latest.npz'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="🧠 Treinar Autoencoder Pixel Art")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(a.epochs, a.batch_size, a.save_every, a.seed, a.resume)