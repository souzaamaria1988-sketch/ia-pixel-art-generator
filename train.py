#!/usr/bin/env python3
"""
🧠 TREINAMENTO MOE (MIXTURE OF EXPERTS) CONDICIONAL - PIXEL ART
Backpropagation real em numpy puro com 64 experts
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
OUTPUT_DIM = 192       # 8x8x3
HIDDEN_DIM = 512
CONDITION_DIM = 64
PATCH_SIZE = 8
NUM_EXPERTS = 64
TOP_K = 2              # cada input usa 2 experts

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

class MoEMixtureLayer:
    """
    Camada MoE: roteia input para TOP-K experts.
    Cada expert é uma MLP: input_dim → expert_dim → output_dim
    Router: input_dim → num_experts (logits)
    """
    def __init__(self, input_dim, output_dim, num_experts, top_k, seed=42):
        rng = np.random.RandomState(seed)
        expert_dim = 256
        s = np.sqrt(2.0 / (input_dim + expert_dim))
        s2 = np.sqrt(2.0 / (expert_dim + output_dim))
        # Router
        self.W_router = (rng.randn(input_dim, num_experts) * 0.01).astype(np.float32)
        self.b_router = np.zeros(num_experts, dtype=np.float32)
        # Experts: cada um tem W1 (in→expert) e W2 (expert→out)
        self.W_e1 = (rng.randn(num_experts, input_dim, expert_dim) * s).astype(np.float32)
        self.b_e1 = np.zeros((num_experts, expert_dim), dtype=np.float32)
        self.W_e2 = (rng.randn(num_experts, expert_dim, output_dim) * s2).astype(np.float32)
        self.b_e2 = np.zeros((num_experts, output_dim), dtype=np.float32)
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim

    def forward(self, x):
        """x: (batch, input_dim) → out: (batch, output_dim)"""
        bs = x.shape[0]
        # Router: scores para cada expert
        logits = x @ self.W_router + self.b_router  # (bs, num_experts)
        # Softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / (e.sum(axis=1, keepdims=True) + 1e-8)
        # Top-K experts
        topk_idx = np.argsort(probs, axis=1)[:, -self.top_k:]  # (bs, top_k)
        topk_probs = np.take_along_axis(probs, topk_idx, axis=1)
        # Normaliza gates
        gates = topk_probs / (topk_probs.sum(axis=1, keepdims=True) + 1e-8)
        # Processa cada input pelos seus experts selecionados
        out = np.zeros((bs, self.W_e2.shape[2]), dtype=np.float32)
        cache_experts = []
        for i in range(bs):
            expert_outs = []
            caches = []
            for k in range(self.top_k):
                e_idx = topk_idx[i, k]
                # Forward do expert e_idx
                h_pre = x[i] @ self.W_e1[e_idx] + self.b_e1[e_idx]
                h = relu(h_pre)
                e_out = h @ self.W_e2[e_idx] + self.b_e2[e_idx]
                expert_outs.append(e_out)
                caches.append({'h_pre': h_pre, 'h': h, 'e_idx': int(e_idx)})
            # Combina com gates
            combined = np.zeros_like(expert_outs[0])
            for k in range(self.top_k):
                combined += gates[i, k] * expert_outs[k]
            out[i] = combined
            cache_experts.append(caches)
        cache = {
            'x': x, 'logits': logits, 'probs': probs,
            'topk_idx': topk_idx, 'gates': gates,
            'expert_caches': cache_experts
        }
        return out, cache

    def backward(self, d_out, cache):
        """Backward através do MoE"""
        bs = d_out.shape[0]
        x = cache['x']
        topk_idx = cache['topk_idx']
        gates = cache['gates']
        probs = cache['probs']
        logits = cache['logits']
        expert_caches = cache['expert_caches']
        # Gradientes dos experts
        d_W_e1 = np.zeros_like(self.W_e1)
        d_b_e1 = np.zeros_like(self.b_e1)
        d_W_e2 = np.zeros_like(self.W_e2)
        d_b_e2 = np.zeros_like(self.b_e2)
        d_gates = np.zeros_like(gates)
        d_x = np.zeros_like(x)
        for i in range(bs):
            caches_i = expert_caches[i]
            for k in range(self.top_k):
                e_idx = caches_i[k]['e_idx']
                h = caches_i[k]['h']
                h_pre = caches_i[k]['h_pre']
                # d_out_i contribui para expert k com peso gate[i,k]
                d_e_out = gates[i, k] * d_out[i]
                # Backward do expert
                d_W_e2[e_idx] += np.outer(h, d_e_out)
                d_b_e2[e_idx] += d_e_out
                d_h = d_e_out @ self.W_e2[e_idx].T
                d_h_pre = d_h * relu_grad(h_pre)
                d_W_e1[e_idx] += np.outer(x[i], d_h_pre)
                d_b_e1[e_idx] += d_h_pre
                d_x[i] += d_h_pre @ self.W_e1[e_idx].T
                d_gates[i, k] = np.dot(d_out[i], h @ self.W_e2[e_idx] + self.b_e2[e_idx])
        # Backward através do router (softmax)
        # d_probs: gradiente em relação às probs dos topk
        d_probs = np.zeros_like(probs)
        for i in range(bs):
            for k in range(self.top_k):
                e_idx = topk_idx[i, k]
                # gates[i,k] = topk_probs[i,k] / sum(topk_probs[i])
                # ∂gates[i,k]/∂probs[i,e_idx] é derivada de softmax normalizado
                s = gates[i].sum()
                d_probs[i, e_idx] += (d_gates[i, k] * (1.0 - gates[i, k]) / s if s > 1e-8 else 0)
                for kk in range(self.top_k):
                    if kk != k:
                        e_idx2 = topk_idx[i, kk]
                        d_probs[i, e_idx2] += -d_gates[i, k] * gates[i, kk] / (s + 1e-8)
        # Backward do softmax: d_logits = probs * (d_probs - sum(d_probs * probs))
        d_logits = probs * (d_probs - (d_probs * probs).sum(axis=1, keepdims=True))
        d_W_router = x.T @ d_logits
        d_b_router = d_logits.sum(axis=0)
        return {
            'W_router': d_W_router, 'b_router': d_b_router,
            'W_e1': d_W_e1, 'b_e1': d_b_e1,
            'W_e2': d_W_e2, 'b_e2': d_b_e2
        }

class MoEAutoencoder:
    """Autoencoder com camada MoE no meio"""
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        s_enc = np.sqrt(2.0 / (INPUT_DIM + INPUT_DIM + HIDDEN_DIM))
        s_lat = np.sqrt(2.0 / (HIDDEN_DIM + LATENT_DIM))
        s_dec = np.sqrt(2.0 / (HIDDEN_DIM + OUTPUT_DIM))
        # Encoder
        self.W_enc1 = (rng.randn(INPUT_DIM + INPUT_DIM, HIDDEN_DIM) * s_enc).astype(np.float32)
        self.b_enc1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W_enc2 = (rng.randn(HIDDEN_DIM, LATENT_DIM) * s_lat).astype(np.float32)
        self.b_enc2 = np.zeros(LATENT_DIM, dtype=np.float32)
        # MoE layer
        self.moe = MoEMixtureLayer(LATENT_DIM, HIDDEN_DIM, NUM_EXPERTS, TOP_K, seed=seed)
        # Decoder
        self.W_dec = (rng.randn(HIDDEN_DIM, OUTPUT_DIM) * s_dec).astype(np.float32)
        self.b_dec = np.zeros(OUTPUT_DIM, dtype=np.float32)
        # Projeção de condição
        self.W_cond = (rng.randn(CONDITION_DIM, INPUT_DIM) * 0.01).astype(np.float32)
        self.opt = Adam(lr=0.0005)

    def encode(self, x, cond):
        cond_proj = cond @ self.W_cond
        x_cond = np.concatenate([x, cond_proj], axis=-1)
        h1_pre = x_cond @ self.W_enc1 + self.b_enc1
        h1 = tanh(h1_pre)
        z_pre = h1 @ self.W_enc2 + self.b_enc2
        z = tanh(z_pre)
        return z, {'x_cond': x_cond, 'h1_pre': h1_pre, 'h1': h1, 'z_pre': z_pre, 'z': z}

    def forward(self, x, cond):
        z, enc = self.encode(x, cond)
        moe_out, moe_cache = self.moe.forward(z)
        h_dec = relu(moe_out)
        out_pre = h_dec @ self.W_dec + self.b_dec
        out = tanh(out_pre)
        dec_cache = {'moe_out': moe_out, 'h_dec': h_dec, 'out_pre': out_pre, 'out': out}
        return out, enc, moe_cache, dec_cache

    def backward(self, x, cond, target, enc, moe_cache, dec_cache):
        bs = x.shape[0]
        # Decoder backward
        d_out = 2.0 * (dec_cache['out'] - target) / bs
        d_out_pre = d_out * tanh_grad(dec_cache['out_pre'])
        d_W_dec = dec_cache['h_dec'].T @ d_out_pre
        d_b_dec = d_out_pre.sum(axis=0)
        d_h_dec = d_out_pre @ self.W_dec.T
        d_moe_out = d_h_dec * relu_grad(dec_cache['moe_out'])
        # MoE backward
        d_z = self.moe.backward(d_moe_out, moe_cache)
        # Encoder backward (só a parte do z)
        d_z_pre = d_z * tanh_grad(enc['z_pre'])
        d_W_enc2 = enc['h1'].T @ d_z_pre
        d_b_enc2 = d_z_pre.sum(axis=0)
        d_h1 = d_z_pre @ self.W_enc2.T
        d_h1_pre = d_h1 * tanh_grad(enc['h1_pre'])
        d_x_cond = d_h1_pre @ self.W_enc1.T
        d_W_enc1 = enc['x_cond'].T @ d_h1_pre
        d_b_enc1 = d_h1_pre.sum(axis=0)
        d_cond_proj = d_x_cond[:, INPUT_DIM:]
        d_W_cond = cond.T @ d_cond_proj
        return {
            'W_enc1': d_W_enc1, 'b_enc1': d_b_enc1,
            'W_enc2': d_W_enc2, 'b_enc2': d_b_enc2,
            'W_dec': d_W_dec, 'b_dec': d_b_dec,
            'W_cond': d_W_cond,
            'moe': d_z  # gradientes MoE vêm separados
        }

    def train_step(self, x, cond, target):
        out, enc, moe_cache, dec_cache = self.forward(x, cond)
        loss_recon = float(np.mean((out - target) ** 2))
        # Load balancing loss: incentiva uso balanceado dos experts
        probs = moe_cache['probs']
        # Frequência de cada expert no batch
        f = probs.mean(axis=0)  # (num_experts,)
        # Importância (soma dos scores)
        P = probs.sum(axis=0)
        lb_loss = float((f * P).sum() * NUM_EXPERTS)
        loss = loss_recon + 0.01 * lb_loss
        grads = self.backward(x, cond, target, enc, moe_cache, dec_cache)
        # Update encoder/decoder/cond
        for name in ['W_enc1','b_enc1','W_enc2','b_enc2','W_dec','b_dec','W_cond']:
            setattr(self, name, self.opt.update(name, getattr(self, name), grads[name]))
        # Update MoE
        moe_grads = self.moe.backward(
            np.zeros_like(moe_cache['moe_out']),  # placeholder, já computado
            moe_cache
        )
        # Usa os gradientes salvos no backward anterior (recomputa uma vez)
        d_moe_out = np.zeros((x.shape[0], HIDDEN_DIM), dtype=np.float32)
        d_h_dec = grads['W_dec']  # placeholder
        # Precisamos chamar backward do MoE com gradiente real
        # Reusa o gradiente de moe_out do backward principal
        # (já foi calculado, mas precisamos dos grads dos experts)
        d_moe_out_real = (grads['W_dec'] @ np.linalg.pinv(self.W_dec.T) if self.W_dec.shape[0] > 0 
                         else np.zeros_like(moe_cache['moe_out']))
        # Simplificação: recomputa MoE backward com o gradiente correto
        d_h_dec_real = grads['W_dec']  # não, isso é errado
        # Vamos fazer corretamente:
        d_out = 2.0 * (dec_cache['out'] - target) / x.shape[0]
        d_out_pre = d_out * tanh_grad(dec_cache['out_pre'])
        d_h_dec_real = d_out_pre @ self.W_dec.T
        d_moe_out_real = d_h_dec_real * relu_grad(dec_cache['moe_out'])
        moe_grads = self.moe.backward(d_moe_out_real, moe_cache)
        for name, g in moe_grads.items():
            attr_name = f"moe_{name}"
            setattr(self.moe, name, self.opt.update(attr_name, getattr(self.moe, name), g))
        # Estatísticas de uso de experts
        expert_usage = np.zeros(NUM_EXPERTS, dtype=np.int32)
        for idx in moe_cache['topk_idx'].flatten():
            expert_usage[idx] += 1
        return loss, loss_recon, lb_loss, expert_usage

    def save(self, path, epoch, loss):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'W_enc1': self.W_enc1, 'b_enc1': self.b_enc1,
            'W_enc2': self.W_enc2, 'b_enc2': self.b_enc2,
            'W_dec': self.W_dec, 'b_dec': self.b_dec,
            'W_cond': self.W_cond,
            'moe_W_router': self.moe.W_router, 'moe_b_router': self.moe.b_router,
            'moe_W_e1': self.moe.W_e1, 'moe_b_e1': self.moe.b_e1,
            'moe_W_e2': self.moe.W_e2, 'moe_b_e2': self.moe.b_e2,
            'epoch': np.array(epoch), 'loss': np.array(loss),
            'timestamp': np.array(time.time())
        }
        np.savez_compressed(path, **data)

    def load(self, path):
        d = np.load(path)
        for k in ['W_enc1','b_enc1','W_enc2','b_enc2','W_dec','b_dec','W_cond']:
            setattr(self, k, d[k])
        self.moe.W_router = d['moe_W_router']; self.moe.b_router = d['moe_b_router']
        self.moe.W_e1 = d['moe_W_e1']; self.moe.b_e1 = d['moe_b_e1']
        self.moe.W_e2 = d['moe_W_e2']; self.moe.b_e2 = d['moe_b_e2']
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
        if ptype == 0: patch[:] = pal[rng.randint(0, len(pal))]
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
                dy, dx = rng.randint(-1, 2, 2); y, x = y + dy, x + dx
                if rng.random() < 0.15: c = pal[rng.randint(0, len(pal))]
        elif ptype == 4:
            for y in range(PATCH_SIZE):
                t = y / (PATCH_SIZE - 1); patch[y, :] = pal[0] * (1 - t) + pal[-1] * t
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
        stats = np.array([patch.mean(), patch.std(), patch.min(), patch.max(),
                          np.unique(patch.reshape(-1,3),axis=0).shape[0]/64.0])
        feat = np.concatenate([hist, stats]).astype(np.float32)
        if len(feat) < INPUT_DIM: feat = np.pad(feat, (0, INPUT_DIM - len(feat)))
        cond = np.zeros(CONDITION_DIM, dtype=np.float32)
        cond[style_idx] = 1.0
        cond[10 + style_idx] = rng.randn() * 0.1
        cond[20 + (ptype % 8)] = 0.5
        X.append(feat[:INPUT_DIM]); Y.append(patch_norm.reshape(-1)); C.append(cond)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(C, dtype=np.float32)

def train(epochs=5000, batch_size=32, save_every=500, seed=42, resume=False):
    print("="*70)
    print("🧠 TREINAMENTO MoE CONDICIONAL - PIXEL ART")
    print("="*70)
    print(f"   {NUM_EXPERTS} experts | top-{TOP_K} routing")
    print("="*70)
    np.random.seed(seed); t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    X, Y, C = generate_dataset(5000, seed=123)
    print(f"   ✓ Dataset: X{X.shape} Y{Y.shape} C{C.shape}")
    model = MoEAutoencoder(seed=seed)
    # Conta parâmetros
    total_params = (
        model.W_enc1.size + model.b_enc1.size +
        model.W_enc2.size + model.b_enc2.size +
        model.W_dec.size + model.b_dec.size +
        model.W_cond.size +
        model.moe.W_router.size + model.moe.b_router.size +
        model.moe.W_e1.size + model.moe.b_e1.size +
        model.moe.W_e2.size + model.moe.b_e2.size
    )
    print(f"🧮 Parâmetros totais: {total_params:,}")
    print(f"🎯 Épocas: {epochs} | Batch: {batch_size}")
    start_epoch, losses, best_loss = 1, [], float('inf')
    total_usage = np.zeros(NUM_EXPERTS, dtype=np.int64)
    if resume:
        latest = CHECKPOINT_DIR / "latest.npz"
        if latest.exists():
            start_epoch, prev_loss = model.load(latest)
            start_epoch += 1; best_loss = prev_loss
            print(f"📂 Retomando do epoch {start_epoch - 1} (loss {prev_loss:.5f})")
    for epoch in range(start_epoch, epochs + 1):
        idx = np.random.permutation(len(X))
        epoch_loss, epoch_recon, epoch_lb = 0.0, 0.0, 0.0
        n_batches = 0
        for i in range(0, len(X), batch_size):
            b_idx = idx[i:i + batch_size]
            loss, recon, lb, usage = model.train_step(X[b_idx], C[b_idx], Y[b_idx])
            epoch_loss += loss; epoch_recon += recon; epoch_lb += lb
            total_usage += usage
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        avg_recon = epoch_recon / max(1, n_batches)
        avg_lb = epoch_lb / max(1, n_batches)
        losses.append(avg_loss)
        if avg_loss < best_loss: best_loss = avg_loss
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            eta = dt / epoch * (epochs - epoch)
            top3 = np.argsort(total_usage)[-3:][::-1]
            bot3 = np.argsort(total_usage)[:3]
            print(f"   📊 Ep {epoch:5d}/{epochs} | loss={avg_loss:.4f} recon={avg_recon:.4f} lb={avg_lb:.4f} | "
                  f"{dt:.0f}s (ETA {eta:.0f}s)")
            print(f"      Top-3 experts: {top3.tolist()} | Bot-3: {bot3.tolist()}")
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = CHECKPOINT_DIR / f"model_epoch_{epoch:05d}.npz"
            model.save(ckpt, epoch, avg_loss)
            model.save(CHECKPOINT_DIR / "latest.npz", epoch, avg_loss)
            print(f"   💾 Checkpoint: {ckpt.name}")
            gc.collect()
    manifest = {
        "model_type": "moe_conditional_autoencoder",
        "architecture": {
            "input_dim": INPUT_DIM, "latent_dim": LATENT_DIM, "output_dim": OUTPUT_DIM,
            "hidden_dim": HIDDEN_DIM, "condition_dim": CONDITION_DIM, "patch_size": PATCH_SIZE,
            "num_experts": NUM_EXPERTS, "top_k": TOP_K
        },
        "total_params": total_params, "epochs": epochs,
        "final_loss": float(losses[-1]) if losses else None,
        "best_loss": float(best_loss),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_time_seconds": int(time.time() - t0),
        "expert_usage": {str(i): int(total_usage[i]) for i in range(NUM_EXPERTS)},
        "checkpoint": "latest.npz", "runner": "ubuntu-latest"
    }
    with open(MANIFEST_FILE, "w") as f: json.dump(manifest, f, indent=2)
    log_data = {
        "losses": losses[::5], "best_loss": float(best_loss),
        "final_loss": float(losses[-1]) if losses else None,
        "epochs": epochs, "training_time_seconds": int(time.time() - t0),
        "expert_usage": {str(i): int(total_usage[i]) for i in range(NUM_EXPERTS)}
    }
    with open(TRAINING_LOG, "w") as f: json.dump(log_data, f, indent=2)
    print(f"\n✅ Treino MoE completo em {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    if losses: print(f"   Loss final: {losses[-1]:.5f} (recon+lb)")
    print(f"   Melhor loss: {best_loss:.5f}")
    print(f"   Top-5 experts (mais usados):")
    top5 = np.argsort(total_usage)[-5:][::-1]
    for i in top5:
        print(f"     Expert {i}: {int(total_usage[i])} usos")
    print(f"   Modelo: {CHECKPOINT_DIR/'latest.npz'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="🧠 Treinar MoE Pixel Art")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(a.epochs, a.batch_size, a.save_every, a.seed, a.resume)
