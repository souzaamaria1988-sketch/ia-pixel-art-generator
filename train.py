#!/usr/bin/env python3
"""
🧠 VAE (Variational Autoencoder) CONDICIONAL - PIXEL ART
Modelo generativo REAL com KL divergence e reparameterization trick
Dataset sintético com sprites reais (não listras/círculos)
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
BETA_KL = 0.001        # peso da KL divergence

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

class ConditionalVAE:
    """VAE condicional: encoder -> (mu, log_var) -> z -> decoder"""
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        # Escalas de inicialização (He/Xavier)
        s_enc = np.sqrt(2.0 / (INPUT_DIM + INPUT_DIM + HIDDEN_DIM))
        s_lat = np.sqrt(2.0 / (HIDDEN_DIM + LATENT_DIM))
        s_dec = np.sqrt(2.0 / (LATENT_DIM + HIDDEN_DIM))

        # ENCODER: (input + cond_proj) -> h1 -> (mu, log_var)
        self.W_enc1 = (rng.randn(INPUT_DIM + INPUT_DIM, HIDDEN_DIM) * s_enc).astype(np.float32)
        self.b_enc1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        # Saída dupla: média + log-variance
        self.W_mu = (rng.randn(HIDDEN_DIM, LATENT_DIM) * s_lat).astype(np.float32)
        self.b_mu = np.zeros(LATENT_DIM, dtype=np.float32)
        self.W_logvar = (rng.randn(HIDDEN_DIM, LATENT_DIM) * s_lat).astype(np.float32)
        self.b_logvar = np.zeros(LATENT_DIM, dtype=np.float32)

        # DECODER: z -> h1 -> out
        self.W_dec1 = (rng.randn(LATENT_DIM, HIDDEN_DIM) * s_dec).astype(np.float32)
        self.b_dec1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W_dec2 = (rng.randn(HIDDEN_DIM, OUTPUT_DIM) * s_dec).astype(np.float32)
        self.b_dec2 = np.zeros(OUTPUT_DIM, dtype=np.float32)

        # Projeção de condição
        self.W_cond = (rng.randn(CONDITION_DIM, INPUT_DIM) * 0.01).astype(np.float32)
        self.opt = Adam(lr=0.0008)

    def encode(self, x, cond):
        """Retorna (mu, log_var) em vez de z direto."""
        cond_proj = cond @ self.W_cond
        x_cond = np.concatenate([x, cond_proj], axis=-1)
        h1_pre = x_cond @ self.W_enc1 + self.b_enc1
        h1 = tanh(h1_pre)
        mu = h1 @ self.W_mu + self.b_mu
        log_var = h1 @ self.W_logvar + self.b_logvar
        # Clamp log_var para estabilidade
        log_var = np.clip(log_var, -8.0, 8.0)
        cache = {'x_cond': x_cond, 'h1_pre': h1_pre, 'h1': h1, 'mu': mu, 'log_var': log_var}
        return mu, log_var, cache

    def reparameterize(self, mu, log_var, rng=None):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = np.exp(0.5 * log_var)
        if rng is None:
            eps = np.random.randn(*mu.shape).astype(np.float32)
        else:
            eps = rng.randn(*mu.shape).astype(np.float32)
        z = mu + std * eps
        return z, eps, std

    def decode(self, z):
        """Forward do decoder: z -> patch"""
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

    def backward(self, x, cond, target, enc_cache, dec_cache, z, eps, std):
        bs = x.shape[0]
        # Gradiente da reconstrução (MSE)
        d_out = 2.0 * (dec_cache['out'] - target) / bs
        d_out_pre = d_out * tanh_grad(dec_cache['out_pre'])
        d_W_dec2 = dec_cache['h1'].T @ d_out_pre
        d_b_dec2 = d_out_pre.sum(axis=0)
        d_h1 = d_out_pre @ self.W_dec2.T
        d_h1_pre = d_h1 * relu_grad(dec_cache['h1_pre'])
        d_W_dec1 = dec_cache['z'].T @ d_h1_pre
        d_b_dec1 = d_h1_pre.sum(axis=0)
        d_z = d_h1_pre @ self.W_dec1.T

        # Gradientes de mu e log_var via reparameterization
        # z = mu + std * eps  where std = exp(0.5 * log_var)
        # dz/d_mu = 1
        # dz/d_logvar = 0.5 * std * eps
        d_mu_recon = d_z
        d_logvar_recon = d_z * 0.5 * std * eps

        # Gradiente da KL divergence
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        # d_KL/d_mu = mu
        # d_KL/d_log_var = 0.5 * (exp(log_var) - 1)
        mu = enc_cache['mu']
        log_var = enc_cache['log_var']
        d_mu_kl = mu / bs
        d_logvar_kl = 0.5 * (np.exp(log_var) - 1.0) / bs

        d_mu = d_mu_recon + BETA_KL * d_mu_kl
        d_logvar = d_logvar_recon + BETA_KL * d_logvar_kl

        # Clamp gradientes de log_var para estabilidade
        d_logvar = np.clip(d_logvar, -1.0, 1.0)

        # Backprop através de mu e log_var até h1
        d_W_mu = enc_cache['h1'].T @ d_mu
        d_b_mu = d_mu.sum(axis=0)
        d_W_logvar = enc_cache['h1'].T @ d_logvar
        d_b_logvar = d_logvar.sum(axis=0)
        d_h1_from_latent = d_mu @ self.W_mu.T + d_logvar @ self.W_logvar.T
        d_h1_pre = d_h1_from_latent * tanh_grad(enc_cache['h1_pre'])
        d_x_cond = d_h1_pre @ self.W_enc1.T
        d_W_enc1 = enc_cache['x_cond'].T @ d_h1_pre
        d_b_enc1 = d_h1_pre.sum(axis=0)
        d_cond_proj = d_x_cond[:, INPUT_DIM:]
        d_W_cond = cond.T @ d_cond_proj

        return {
            'W_enc1': d_W_enc1, 'b_enc1': d_b_enc1,
            'W_mu': d_W_mu, 'b_mu': d_b_mu,
            'W_logvar': d_W_logvar, 'b_logvar': d_b_logvar,
            'W_dec1': d_W_dec1, 'b_dec1': d_b_dec1,
            'W_dec2': d_W_dec2, 'b_dec2': d_b_dec2,
            'W_cond': d_W_cond
        }

    def train_step(self, x, cond, target):
        out, mu, log_var, enc, dec, z, eps, std = self.forward(x, cond)
        recon_loss = float(np.mean((out - target) ** 2))
        # KL divergence analítica para gaussiana
        kl_loss = float(-0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var)))
        total_loss = recon_loss + BETA_KL * kl_loss
        grads = self.backward(x, cond, target, enc, dec, z, eps, std)
        for name, g in grads.items():
            setattr(self, name, self.opt.update(name, getattr(self, name), g))
        return total_loss, recon_loss, kl_loss

    def save(self, path, epoch, loss):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path,
            W_enc1=self.W_enc1, b_enc1=self.b_enc1,
            W_mu=self.W_mu, b_mu=self.b_mu,
            W_logvar=self.W_logvar, b_logvar=self.b_logvar,
            W_dec1=self.W_dec1, b_dec1=self.b_dec1,
            W_dec2=self.W_dec2, b_dec2=self.b_dec2,
            W_cond=self.W_cond,
            epoch=np.array(epoch), loss=np.array(loss),
            timestamp=np.array(time.time()))

    def load(self, path):
        d = np.load(path)
        for k in ['W_enc1','b_enc1','W_mu','b_mu','W_logvar','b_logvar',
                  'W_dec1','b_dec1','W_dec2','b_dec2','W_cond']:
            if k in d.files: setattr(self, k, d[k])
        return int(d['epoch']), float(d['loss'])

def generate_realistic_patches(n_samples=8000, seed=123):
    """Gera patches que parecem pixel art de verdade."""
    print("📦 Gerando dataset REALISTA de pixel art...")
    rng = np.random.RandomState(seed)

    palettes = {
        'nes': np.array([[0x7C,0x7C,0x7C],[0x00,0x00,0xFC],[0xA8,0x10,0x00],[0x00,0xB8,0x00],[0xF8,0x38,0x00],[0xFC,0xFC,0xFC],[0x58,0xD8,0x54],[0xF8,0xB8,0x00]],dtype=np.float32)/255.,
        'gameboy': np.array([[0x0f,0x38,0x0f],[0x30,0x62,0x30],[0x8b,0xac,0x0f],[0x9b,0xbc,0x0e]],dtype=np.float32)/255.,
        'dark': np.array([[0x0A,0x0A,0x0A],[0x1A,0x1A,0x1A],[0x2C,0x2C,0x2C],[0x3D,0x3D,0x3D],[0x4F,0x4F,0x4F],[0x80,0x00,0x00],[0xA0,0x20,0x20]],dtype=np.float32)/255.,
        'neon': np.array([[0xFF,0x00,0x6E],[0x00,0xF0,0xFF],[0x39,0xFF,0x14],[0xFF,0xF2,0x00],[0xBC,0x13,0xFE],[0xFF,0x88,0x00]],dtype=np.float32)/255.,
        'cga': np.array([[0,0,0],[0x55,0xFF,0xFF],[0xFF,0x55,0xFF],[0xFF,0xFF,0xFF]],dtype=np.float32)/255.,
        'nature': np.array([[0x2C,0x7B,0x2C],[0x4C,0xA8,0x4C],[0x6B,0xC8,0x6B],[0x8B,0x5A,0x2B],[0xA0,0x82,0x6F],[0xE8,0xC1,0x70]],dtype=np.float32)/255.,
    }
    pal_names = list(palettes.keys())

    X, Y, C = [], [], []

    for i in range(n_samples):
        pal_name = pal_names[i % len(pal_names)]
        pal = palettes[pal_name]
        patch = np.full((PATCH_SIZE, PATCH_SIZE, 3), pal[0], dtype=np.float32)  # fundo
        ptype = rng.randint(0, 12)
        bg = pal[0]
        fg = pal[rng.randint(1, len(pal))]

        if ptype == 0:  # SPRITE SIMÉTRICO (estilo Space Invader)
            half_w = PATCH_SIZE // 2
            for y in range(PATCH_SIZE):
                for x in range(half_w):
                    if rng.random() < 0.45:
                        patch[y, x] = fg
                        patch[y, PATCH_SIZE - 1 - x] = fg

        elif ptype == 1:  # HUMANOIDE SIMPLES (cabeça + corpo)
            # Cabeça
            head_y, head_x = 1, PATCH_SIZE // 2
            head_size = 3
            for dy in range(head_size):
                for dx in range(-head_size//2, head_size//2+1):
                    y, x = head_y + dy, head_x + dx
                    if 0 <= y < PATCH_SIZE and 0 <= x < PATCH_SIZE:
                        patch[y, x] = pal[rng.randint(1, min(4, len(pal)))]
            # Corpo
            body_color = pal[rng.randint(1, min(4, len(pal)))]
            for dy in range(head_size, PATCH_SIZE - 1):
                for dx in range(-1, 2):
                    y, x = head_y + dy, head_x + dx
                    if 0 <= y < PATCH_SIZE and 0 <= x < PATCH_SIZE:
                        patch[y, x] = body_color

        elif ptype == 2:  # TILE DE GRAMA COM TEXTURA
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    if rng.random() < 0.7:
                        patch[y, x] = pal[rng.choice([0, 1, 2]) if len(pal) > 2 else 0]
                    else:
                        patch[y, x] = bg

        elif ptype == 3:  # TILE DE ÁGUA COM ONDAS
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    wave = int((np.sin(x * 0.8 + y * 0.3) + 1) * 1.5) % min(3, len(pal))
                    patch[y, x] = pal[wave]

        elif ptype == 4:  # TILE DE PEDRA/TIJOLOS
            brick_h = 3
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    row = y // brick_h
                    offset = (row % 2) * (PATCH_SIZE // 2)
                    col = (x + offset) % (PATCH_SIZE // 2)
                    if y % brick_h == 0 or col == 0:
                        patch[y, x] = pal[min(3, len(pal)-1)]  # argamassa
                    else:
                        patch[y, x] = pal[rng.randint(0, min(3, len(pal)))]

        elif ptype == 5:  # ITEM: ESPADA
            # Lâmina vertical
            cx = PATCH_SIZE // 2
            for y in range(1, PATCH_SIZE - 3):
                patch[y, cx] = pal[min(5, len(pal)-1)]  # metal
            # Guarda
            patch[PATCH_SIZE-3, cx-1:cx+2] = pal[min(6, len(pal)-1)]
            # Cabo
            patch[PATCH_SIZE-2:PATCH_SIZE, cx] = pal[min(3, len(pal)-1)]

        elif ptype == 6:  # ITEM: POÇÃO
            cx, cy = PATCH_SIZE // 2, PATCH_SIZE // 2
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    d = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if d < 2.5:
                        patch[y, x] = pal[rng.randint(1, min(4, len(pal)))]
                    elif d < 3.2:
                        patch[y, x] = pal[min(5, len(pal)-1)]

        elif ptype == 7:  # ÁRVORE / VEGETAÇÃO
            # Tronco
            cx = PATCH_SIZE // 2
            for y in range(PATCH_SIZE//2, PATCH_SIZE):
                patch[y, cx] = pal[min(3, len(pal)-1)]
            # Copa
            cy = PATCH_SIZE // 3
            for y in range(PATCH_SIZE//2):
                for x in range(PATCH_SIZE):
                    d = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if d < 3 and rng.random() < 0.7:
                        patch[y, x] = pal[rng.choice([0, 1, 2]) if len(pal) > 2 else 0]

        elif ptype == 8:  # NUVEM / CÉU
            sky = pal[0]
            cloud = pal[min(5, len(pal)-1)]
            patch[:] = sky
            cx = rng.randint(2, PATCH_SIZE-2)
            cy = rng.randint(1, PATCH_SIZE-2)
            for _ in range(15):
                y, x = cy + rng.randint(-1, 2), cx + rng.randint(-2, 3)
                if 0 <= y < PATCH_SIZE and 0 <= x < PATCH_SIZE:
                    patch[y, x] = cloud

        elif ptype == 9:  # BORDA/OUTLINE (silhueta com contorno)
            # Primeiro gera forma aleatória
            mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
            cy, cx = PATCH_SIZE//2, PATCH_SIZE//2
            for _ in range(20):
                if 0 <= cy < PATCH_SIZE and 0 <= cx < PATCH_SIZE:
                    mask[cy, cx] = True
                cy += rng.randint(-1, 2); cx += rng.randint(-1, 2)
            # Preenche interior
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    if mask[max(0,y-1):y+2, max(0,x-1):x+2].sum() >= 2:
                        mask[y, x] = True
            # Aplica cores com outline
            outline_color = pal[0]
            inner_color = pal[rng.randint(1, min(4, len(pal)))]
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    if mask[y, x]:
                        # Verifica se é borda
                        is_border = False
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < PATCH_SIZE and 0 <= nx < PATCH_SIZE:
                                if not mask[ny, nx]:
                                    is_border = True; break
                        patch[y, x] = outline_color if is_border else inner_color

        elif ptype == 10:  # GRADIENTE HORIZONTAL COM FAIXAS
            for y in range(PATCH_SIZE):
                t = y / (PATCH_SIZE - 1)
                c1 = pal[rng.randint(0, min(3, len(pal)))]
                c2 = pal[rng.randint(0, min(3, len(pal)))]
                patch[y, :] = c1 * (1 - t) + c2 * t

        else:  # FORMA GEOMÉTRICA: círculo ou diamante
            cx, cy = PATCH_SIZE/2, PATCH_SIZE/2
            shape = rng.choice(['circle', 'diamond', 'triangle'])
            for y in range(PATCH_SIZE):
                for x in range(PATCH_SIZE):
                    if shape == 'circle':
                        d = np.sqrt((x-cx)**2 + (y-cy)**2)
                        if d < 3: patch[y, x] = fg
                    elif shape == 'diamond':
                        if abs(x-cx) + abs(y-cy) < 3: patch[y, x] = fg
                    else:  # triangle
                        if y > cy and abs(x-cx) < (y-cy)*0.8:
                            patch[y, x] = fg

        patch_norm = patch * 2.0 - 1.0
        hist = np.histogram(patch.reshape(-1, 3), bins=16, range=(0, 1))[0]
        hist = hist / (hist.sum() + 1e-8)
        stats = np.array([patch.mean(), patch.std(), patch.min(), patch.max(),
                          np.unique(patch.reshape(-1,3),axis=0).shape[0]/64.0])
        feat = np.concatenate([hist, stats]).astype(np.float32)
        if len(feat) < INPUT_DIM: feat = np.pad(feat, (0, INPUT_DIM - len(feat)))
        cond = np.zeros(CONDITION_DIM, dtype=np.float32)
        pal_idx = pal_names.index(pal_name)
        cond[pal_idx] = 1.0
        cond[10 + pal_idx] = rng.randn() * 0.1
        cond[20 + (ptype % 12)] = 0.5
        X.append(feat[:INPUT_DIM]); Y.append(patch_norm.reshape(-1)); C.append(cond)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(C, dtype=np.float32)

def train(epochs=5000, batch_size=64, save_every=500, seed=42, resume=False):
    print("="*70)
    print("🧠 TREINAMENTO VAE CONDICIONAL - PIXEL ART")
    print("="*70)
    print(f"   LATENT: {LATENT_DIM}d | BETA_KL: {BETA_KL}")
    print("="*70)
    np.random.seed(seed); t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    X, Y, C = generate_realistic_patches(8000, seed=123)
    print(f"   ✓ Dataset: X{X.shape} Y{Y.shape} C{C.shape}")
    model = ConditionalVAE(seed=seed)
    total_params = sum(p.size for p in [
        model.W_enc1, model.b_enc1, model.W_mu, model.b_mu,
        model.W_logvar, model.b_logvar,
        model.W_dec1, model.b_dec1, model.W_dec2, model.b_dec2, model.W_cond])
    print(f"🧮 Parâmetros: {total_params:,}")
    print(f"🎯 Épocas: {epochs} | Batch: {batch_size}")
    start_epoch, losses, best_loss = 1, [], float('inf')
    recon_losses, kl_losses = [], []
    if resume:
        latest = CHECKPOINT_DIR / "latest.npz"
        if latest.exists():
            start_epoch, prev_loss = model.load(latest)
            start_epoch += 1; best_loss = prev_loss
            print(f"📂 Retomando do epoch {start_epoch - 1}")
    for epoch in range(start_epoch, epochs + 1):
        idx = np.random.permutation(len(X))
        ep_loss, ep_recon, ep_kl, n_b = 0.0, 0.0, 0.0, 0
        for i in range(0, len(X), batch_size):
            b_idx = idx[i:i + batch_size]
            loss, recon, kl = model.train_step(X[b_idx], C[b_idx], Y[b_idx])
            ep_loss += loss; ep_recon += recon; ep_kl += kl; n_b += 1
        avg_l = ep_loss / max(1, n_b)
        avg_r = ep_recon / max(1, n_b)
        avg_k = ep_kl / max(1, n_b)
        losses.append(avg_l); recon_losses.append(avg_r); kl_losses.append(avg_k)
        if avg_l < best_loss: best_loss = avg_l
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            eta = dt / epoch * (epochs - epoch)
            print(f"   📊 Ep {epoch:5d}/{epochs} | L={avg_l:.4f} R={avg_r:.4f} KL={avg_k:.4f} | {dt:.0f}s (ETA {eta:.0f}s)")
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = CHECKPOINT_DIR / f"model_epoch_{epoch:05d}.npz"
            model.save(ckpt, epoch, avg_l)
            model.save(CHECKPOINT_DIR / "latest.npz", epoch, avg_l)
            print(f"   💾 Checkpoint: {ckpt.name}")
            gc.collect()
    manifest = {
        "model_type": "conditional_vae",
        "architecture": {"input_dim": INPUT_DIM, "latent_dim": LATENT_DIM, "output_dim": OUTPUT_DIM,
                         "hidden_dim": HIDDEN_DIM, "condition_dim": CONDITION_DIM, "patch_size": PATCH_SIZE,
                         "beta_kl": BETA_KL},
        "total_params": total_params, "epochs": epochs,
        "final_loss": float(losses[-1]) if losses else None,
        "final_recon": float(recon_losses[-1]) if recon_losses else None,
        "final_kl": float(kl_losses[-1]) if kl_losses else None,
        "best_loss": float(best_loss),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_time_seconds": int(time.time() - t0),
        "checkpoint": "latest.npz", "runner": "ubuntu-latest"
    }
    with open(MANIFEST_FILE, "w") as f: json.dump(manifest, f, indent=2)
    log_data = {"losses": losses[::5], "recon_losses": recon_losses[::5],
                "kl_losses": kl_losses[::5], "best_loss": float(best_loss),
                "epochs": epochs, "training_time_seconds": int(time.time() - t0)}
    with open(TRAINING_LOG, "w") as f: json.dump(log_data, f, indent=2)
    print(f"\n✅ VAE completo em {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    if losses:
        print(f"   Loss total: {losses[-1]:.5f}")
        print(f"   Recon: {recon_losses[-1]:.5f} | KL: {kl_losses[-1]:.5f}")
    print(f"   Melhor loss: {best_loss:.5f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="🧠 Treinar VAE Pixel Art")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(a.epochs, a.batch_size, a.save_every, a.seed, a.resume)