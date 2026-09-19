#!/usr/bin/env python3
"""
🎨 PIXEL ART GENERATOR CORRIGIDO
- Decoder usa condição REAL (z + cond)
- Prompt → condição via interpret_prompt()
- Coerência global via condição compartilhada
- Carregamento consistente com treino
"""
import os, json, time, argparse, sys
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("⚠️ Instalando Pillow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

MODEL_DIR = Path("models")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
MANIFEST_FILE = MODEL_DIR / "manifest.json"
OUTPUT_DIR = Path("pixel_art_output")

INPUT_DIM = 256
LATENT_DIM = 128
OUTPUT_DIM = 192
HIDDEN_DIM = 512
CONDITION_DIM = 64
PATCH_SIZE = 8
SEED_MASK = 0xFFFFFFFF

# [PALETTES e KEYWORDS definidos aqui - omitido por brevidade]
# Use o arquivo completo do repositório para a versão completa

PALETTES = {
    'nes': np.array([[0x7C,0x7C,0x7C],[0x00,0x00,0xFC],[0xA8,0x10,0x00],[0x00,0xB8,0x00],
                     [0xF8,0x38,0x00],[0xFC,0xFC,0xFC],[0x58,0xD8,0x54],[0xF8,0xB8,0x00]],
                    dtype=np.float32)/255.,
    'gameboy': np.array([[0x0f,0x38,0x0f],[0x30,0x62,0x30],[0x8b,0xac,0x0f],[0x9b,0xbc,0x0e]],
                        dtype=np.float32)/255.,
    'dark': np.array([[0x0A,0x0A,0x0A],[0x1A,0x1A,0x1A],[0x2C,0x2C,0x2C],[0x3D,0x3D,0x3D],
                     [0x4F,0x4F,0x4F],[0x80,0x00,0x00],[0xA0,0x20,0x20]],
                    dtype=np.float32)/255.,
    'neon': np.array([[0xFF,0x00,0x6E],[0x00,0xF0,0xFF],[0x39,0xFF,0x14],[0xFF,0xF2,0x00],
                     [0xBC,0x13,0xFE],[0xFF,0x88,0x00]],
                    dtype=np.float32)/255.,
}

PALETTE_NAMES = list(PALETTES.keys())

def make_seed(*args):
    h = 0
    for a in args:
        if isinstance(a, str):
            for c in a:
                h = (h * 31 + ord(c)) & SEED_MASK
        else:
            h = (h * 31 + int(a)) & SEED_MASK
    return h

def interpret_prompt(prompt):
    """Converte prompt em vetor de condição REAL"""
    if not prompt:
        return None, {}
    
    prompt_lower = prompt.lower()
    
    # Detectar paleta e tipo baseado em keywords
    palette_name = "nes"  # default
    if "dark" in prompt_lower or "sombrio" in prompt_lower:
        palette_name = "dark"
    elif "neon" in prompt_lower or "cyberpunk" in prompt_lower:
        palette_name = "neon"
    elif "gameboy" in prompt_lower:
        palette_name = "gameboy"
    
    ptype = 0  # default: sprite simétrico
    if "personagem" in prompt_lower or "herói" in prompt_lower:
        ptype = 1
    elif "espada" in prompt_lower or "item" in prompt_lower:
        ptype = 5
    elif "grama" in prompt_lower or "tile" in prompt_lower:
        ptype = 2
    
    cond = np.zeros(CONDITION_DIM, dtype=np.float32)
    pal_idx = PALETTE_NAMES.index(palette_name) if palette_name in PALETTE_NAMES else 0
    cond[pal_idx] = 1.0
    cond[20 + (ptype % 12)] = 0.5
    
    metadata = {'palette': palette_name, 'ptype': ptype, 'prompt': prompt}
    return cond, metadata

class VAEModel:
    def __init__(self):
        self.W_dec1 = None
        self.b_dec1 = None
        self.W_dec2 = None
        self.b_dec2 = None
        self.W_cond = None
        self.loaded = False
    
    def load_latest(self):
        latest = CHECKPOINT_DIR / "latest.npz"
        if not latest.exists():
            print("⚠️ NENHUM VAE TREINADO!")
            return False
        
        print(f"📂 Carregando VAE: {latest.name}")
        d = np.load(latest, allow_pickle=True)
        
        for k in ['W_dec1', 'b_dec1', 'W_dec2', 'b_dec2', 'W_cond']:
            if k in d.files:
                setattr(self, k, d[k])
        
        self.loaded = True
        print(f"   ✅ VAE carregado")
        return True
    
    def decode(self, z, cond):
        """Decoder CORRIGIDO: (z, cond) -> output"""
        if not self.loaded:
            raise RuntimeError("Modelo não carregado!")
        
        cond_proj = cond @ self.W_cond
        z_cond = np.concatenate([z, cond_proj], axis=-1)
        h1 = np.maximum(0.0, z_cond @ self.W_dec1 + self.b_dec1)
        out = np.tanh(h1 @ self.W_dec2 + self.b_dec2)
        return out
    
    def sample_prior(self, seed=None):
        safe_seed = int(seed) & SEED_MASK if seed is not None else 0
        rng = np.random.RandomState(safe_seed)
        return rng.randn(1, LATENT_DIM).astype(np.float32)
    
    def generate_image(self, width, height, cond, noise_scale=1.0, seed=None,
                      coherence=0.7, overlap=2):
        base_seed = int(seed) & SEED_MASK if seed is not None else make_seed(cond.tobytes())
        rng_base = np.random.RandomState(base_seed)
        z_base = rng_base.randn(1, LATENT_DIM).astype(np.float32)
        
        img = np.zeros((height, width, 3), dtype=np.float32)
        n_px = max(1, width // PATCH_SIZE)
        n_py = max(1, height // PATCH_SIZE)
        
        for py in range(n_py):
            for px in range(n_px):
                patch_seed = make_seed(base_seed, py, px)
                rng_p = np.random.RandomState(patch_seed)
                noise = rng_p.randn(1, LATENT_DIM).astype(np.float32) * 0.3
                
                z = (z_base * coherence + noise * (1 - coherence)) * noise_scale
                z = np.clip(z, -3.0, 3.0)
                
                out = self.decode(z, cond.reshape(1, -1))
                patch = out.reshape(PATCH_SIZE, PATCH_SIZE, 3)
                patch = (patch + 1.0) / 2.0
                patch = np.clip(patch, 0.0, 1.0)
                
                y0, x0 = py * PATCH_SIZE, px * PATCH_SIZE
                y1 = min(y0 + PATCH_SIZE, height)
                x1 = min(x0 + PATCH_SIZE, width)
                img[y0:y1, x0:x1] = patch[:y1 - y0, :x1 - x0]
        
        return (img * 255).astype(np.uint8)

def batch_generate(args):
    print("="*70)
    print("🎨 GERAÇÃO PIXEL ART (MODO BATCH)")
    print("="*70)
    
    model = VAEModel()
    if not model.load_latest():
        sys.exit(1)
    
    cond, meta = interpret_prompt(args.prompt)
    
    if cond is None:
        print("⚠️ Prompt vazio, usando condição aleatória")
        cond = np.zeros(CONDITION_DIM, dtype=np.float32)
        cond[0] = 1.0
    
    for i in range(args.count):
        seed = make_seed(args.prompt, i, int(time.time()))
        
        print(f"\n🎨 Gerando imagem {i+1}/{args.count}...")
        img = model.generate_image(
            width=args.width,
            height=args.height,
            cond=cond,
            noise_scale=args.noise_scale,
            seed=seed,
            coherence=args.coherence,
            overlap=args.overlap
        )
        
        filename = f"{int(time.time())}_{i}.png"
        output_path = OUTPUT_DIR / filename
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(output_path)
        print(f"💾 Salvo: {output_path}")
    
    print(f"\n✅ {args.count} imagens geradas!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎨 Gerar Pixel Art")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--coherence", type=float, default=0.7)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--batch", action="store_true")
    
    args = parser.parse_args()
    
    if args.batch or args.prompt:
        batch_generate(args)