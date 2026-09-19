#!/usr/bin/env python3
"""
🎨 GERADOR DE PIXEL ART COM VAE TREINADO
"""
import os, sys, json, hashlib, argparse, random
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

OUTPUT_DIR = Path("pixel_art_output")
MODEL_DIR = Path("models")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
MANIFEST_FILE = MODEL_DIR / "manifest.json"
KNOWLEDGE_FILE = Path("knowledge_base.json")

PATCH_SIZE = 8
INPUT_DIM = 256
LATENT_DIM = 128
OUTPUT_DIM = 192
HIDDEN_DIM = 512
CONDITION_DIM = 64
SEED_MASK = 0xFFFFFFFF

def make_seed(*parts):
    base = int.from_bytes(os.urandom(4), "little")
    for p in parts: base ^= hash(str(p))
    return base & SEED_MASK

PALETTE_NES = ["#7C7C7C","#0000FC","#0000BC","#4428BC","#940084","#A80020","#A81000","#881400","#503000","#007800","#006800","#005800","#004058","#000000","#BCBCBC","#0078F8","#0058F8","#6844FC","#D800CC","#E40058","#F83800","#E45C10","#AC7C00","#00B800","#00A800","#00A844","#008888","#F8F8F8","#3CBCFC","#6888FC","#9878F8","#F878F8","#F85898","#F87858","#FCA044","#F8B800","#B8F818","#58D854","#58F898","#00E8D8","#787878","#FCFCFC","#A4E4FC","#B8B8F8","#D8B8F8","#F8B8F8","#F8A4C0","#F0D0B0","#FCE0A8","#F8D878","#D8F878","#B8F8B8","#B8F8D8","#00FCFC"]
PALETTE_GAMEBOY = ["#0f380f","#306230","#8bac0f","#9bbc0e"]
PALETTE_CGA = ["#000000","#55FFFF","#FF55FF","#FFFFFF"]
BAYER_2x2 = np.array([[0,2],[3,1]],dtype=np.float32)/4.0
BAYER_4x4 = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]],dtype=np.float32)/16.0
BAYER_8x8 = np.array([[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],[12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],[3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],[15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]],dtype=np.float32)/64.0

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def make_palette(name="nes", n=16, seed=None):
    rng = np.random.RandomState(seed)
    palettes = {
        "nes": PALETTE_NES, "gameboy": PALETTE_GAMEBOY, "cga": PALETTE_CGA,
        "monochrome": ["#000000","#555555","#AAAAAA","#FFFFFF"],
        "sepia": ["#3B2414","#6B4423","#A67B5B","#D4A574","#E8C8A0","#F5E6D3","#F8F0E3","#FFFFFF"],
        "pastel": ["#FFD1DC","#FFB7C5","#FDFD96","#B5EAD7","#C7CEEA","#E2F0CB","#FFDAC1","#B5D8EB","#F0E6FF","#FFE4E1","#DCD0FF","#C1E1C1","#FFEBCD","#E0FFFF","#F5DEB3","#FFFACD"],
        "neon": ["#FF006E","#FF4DA6","#FF00FF","#CC00FF","#8B00FF","#00F0FF","#00FFFF","#39FF14","#CCFF00","#FFF200","#FF8C00","#FF1744","#D500F9","#651FFF","#00E5FF","#76FF03"],
        "dark": ["#0A0A0A","#1A1A1A","#2C2C2C","#3D3D3D","#4F4F4F","#1A0000","#2A0000","#3A0000","#00001A","#00002A","#00003A","#1A001A","#2A002A","#3A003A","#0A1A0A","#1A2A1A"],
        "bright": ["#FFFFFF","#FFFFAA","#AAFFFF","#FFAAFF","#AAFFAA","#FFAAAA","#AAAAFF","#FFCC00","#00CCFF","#FF00CC","#00FFCC","#CCFF00","#CC00FF","#00FF00","#FF0000","#0000FF"],
    }
    colors = palettes.get(name, PALETTE_NES)
    n = min(n, len(colors))
    selected = list(colors)
    if n < len(selected):
        rng.shuffle(selected); selected = selected[:n]
    return np.array([hex_to_rgb(c) for c in selected], dtype=np.uint8)

def quantize(img, pal):
    h, w, _ = img.shape
    flat = img.reshape(-1, 3).astype(np.float32)
    p = pal.astype(np.float32)
    dist = ((flat[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    idx = dist.argmin(axis=1)
    return p[idx].astype(np.uint8).reshape(h, w, 3), idx.reshape(h, w)

def dither_bayer(img, pal, matrix):
    h, w, _ = img.shape; mh, mw = matrix.shape
    p = pal.astype(np.float32); out = np.zeros_like(img)
    idx_out = np.zeros((h, w), dtype=np.int32); imgf = img.astype(np.float32)
    for y in range(h):
        for x in range(w):
            pix = np.clip(imgf[y, x] + (matrix[y % mh, x % mw] - 0.5) * 64, 0, 255)
            d = ((p - pix[None, :]) ** 2).sum(axis=1); i = d.argmin()
            out[y, x] = p[i].astype(np.uint8); idx_out[y, x] = i
    return out, idx_out

def dither_floyd(img, pal):
    h, w, _ = img.shape; p = pal.astype(np.float32)
    img = img.astype(np.float32).copy(); out = np.zeros_like(img)
    idx_out = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            old = img[y, x].copy()
            d = ((p - old[None, :]) ** 2).sum(axis=1); i = d.argmin(); new = p[i]
            out[y, x] = new.astype(np.uint8); idx_out[y, x] = i
            err = old - new
            if x + 1 < w: img[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0: img[y + 1, x - 1] += err * 3 / 16
                img[y + 1, x] += err * 5 / 16
                if x + 1 < w: img[y + 1, x + 1] += err * 1 / 16
    return np.clip(out, 0, 255).astype(np.uint8), idx_out

def dither_atkinson(img, pal):
    h, w, _ = img.shape; p = pal.astype(np.float32)
    img = img.astype(np.float32).copy(); out = np.zeros_like(img)
    idx_out = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            old = img[y, x].copy()
            d = ((p - old[None, :]) ** 2).sum(axis=1); i = d.argmin(); new = p[i]
            out[y, x] = new.astype(np.uint8); idx_out[y, x] = i
            err = (old - new) / 8
            if x + 1 < w: img[y, x + 1] += err
            if x + 2 < w: img[y, x + 2] += err
            if y + 1 < h:
                if x - 1 >= 0: img[y + 1, x - 1] += err
                img[y + 1, x] += err
                if x + 1 < w: img[y + 1, x + 1] += err
            if y + 2 < h: img[y + 2, x] += err
    return np.clip(out, 0, 255).astype(np.uint8), idx_out

def apply_dithering(img, pal, method="bayer4x4"):
    if method == "bayer2x2": return dither_bayer(img, pal, BAYER_2x2)
    if method == "bayer4x4": return dither_bayer(img, pal, BAYER_4x4)
    if method == "bayer8x8": return dither_bayer(img, pal, BAYER_8x8)
    if method == "floyd_steinberg": return dither_floyd(img, pal)
    if method == "atkinson": return dither_atkinson(img, pal)
    return quantize(img, pal)

def add_outline(img, color="dark"):
    h, w, _ = img.shape; out = img.copy()
    ol = np.array([0, 0, 0] if color == "dark" else [255, 255, 255], dtype=np.uint8)
    bg = np.array([255, 255, 255], dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if np.array_equal(img[y, x], bg): continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and np.array_equal(img[ny, nx], bg):
                    out[y, x] = ol; break
    return out

def resize_nearest(img, scale=4):
    pil = Image.fromarray(img)
    return np.array(pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST))

class VAEModel:
    def __init__(self):
        self.W_dec1 = None; self.b_dec1 = None
        self.W_dec2 = None; self.b_dec2 = None
        self.loaded = False; self.epoch = 0; self.loss = 0.0
        self.manifest = {}

    def load_latest(self):
        latest = CHECKPOINT_DIR / "latest.npz"
        if not latest.exists():
            ckpts = sorted(CHECKPOINT_DIR.glob("model_epoch_*.npz"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not ckpts:
                print("⚠️ NENHUM VAE TREINADO!")
                self._init_random(); return False
            latest = ckpts[0]
        print(f"📂 Carregando VAE: {latest.name}")
        d = np.load(latest)
        for k in ['W_dec1', 'b_dec1', 'W_dec2', 'b_dec2']:
            if k in d.files: setattr(self, k, d[k])
        self.epoch = int(d['epoch']) if 'epoch' in d.files else 0
        self.loss = float(d['loss']) if 'loss' in d.files else 0.0
        self.loaded = True
        if MANIFEST_FILE.exists():
            try:
                with open(MANIFEST_FILE) as f: self.manifest = json.load(f)
            except Exception: pass
        print(f"   ✅ VAE: epoch {self.epoch}, loss {self.loss:.5f}")
        return True

    def _init_random(self):
        rng = np.random.RandomState(42)
        s = np.sqrt(2.0 / (LATENT_DIM + HIDDEN_DIM))
        self.W_dec1 = (rng.randn(LATENT_DIM, HIDDEN_DIM) * s).astype(np.float32)
        self.b_dec1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.W_dec2 = (rng.randn(HIDDEN_DIM, OUTPUT_DIM) * s).astype(np.float32)
        self.b_dec2 = np.zeros(OUTPUT_DIM, dtype=np.float32)
        self.loaded = False

    def decode(self, z):
        h1 = np.maximum(0.0, z @ self.W_dec1 + self.b_dec1)
        return np.tanh(h1 @ self.W_dec2 + self.b_dec2)

    def sample_prior(self, seed=None):
        safe_seed = int(seed) & SEED_MASK if seed is not None else 0
        rng = np.random.RandomState(safe_seed)
        return rng.randn(1, LATENT_DIM).astype(np.float32)

    def generate_patch(self, noise_scale=1.0, seed=None):
        z = self.sample_prior(seed) * noise_scale
        z = np.clip(z, -3.0, 3.0)
        out = self.decode(z)
        patch = out.reshape(PATCH_SIZE, PATCH_SIZE, 3)
        patch = (patch + 1.0) / 2.0
        return np.clip(patch, 0.0, 1.0)

    def generate_image(self, width, height, noise_scale=1.0, seed=None, coherence=0.3):
        base_seed = int(seed) & SEED_MASK if seed is not None else make_seed()
        n_px = max(1, width // PATCH_SIZE)
        n_py = max(1, height // PATCH_SIZE)
        img = np.zeros((height, width, 3), dtype=np.float32)
        rng_base = np.random.RandomState(base_seed)
        z_base = rng_base.randn(1, LATENT_DIM).astype(np.float32)
        for py in range(n_py):
            for px in range(n_px):
                patch_seed = make_seed(base_seed, py, px)
                rng_p = np.random.RandomState(patch_seed)
                noise = rng_p.randn(1, LATENT_DIM).astype(np.float32) * 0.3
                z = (z_base * coherence + noise * (1 - coherence)) * noise_scale
                z = np.clip(z, -3.0, 3.0)
                out = self.decode(z)
                patch = out.reshape(PATCH_SIZE, PATCH_SIZE, 3)
                patch = (patch + 1.0) / 2.0
                patch = np.clip(patch, 0.0, 1.0)
                y0, x0 = py * PATCH_SIZE, px * PATCH_SIZE
                y1 = min(y0 + PATCH_SIZE, height)
                x1 = min(x0 + PATCH_SIZE, width)
                img[y0:y1, x0:x1] = patch[:y1 - y0, :x1 - x0]
        return (img * 255).astype(np.uint8)

KEYWORDS = {
    "nes": {"style": "nes_8bit", "palette": "nes", "palette_size": 16, "dithering": "bayer2x2"},
    "8-bit": {"style": "nes_8bit", "palette": "nes", "palette_size": 16},
    "8bit": {"style": "nes_8bit", "palette": "nes", "palette_size": 16},
    "snes": {"style": "snes_16bit", "palette": "nes", "palette_size": 32},
    "16-bit": {"style": "snes_16bit", "palette": "nes", "palette_size": 32},
    "gameboy": {"style": "gameboy", "palette": "gameboy", "palette_size": 4},
    "game boy": {"style": "gameboy", "palette": "gameboy", "palette_size": 4},
    "cga": {"style": "cga", "palette": "cga", "palette_size": 4},
    "retrô": {"style": "nes_8bit"}, "retro": {"style": "nes_8bit"},
    "cyberpunk": {"style": "cyberpunk_pixel", "palette": "neon"},
    "steampunk": {"style": "steampunk_pixel", "palette": "sepia"},
    "medieval": {"style": "medieval_pixel"}, "anime": {"style": "anime_pixel"},
    "horror": {"style": "horror_pixel", "palette": "dark"},
    "fantasia": {"style": "fantasy_pixel"}, "espacial": {"style": "space_pixel"},
    "neon": {"palette": "neon", "style": "neon_pixel"}, "pastel": {"palette": "pastel"},
    "monocromático": {"palette": "monochrome", "palette_size": 4},
    "monocromatico": {"palette": "monochrome", "palette_size": 4},
    "escuro": {"palette": "dark"}, "claro": {"palette": "bright"},
    "personagem": {"subject": "character"}, "herói": {"subject": "hero"},
    "dragão": {"subject": "dragon"}, "slime": {"subject": "slime"},
    "floresta": {"scene": "forest"}, "dungeon": {"scene": "dungeon"},
    "cidade": {"scene": "city"}, "espaço": {"scene": "space"},
    "deserto": {"scene": "desert"}, "montanha": {"scene": "mountain"},
    "subaquático": {"scene": "underwater"}, "mar": {"scene": "water"},
    "tileset": {"category": "tileset"}, "terreno": {"scene": "terrain"},
    "água": {"scene": "water"}, "agua": {"scene": "water"}, "noite": {"palette": "dark"},
}

def interpret_prompt(prompt):
    cfg = {"style": "modern_pixel", "palette": "nes", "palette_size": 16,
           "dithering": "bayer4x4", "outline": "none", "subject": None,
           "scene": None, "category": None, "tags": []}
    pl = prompt.lower()
    for key in sorted(KEYWORDS.keys(), key=len, reverse=True):
        if key in pl:
            for k, v in KEYWORDS[key].items():
                if v is not None: cfg[k] = v
            cfg["tags"].append(key)
    return cfg

class RAG:
    def __init__(self, path=KNOWLEDGE_FILE):
        self.entries = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f).get("entries", [])
            except Exception: pass
    def retrieve(self, query, top_k=3):
        if not self.entries: return []
        qt = set(query.lower().replace(",", " ").replace("-", " ").split())
        scored = []
        for e in self.entries:
            s = len(qt & set(t.lower() for t in e.get("tags", []))) * 3
            if e.get("style", "").lower() in query.lower(): s += 5
            if s > 0: scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

def get_next_number():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nums = [int(f.stem) for f in OUTPUT_DIR.glob("*.png") if f.stem.isdigit()]
    return max(nums) + 1 if nums else 1

def save(img, meta, scale=4):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num = get_next_number()
    Image.fromarray(img).save(OUTPUT_DIR / f"{num}.png")
    if scale > 1:
        Image.fromarray(resize_nearest(img, scale)).save(OUTPUT_DIR / f"{num}_x{scale}.png")
    with open(OUTPUT_DIR / f"{num}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    meta["number"] = num; meta["path"] = str(OUTPUT_DIR / f"{num}.png")
    return meta

def generate_pixel_art(prompt, width=64, height=64, style=None, palette_size=16,
                       dithering="bayer4x4", outline="none", scale=4, seed=None,
                       noise_scale=1.0, coherence=0.3, model=None):
    print(f"\n🎨 Gerando: '{prompt}'")
    if model is None:
        model = VAEModel(); model.load_latest()
    cfg = interpret_prompt(prompt)
    if style: cfg["style"] = style
    cfg["palette_size"] = palette_size
    cfg["dithering"] = dithering; cfg["outline"] = outline
    rag = RAG(); matches = rag.retrieve(prompt, top_k=3)
    if matches:
        top = matches[0]
        if not style and top.get("style"): cfg["style"] = top["style"]
        if top.get("palette"): cfg["palette"] = top["palette"]
        print(f"   📚 RAG: {[m.get('style') for m in matches]}")
    if seed is None: seed = make_seed(prompt)
    else: seed = int(seed) & SEED_MASK
    print(f"   🧠 VAE: epoch {model.epoch}, loss {model.loss:.4f}, treinado={model.loaded}")
    print(f"   🎲 Seed: {seed} | Noise: {noise_scale} | Coherence: {coherence}")
    base_img = model.generate_image(width, height, noise_scale, seed, coherence)
    pal = make_palette(cfg["palette"], cfg["palette_size"], seed)
    dithered, _ = apply_dithering(base_img, pal, cfg["dithering"])
    if cfg["outline"] not in (None, "none"):
        dithered = add_outline(dithered, cfg["outline"])
    meta = {
        "prompt": prompt, "config": {k: v for k, v in cfg.items() if k != "tags"},
        "tags": cfg["tags"], "rag_matches": [m.get("style") for m in matches],
        "model_epoch": model.epoch, "model_loss": model.loss,
        "model_trained": model.loaded, "seed": seed,
        "noise_scale": noise_scale, "coherence": coherence,
        "palette_name": cfg["palette"], "palette_size": int(cfg["palette_size"]),
        "dithering": cfg["dithering"], "outline": cfg["outline"],
        "timestamp": datetime.now().isoformat(),
        "resolution": [width, height], "scale": scale,
    }
    saved = save(dithered, meta, scale)
    print(f"   ✅ Salvo: {saved['path']}")
    return saved

def batch_generate(prompt, count=1, **kw):
    model = VAEModel(); model.load_latest()
    kw.pop('seed', None)
    results = []
    for i in range(count):
        s = make_seed(prompt, i)
        results.append(generate_pixel_art(prompt, seed=s, model=model, **kw))
    return results

def main():
    p = argparse.ArgumentParser(description="🎨 Gerador Pixel Art (VAE)")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--prompt", type=str)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--style", type=str, default=None)
    p.add_argument("--palette-size", type=int, default=16)
    p.add_argument("--dithering", type=str, default="bayer4x4")
    p.add_argument("--outline", type=str, default="none")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--coherence", type=float, default=0.3)
    args = p.parse_args()
    print("=" * 60); print("🎨 PIXEL ART VAE - MODELO TREINADO"); print("=" * 60)
    if args.batch:
        if not args.prompt: print("❌ --prompt obrigatório"); sys.exit(1)
        batch_generate(args.prompt, count=args.count, width=args.width, height=args.height,
                       style=args.style, palette_size=args.palette_size, dithering=args.dithering,
                       outline=args.outline, scale=args.scale, seed=args.seed,
                       noise_scale=args.noise_scale, coherence=args.coherence)
        print(f"\n✅ {args.count} pixel arts em {OUTPUT_DIR}/"); return
    model = VAEModel(); model.load_latest()
    while True:
        print("\n1.Gerar  2.Aleatório  3.Batch(5)  4.Listar  5.Info  6.Sair")
        op = input("Opção: ").strip()
        if op == "1":
            pr = input("Prompt: ").strip()
            if pr: generate_pixel_art(pr, model=model)
        elif op == "2":
            generate_pixel_art(random.choice([
                "herói estilo NES", "dragão 16-bit", "floresta Game Boy",
                "nave cyberpunk", "dungeon escura", "cidade neon"
            ]), model=model)
        elif op == "3":
            pr = input("Prompt: ").strip()
            if pr: batch_generate(pr, count=5)
        elif op == "4":
            if OUTPUT_DIR.exists():
                for f in sorted(OUTPUT_DIR.glob("*.png"))[-10:]: print(f"  - {f.name}")
        elif op == "5":
            print(f"   VAE: {model.loaded} | Epoch: {model.epoch} | Loss: {model.loss:.5f}")
        elif op == "6": break

if __name__ == "__main__":
    main()
