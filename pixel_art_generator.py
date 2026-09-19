#!/usr/bin/env python3
"""
GERADOR DE PIXEL ART COM VAE TREINADO v2.0

Melhorias sobre v1.0:
- Dimensoes nao-multiplas de PATCH_SIZE agora funcionam (math.ceil)
- Overlap real com blending linear (acumulador + pesos)
- interpret_prompt estruturado usando config
- Paletas sincronizadas com treino
- Checagem de arch_version ao carregar checkpoint
"""
import os, sys, json, hashlib, argparse, random, math
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

import config as C

OUTPUT_DIR = Path("pixel_art_output")
MODEL_DIR = Path("models")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
MANIFEST_FILE = MODEL_DIR / "manifest.json"
KNOWLEDGE_FILE = Path("knowledge_base.json")

SEED_MASK = 0xFFFFFFFF


def make_seed(*parts):
    base = int.from_bytes(os.urandom(4), "little")
    for p in parts: base ^= hash(str(p))
    return base & SEED_MASK


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def make_palette(name="nes", n=16, seed=None):
    rng = np.random.RandomState(seed)
    if name not in C.PALETTES_HEX:
        print(f"   ⚠️ Paleta '{name}' desconhecida, usando 'nes'")
        name = "nes"
    colors = C.PALETTES_HEX[name]
    n = min(n, len(colors))
    selected = list(colors)
    if n < len(selected):
        rng.shuffle(selected); selected = selected[:n]
    return np.array([hex_to_rgb(c) for c in selected], dtype=np.uint8)


BAYER_2x2 = np.array([[0,2],[3,1]],dtype=np.float32)/4.0
BAYER_4x4 = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]],dtype=np.float32)/16.0
BAYER_8x8 = np.array([[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],
                      [12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
                      [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],
                      [15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]],dtype=np.float32)/64.0


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
        best = CHECKPOINT_DIR / "best.npz"
        chosen = None
        if best.exists():
            chosen = best
        elif latest.exists():
            chosen = latest
        else:
            ckpts = sorted(CHECKPOINT_DIR.glob("model_epoch_*.npz"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not ckpts:
                print("⚠️ NENHUM VAE TREINADO!")
                self._init_random(); return False
            chosen = ckpts[0]
        print(f"📂 Carregando VAE: {chosen.name}")
        d = np.load(chosen, allow_pickle=False)
        # Checa versao da arquitetura
        if 'arch_version' in d.files:
            arch = bytes(d['arch_version']).decode('utf-8')
            if arch != C.ARCH_VERSION:
                print(f"❌ Checkpoint arch_version={arch}, codigo atual e {C.ARCH_VERSION}.")
                print("   Arquitetura mudou - necessario treinar novamente.")
                self._init_random(); return False
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
        s = np.sqrt(2.0 / (C.LATENT_DIM + C.HIDDEN_DIM))
        self.W_dec1 = (rng.randn(C.LATENT_DIM, C.HIDDEN_DIM) * s).astype(np.float32)
        self.b_dec1 = np.zeros(C.HIDDEN_DIM, dtype=np.float32)
        self.W_dec2 = (rng.randn(C.HIDDEN_DIM, C.OUTPUT_DIM) * s).astype(np.float32)
        self.b_dec2 = np.zeros(C.OUTPUT_DIM, dtype=np.float32)
        self.loaded = False

    def decode(self, z):
        h1 = np.maximum(0.0, z @ self.W_dec1 + self.b_dec1)
        return np.tanh(h1 @ self.W_dec2 + self.b_dec2)

    def sample_prior(self, seed=None):
        safe_seed = int(seed) & SEED_MASK if seed is not None else 0
        rng = np.random.RandomState(safe_seed)
        return rng.randn(1, C.LATENT_DIM).astype(np.float32)

    def generate_patch(self, noise_scale=1.0, seed=None):
        z = self.sample_prior(seed) * noise_scale
        z = np.clip(z, -3.0, 3.0)
        out = self.decode(z)
        patch = out.reshape(C.PATCH_SIZE, C.PATCH_SIZE, 3)
        patch = (patch + 1.0) / 2.0
        return np.clip(patch, 0.0, 1.0)

    def generate_image(self, width, height, noise_scale=1.0, seed=None,
                       coherence=0.3, overlap=2):
        """Gera imagem com overlap real e blending linear."""
        base_seed = int(seed) & SEED_MASK if seed is not None else make_seed()
        # Usa ceil para suportar dimensoes nao-multiplas de PATCH_SIZE
        n_px = max(1, math.ceil(width / C.PATCH_SIZE))
        n_py = max(1, math.ceil(height / C.PATCH_SIZE))
        # stride real com overlap
        stride_x = max(1, C.PATCH_SIZE - overlap)
        stride_y = max(1, C.PATCH_SIZE - overlap)
        img_acc = np.zeros((height, width, 3), dtype=np.float32)
        weight_acc = np.zeros((height, width), dtype=np.float32)

        # Matriz de pesos triangular para o blend
        w1d = np.linspace(0.1, 1.0, C.PATCH_SIZE, dtype=np.float32)
        # peso maior no centro, menor nas bordas
        w1d = 0.5 + 0.5 * np.sin(np.linspace(0, np.pi, C.PATCH_SIZE))
        weight_mask = np.outer(w1d, w1d)  # 2D

        rng_base = np.random.RandomState(base_seed)
        z_base = rng_base.randn(1, C.LATENT_DIM).astype(np.float32)

        for py in range(n_py):
            for px in range(n_px):
                y0 = py * stride_y
                x0 = px * stride_x
                if y0 >= height: y0 = max(0, height - C.PATCH_SIZE)
                if x0 >= width: x0 = max(0, width - C.PATCH_SIZE)
                y1 = min(y0 + C.PATCH_SIZE, height)
                x1 = min(x0 + C.PATCH_SIZE, width)
                ph = y1 - y0
                pw = x1 - x0

                patch_seed = make_seed(base_seed, py, px)
                rng_p = np.random.RandomState(patch_seed)
                noise = rng_p.randn(1, C.LATENT_DIM).astype(np.float32) * 0.3
                z = (z_base * coherence + noise * (1 - coherence)) * noise_scale
                z = np.clip(z, -3.0, 3.0)
                out = self.decode(z)
                patch = out.reshape(C.PATCH_SIZE, C.PATCH_SIZE, 3)
                patch = (patch + 1.0) / 2.0
                patch = np.clip(patch, 0.0, 1.0)

                patch_crop = patch[:ph, :pw, :]
                w_crop = weight_mask[:ph, :pw]

                img_acc[y0:y1, x0:x1] += patch_crop * w_crop[:, :, None]
                weight_acc[y0:y1, x0:x1] += w_crop

        weight_acc = np.maximum(weight_acc, 1e-6)
        img = img_acc / weight_acc[:, :, None]
        return (img * 255).astype(np.uint8)


def build_condition_from_prompt(cfg, seed):
    """Constroi vetor de condicao a partir de cfg estruturado."""
    cond = np.zeros(C.CONDITION_DIM, dtype=np.float32)
    # Paleta
    pal_name = cfg.get("palette", "nes")
    if pal_name in C.PALETTE_NAMES:
        pal_idx = C.PALETTE_NAMES.index(pal_name)
        cond[C.COND_SLICE_PAL][pal_idx] = 1.0
    else:
        cond[C.COND_SLICE_PAL][0] = 1.0  # nes default
    # Tipo
    ptype = cfg.get("pixel_art_type")
    if ptype and ptype in C.PIXEL_ART_TYPE_TO_IDX:
        cond[C.COND_SLICE_TYPE][C.PIXEL_ART_TYPE_TO_IDX[ptype]] = 1.0
    # Humor
    mood = cfg.get("mood")
    if mood and mood in C.MOOD_TO_IDX:
        cond[C.COND_SLICE_MOOD][C.MOOD_TO_IDX[mood]] = 1.0
    # Traits
    for trait in cfg.get("traits", []):
        if trait in C.TRAIT_TO_IDX:
            cond[C.COND_SLICE_TRAIT][C.TRAIT_TO_IDX[trait]] = 1.0
    # Features continuas a partir do prompt (hash deterministico)
    h = int(hashlib.md5(cfg.get("raw_prompt", "").encode()).hexdigest()[:8], 16)
    feat_rng = np.random.RandomState(h ^ (seed & 0xFFFF))
    cond[C.COND_SLICE_FEAT] = feat_rng.randn(C.COND_SLICE_FEAT.stop - C.COND_SLICE_FEAT.start).astype(np.float32) * 0.1
    return cond


# ============================================================
# INTERPRETADOR DE PROMPTS ESTRUTURADO
# ============================================================
KEYWORDS = {
    # Paletas
    "nes": {"palette": "nes"}, "8-bit": {"palette": "nes"}, "8bit": {"palette": "nes"},
    "snes": {"palette": "nes"}, "16-bit": {"palette": "nes"},
    "gameboy": {"palette": "gameboy"}, "game boy": {"palette": "gameboy"},
    "cga": {"palette": "cga"},
    "neon": {"palette": "neon", "mood": "neon"},
    "dark": {"palette": "dark", "mood": "dark"},
    "escuro": {"palette": "dark", "mood": "dark"},
    "nature": {"palette": "nature", "mood": "nature"},
    "natureza": {"palette": "nature", "mood": "nature"},
    "monocromático": {"palette": "monochrome", "palette_size": 4},
    "monocromatico": {"palette": "monochrome", "palette_size": 4},
    "claro": {"palette": "bright"}, "bright": {"palette": "bright"},
    "pastel": {"palette": "pastel"},
    "sepia": {"palette": "sepia"},
    # Tipos
    "personagem": {"pixel_art_type": "character"}, "character": {"pixel_art_type": "character"},
    "heroi": {"pixel_art_type": "character"}, "herói": {"pixel_art_type": "character"},
    "monstro": {"pixel_art_type": "character"}, "inimigo": {"pixel_art_type": "character"},
    "dragão": {"pixel_art_type": "character"}, "dragao": {"pixel_art_type": "character"},
    "slime": {"pixel_art_type": "character"}, "goblin": {"pixel_art_type": "character"},
    "sprite": {"pixel_art_type": "sprite"},
    "grama": {"pixel_art_type": "grass"}, "terreno": {"pixel_art_type": "grass"},
    "agua": {"pixel_art_type": "water"}, "água": {"pixel_art_type": "water"},
    "mar": {"pixel_art_type": "water"}, "oceano": {"pixel_art_type": "water"},
    "pedra": {"pixel_art_type": "stone"}, "tijolo": {"pixel_art_type": "stone"},
    "dungeon": {"pixel_art_type": "stone", "mood": "dark"},
    "espada": {"pixel_art_type": "sword"},
    "pocao": {"pixel_art_type": "potion"}, "poção": {"pixel_art_type": "potion"},
    "arvore": {"pixel_art_type": "tree"}, "árvore": {"pixel_art_type": "tree"},
    "floresta": {"pixel_art_type": "tree", "palette": "nature"},
    "nuvem": {"pixel_art_type": "cloud"}, "ceu": {"pixel_art_type": "cloud"},
    "outline": {"pixel_art_type": "outline", "traits": ["outline"]},
    # Moods
    "cyberpunk": {"mood": "neon", "palette": "neon"},
    "medieval": {"mood": "medieval"},
    "horror": {"mood": "dark", "palette": "dark"},
    "retrô": {"mood": "retro"}, "retro": {"mood": "retro"},
    # Traits
    "brilhante": {"traits": ["bright", "glow"]},
    "pequeno": {"traits": ["small"]}, "grande": {"traits": ["large"]},
    "simétrico": {"traits": ["symmetric"]}, "simetrico": {"traits": ["symmetric"]},
    "texturizado": {"traits": ["textured"]},
    "molhado": {"traits": ["wet"]},
}


def interpret_prompt(prompt):
    """Interpreta prompt e retorna cfg estruturado."""
    cfg = {
        "raw_prompt": prompt,
        "style": "modern_pixel",
        "palette": "nes",
        "palette_size": 16,
        "dithering": "bayer4x4",
        "outline": "none",
        "pixel_art_type": None,
        "mood": None,
        "traits": [],
        "tags": [],
    }
    pl = prompt.lower()
    traits_set = set()
    for key in sorted(KEYWORDS.keys(), key=len, reverse=True):
        if key in pl:
            for k, v in KEYWORDS[key].items():
                if k == "traits":
                    traits_set.update(v)
                elif v is not None:
                    cfg[k] = v
            cfg["tags"].append(key)
    cfg["traits"] = list(traits_set)
    # Heuristicas de fallback baseadas em tags
    if "dragão" in pl or "dragon" in pl:
        cfg["pixel_art_type"] = "character"
        cfg["traits"] = list(set(cfg["traits"] + ["symmetric", "outline"]))
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
                       noise_scale=1.0, coherence=0.3, overlap=2, model=None):
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
    print(f"   🎲 Seed: {seed} | Noise: {noise_scale} | Coherence: {coherence} | Overlap: {overlap}")
    print(f"   🎨 Palette: {cfg['palette']} | Type: {cfg['pixel_art_type']} | Mood: {cfg['mood']}")
    base_img = model.generate_image(width, height, noise_scale, seed, coherence, overlap)
    pal = make_palette(cfg["palette"], cfg["palette_size"], seed)
    dithered, _ = apply_dithering(base_img, pal, cfg["dithering"])
    if cfg["outline"] not in (None, "none"):
        dithered = add_outline(dithered, cfg["outline"])
    meta = {
        "prompt": prompt, "config": {k: v for k, v in cfg.items() if k != "tags"},
        "tags": cfg["tags"], "rag_matches": [m.get("style") for m in matches],
        "model_epoch": model.epoch, "model_loss": model.loss,
        "model_trained": model.loaded, "seed": seed,
        "noise_scale": noise_scale, "coherence": coherence, "overlap": overlap,
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
    p = argparse.ArgumentParser(description="🎨 Gerador Pixel Art (VAE v" + C.ARCH_VERSION + ")")
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
    p.add_argument("--overlap", type=int, default=2, help="Overlap entre patches (0-4)")
    args = p.parse_args()
    print("=" * 60); print("🎨 PIXEL ART VAE v" + C.ARCH_VERSION); print("=" * 60)
    if args.batch:
        if not args.prompt: print("❌ --prompt obrigatorio"); sys.exit(1)
        batch_generate(args.prompt, count=args.count, width=args.width, height=args.height,
                       style=args.style, palette_size=args.palette_size, dithering=args.dithering,
                       outline=args.outline, scale=args.scale, seed=args.seed,
                       noise_scale=args.noise_scale, coherence=args.coherence,
                       overlap=args.overlap)
        print(f"\n✅ {args.count} pixel arts em {OUTPUT_DIR}/"); return
    model = VAEModel(); model.load_latest()
    while True:
        print("\n1.Gerar  2.Aleatorio  3.Batch(5)  4.Listar  5.Info  6.Sair")
        op = input("Opcao: ").strip()
        if op == "1":
            pr = input("Prompt: ").strip()
            if pr: generate_pixel_art(pr, model=model)
        elif op == "2":
            generate_pixel_art(random.choice([
                "heroi estilo NES", "dragao 16-bit", "floresta Game Boy",
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
