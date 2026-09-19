#!/usr/bin/env python3
"""
🎨 PIXEL ART AI GENERATOR
Gera pixel art procedural baseada em prompt usando paletas reais de consoles clássicos.
Autor: gerador procedural (numpy + scipy + pillow)
"""
import argparse
import hashlib
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image
from scipy import ndimage

# ============================================================
# PALETAS DE ESTILOS (cores reais de hardware clássico)
# ============================================================
STYLES = {
    'gameboy': ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'],
    'gameboy-pocket': ['#000000', '#555555', '#aaaaaa', '#ffffff'],
    'gameboy-light': ['#2b2b2b', '#5a5a5a', '#9a9a9a', '#dadada'],
    'nes': [
        '#7c7c7c','#0000fc','#0000bc','#4028bc','#940084','#a80020','#a81000','#881400',
        '#503000','#007800','#006800','#005800','#004058','#000000','#000000','#000000',
        '#bcbcbc','#0078f8','#0058f8','#6844fc','#d800cc','#e40058','#f83800','#e45c10',
        '#ac7c00','#00b800','#00a800','#00a844','#008888','#000000','#000000','#000000',
        '#f8f8f8','#3cbcfc','#6888fc','#9878f8','#f878f8','#f85898','#f87858','#fca044',
        '#f8b800','#b8f818','#58d854','#58f898','#00e8d8','#787878','#000000','#000000',
        '#fcfcfc','#a4e4fc','#b8b8f8','#d8b8f8','#f8b8f8','#f8a4c0','#f0d0b0','#fce0a8',
        '#f8d878','#d8f878','#b8f8b8','#b8f8d8','#00fcfc','#f8d8f8','#000000','#000000',
    ],
    'snes': ['#000000','#1a1a2e','#16213e','#0f3460','#533483','#e94560','#f5b461','#f5d76e'],
    'pico8': ['#000000','#1d2b53','#7e2553','#008751','#ab5236','#5f574f',
              '#c2c3c7','#fff1e8','#ff004d','#ffa300','#ffec27','#00e436',
              '#29adff','#83769c','#ff77a8','#ffccaa'],
    'cga': ['#000000','#555555','#0000aa','#5555ff','#00aa00','#55ff55',
            '#00aaaa','#55ffff','#aa0000','#ff5555','#aa00aa','#ff55ff',
            '#aa5500','#ffff55','#aaaaaa','#ffffff'],
    'cga-high': ['#000000','#55ffff','#ff55ff','#ffffff'],
    'cga-low':  ['#000000','#ff5555','#ffff55','#ffffff'],
    'cga-bw':   ['#000000','#ffffff'],
    'ega': ['#000000','#0000aa','#00aa00','#00aaaa','#aa0000','#aa00aa','#aa5500','#aaaaaa',
            '#555555','#5555ff','#55ff55','#55ffff','#ff5555','#ff55ff','#ffff55','#ffffff'],
    'grayscale': [f'#{i:02x}{i:02x}{i:02x}' for i in range(0, 256, 16)],
    'monochrome': ['#000000', '#ffffff'],
    'neon':  ['#0d0221','#0a0a2a','#ff006e','#fb5607','#ffbe0b','#8338ec','#3a86ff','#ffffff'],
    'pastel':['#f8f9fa','#ffd6e0','#d4f1f4','#b5ead7','#c7ceea','#fff5ba','#f0e6ff','#ffffff'],
    'autumn':['#2b170a','#5c2e0a','#8b3a0a','#b84a0a','#d96c0a','#f59e0a','#f8c471','#fae3c7'],
    'cyber': ['#000000','#1a0b2e','#ff00ff','#00ffff','#ff0080','#8000ff','#ffffff','#ffff00'],
}

# ============================================================
# MATRIZES DE DITHERING (ordered Bayer)
# ============================================================
DITHER_MATRICES = {
    'none':    None,
    'bayer2x2': np.array([[0, 2],
                          [3, 1]], dtype=float) / 4.0 - 0.5,
    'bayer4x4': np.array([[ 0, 8, 2,10],
                          [12, 4,14, 6],
                          [ 3,11, 1, 9],
                          [15, 7,13, 5]], dtype=float) / 16.0 - 0.5,
    'bayer8x8': np.array([
        [ 0,32, 8,40, 2,34,10,42],
        [48,16,56,24,50,18,58,26],
        [12,44, 4,36,14,46, 6,38],
        [60,28,52,20,62,30,54,22],
        [ 3,35,11,43, 1,33, 9,41],
        [51,19,59,27,49,17,57,25],
        [15,47, 7,39,13,45, 5,37],
        [63,31,55,23,61,29,53,21]
    ], dtype=float) / 64.0 - 0.5,
}

# ============================================================
# HELPERS
# ============================================================
def hex_to_rgb(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def seed_from_prompt(prompt, idx=0):
    """Hash determinístico a partir do prompt + índice."""
    h = hashlib.md5(f"{prompt}::{idx}".encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'big')

def make_rng(seed):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)

def detect_archetype(prompt):
    """Detecta tipo de sprite a partir do prompt para ajustar a forma."""
    p = (prompt or '').lower()
    if any(k in p for k in ['dragão','dragon','monstro','monster','beast','demon','slime',
                            'orc','goblin','serpent','snake','spider','boss','criatura']):
        return 'creature'
    if any(k in p for k in ['personagem','character','hero','herói','knight','guerreiro',
                            'warrior','mage','mago','elf','elfo','samurai','ninja','girl',
                            'boy','man','woman','pirate','king','queen','rei','rainha']):
        return 'character'
    if any(k in p for k in ['espada','sword','potion','poção','shield','escudo','bow','arco',
                            'axe','machado','key','chave','coin','moeda','gem','joia',
                            'item','arma','weapon','staff','ring','anel','heart','coração']):
        return 'item'
    if any(k in p for k in ['castelo','castle','torre','tower','casa','house','dungeon',
                            'cidade','city','building','prédio','templo','temple']):
        return 'building'
    if any(k in p for k in ['árvore','tree','planta','plant','flower','flor','cogumelo',
                            'mushroom','grass','grama','bush','arbusto','leaf','folha']):
        return 'plant'
    if any(k in p for k in ['montanha','mountain','nuvem','cloud','star','estrela',
                            'moon','lua','sun','sol','sky','céu','landscape','paisagem']):
        return 'landscape'
    if any(k in p for k in ['carro','car','ship','nave','spaceship','rocket','foguet',
                            'plane','avião','tank','tanque','ufo','ovni','mech','robô','robot']):
        return 'vehicle'
    return 'blob'

ARCHETYPE_PARAMS = {
    'creature':   {'fill': 0.50, 'tall': 0.9, 'detail': 0.7, 'eyes': 0.9, 'spiky': 0.6},
    'character':  {'fill': 0.45, 'tall': 1.4, 'detail': 0.8, 'eyes': 1.0, 'spiky': 0.1},
    'item':       {'fill': 0.35, 'tall': 1.0, 'detail': 0.9, 'eyes': 0.0, 'spiky': 0.3},
    'building':   {'fill': 0.60, 'tall': 1.2, 'detail': 0.5, 'eyes': 0.0, 'spiky': 0.4},
    'plant':      {'fill': 0.40, 'tall': 1.3, 'detail': 0.6, 'eyes': 0.0, 'spiky': 0.2},
    'landscape':  {'fill': 0.55, 'tall': 0.6, 'detail': 0.4, 'eyes': 0.0, 'spiky': 0.0},
    'vehicle':    {'fill': 0.45, 'tall': 0.8, 'detail': 0.7, 'eyes': 0.2, 'spiky': 0.5},
    'blob':       {'fill': 0.45, 'tall': 1.0, 'detail': 0.6, 'eyes': 0.5, 'spiky': 0.3},
}

# ============================================================
# GERADOR DE SILHUETA SIMÉTRICA
# ============================================================
def generate_silhouette(w, h, rng, fill_ratio=0.45, spiky=0.0, tall_ratio=1.0):
    """Cria uma máscara binária simétrica horizontalmente."""
    # Ajustar proporção efetiva baseada em tall_ratio
    eff_w = max(4, int(w / max(0.5, tall_ratio)))
    eff_w = min(eff_w, w)
    half_w = max(2, (eff_w + 1) // 2)

    noise = rng.random((h, half_w))
    ys = np.linspace(-1.0, 1.0, h)
    xs = np.linspace(-1.0, 1.0, half_w)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    dist = np.sqrt(xx**2 + yy**2)
    # Envelope elíptico suave
    bias = np.exp(-dist * 1.6) * fill_ratio * 2.2
    # Adicionar "espinhos" aleatórios na borda
    if spiky > 0:
        spike_noise = rng.random((h, half_w)) * spiky * 0.3
        # spikes crescem mais longe do centro
        bias = bias + spike_noise * dist * 1.5

    half_mask = (noise < bias).astype(np.uint8)

    # Espelhar
    if eff_w % 2 == 0:
        mask = np.concatenate([half_mask, half_mask[:, ::-1]], axis=1)
    else:
        mask = np.concatenate([half_mask, half_mask[:, :-1][:, ::-1]], axis=1)

    # Centralizar horizontalmente no canvas original
    if eff_w < w:
        pad_left = (w - eff_w) // 2
        pad_right = w - eff_w - pad_left
        mask = np.pad(mask, ((0,0),(pad_left, pad_right)), mode='constant')

    # Limpeza morfológica
    mask = clean_mask(mask)
    return mask

def clean_mask(mask, min_size=6):
    """Remove ilhas pequenas e garante ao menos 1 componente."""
    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_size:
            mask[labeled == i] = 0
    if mask.sum() < min_size:
        cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
        r = max(1, min(mask.shape) // 6)
        mask[max(0, cy-r):cy+r+1, max(0, cx-r):cx+r+1] = 1
    # Fecha pequenos buracos internos
    closed = ndimage.binary_fill_holes(mask.astype(bool)).astype(np.uint8)
    if closed.sum() > mask.sum() * 0.9:
        mask = closed
    return mask

# ============================================================
# DETALHES: OLHOS, SOMBRA E HIGHLIGHT
# ============================================================
def compute_details(base_mask, rng, detail_intensity=0.6, eyes_chance=0.8):
    h, w = base_mask.shape

    # Sombra na metade inferior + bordas
    ys = np.linspace(0.0, 1.0, h)[:, None] * np.ones((h, w))
    shadow = (base_mask == 1) & (ys > (0.55 + rng.random() * 0.15))

    # Highlight no topo
    highlight = (base_mask == 1) & (ys < (0.25 + rng.random() * 0.1))

    # "Olhos" simétricos
    eye_mask = np.zeros_like(base_mask, dtype=bool)
    if rng.random() < eyes_chance and detail_intensity > 0:
        eye_y = int(h * (0.30 + rng.random() * 0.15))
        eye_x = int(w * (0.30 + rng.random() * 0.10))
        eye_r = max(1, int(min(w, h) * (0.05 + rng.random() * 0.04)))
        for ey in range(max(0, eye_y - eye_r), min(h, eye_y + eye_r + 1)):
            for ex in (eye_x, w - 1 - eye_x):
                for dx in range(-eye_r, eye_r + 1):
                    for dy in range(-eye_r, eye_r + 1):
                        if dx*dx + dy*dy <= eye_r*eye_r:
                            py, px = ey + dy, ex + dx
                            if 0 <= py < h and 0 <= px < w:
                                eye_mask[py, px] = True

    # Detalhes internos (pintas/stripes) quando intensity é alta
    if detail_intensity > 0.6 and rng.random() < 0.7:
        inner = base_mask == 1
        # Stripes verticais fracas
        stripe = (np.arange(w) % max(2, int(4 + rng.random()*6))) < 1
        stripe_band = np.tile(stripe, (h, 1))
        detail = inner & stripe_band & (rng.random((h, w)) < 0.5)
        # Adiciona como sombra adicional
        shadow = shadow | detail

    # Resolver prioridades: olho > highlight > sombra > base
    highlight = highlight & ~eye_mask
    shadow    = shadow & ~eye_mask & ~highlight

    return shadow.astype(np.uint8), highlight.astype(np.uint8), eye_mask.astype(np.uint8)

# ============================================================
# OUTLINE
# ============================================================
def apply_outline(img_rgb, base_mask, outline_color, thickness=1):
    struct = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(base_mask.astype(bool), struct, iterations=thickness)
    outline = dilated & ~base_mask.astype(bool)
    result = img_rgb.copy()
    result[outline] = outline_color
    return result

# ============================================================
# DITHERING + QUANTIZAÇÃO (tudo junto, vetorizado)
# ============================================================
def apply_dither_and_quantize(img_rgb, palette_hex, dither_name):
    """Aplica ordered dithering e quantiza para a paleta em uma única passada."""
    palette = np.array([hex_to_rgb(c) for c in palette_hex], dtype=float)
    h, w, _ = img_rgb.shape
    img_f = img_rgb.astype(float)

    mat = DITHER_MATRICES.get(dither_name)
    if mat is None:
        # Apenas quantiza
        flat = img_f.reshape(-1, 3)
        diff = flat[:, None, :] - palette[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        return palette[idx].reshape(h, w, 3).astype(np.uint8)

    # Tile da matriz de Bayer
    mh, mw = mat.shape
    th = np.tile(mat, (int(np.ceil(h / mh)), int(np.ceil(w / mw))))[:h, :w]

    # Escala do dithering (quanto perturbar antes de quantizar)
    dither_scale = 48.0
    perturbed = img_f + th[..., None] * dither_scale
    perturbed = np.clip(perturbed, 0.0, 255.0)

    flat = perturbed.reshape(-1, 3)
    diff = flat[:, None, :] - palette[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    idx = np.argmin(dist, axis=1)

    return palette[idx].reshape(h, w, 3).astype(np.uint8)

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def generate_pixel_art(prompt, width=64, height=64, palette_size=16,
                       dithering='none', outline='none', style='auto',
                       seed=0, scale=1, archetype=None):
    """Gera UMA imagem de pixel art a partir do prompt."""
    rng = make_rng(seed_from_prompt(prompt, seed))

    # 1) Paleta
    if style == 'auto' or style not in STYLES:
        available = [k for k in STYLES.keys() if k not in ('auto',)]
        style = available[seed_from_prompt(prompt, 0) % len(available)]
    palette = list(STYLES[style])
    while len(palette) < max(4, palette_size):
        palette.append(palette[len(palette) % len(STYLES[style])])
    palette = palette[:max(4, palette_size)]

    # 2) Archetype (ajusta proporção/forma)
    if archetype is None:
        archetype = detect_archetype(prompt)
    params = ARCHETYPE_PARAMS.get(archetype, ARCHETYPE_PARAMS['blob'])
    # Leve variação aleatória em cima dos parâmetros
    fill = params['fill'] + (rng.random() - 0.5) * 0.1
    fill = float(np.clip(fill, 0.25, 0.75))

    # 3) Silhueta
    base_mask = generate_silhouette(
        width, height, rng,
        fill_ratio=fill,
        spiky=params['spiky'],
        tall_ratio=params['tall']
    )

    # 4) Detalhes
    shadow, highlight, eyes = compute_details(
        base_mask, rng,
        detail_intensity=params['detail'],
        eyes_chance=params['eyes']
    )

    # 5) Mapeamento de cores da paleta
    #    bg (fundo)         = paleta[0]
    #    base (corpo)       = paleta[2]
    #    sombra             = paleta[1]
    #    highlight          = paleta[-1] (mais clara)
    #    olho               = cor mais escura (geralmente paleta[0] ou [1])
    bg_color        = hex_to_rgb(palette[0])
    base_color      = hex_to_rgb(palette[2 % len(palette)])
    shadow_color    = hex_to_rgb(palette[1 % len(palette)])
    highlight_color = hex_to_rgb(palette[-1])
    eye_color       = hex_to_rgb(palette[0]) if len(palette) > 0 else (0, 0, 0)

    # Monta imagem em RGB
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    body = (base_mask == 1) & (shadow == 0) & (highlight == 0) & (eyes == 0)
    img[body]          = base_color
    img[shadow == 1]   = shadow_color
    img[highlight == 1]= highlight_color
    img[eyes == 1]     = eye_color

    # 6) Outline (opcional, antes do dither)
    outline_applied = False
    if outline not in (None, 'none', 'None', '', 'off', 'false'):
        if outline == 'black':
            oc = (0, 0, 0)
        elif outline == 'white':
            oc = (255, 255, 255)
        elif outline == 'dark':
            oc = hex_to_rgb(palette[0])
        else:
            try:
                oc = hex_to_rgb(outline)
            except Exception:
                oc = (0, 0, 0)
        img = apply_outline(img, base_mask, oc, thickness=1)
        outline_applied = True

    # 7) Dither + quantização final
    img = apply_dither_and_quantize(img, palette, dithering)

    # 8) Scale (pixel-perfect via NEAREST)
    if scale and int(scale) > 1:
        pil = Image.fromarray(img)
        pil = pil.resize((width * int(scale), height * int(scale)), Image.NEAREST)
    else:
        pil = Image.fromarray(img)

    pil.info['style'] = style
    pil.info['archetype'] = archetype
    pil.info['outline_applied'] = outline_applied
    return pil

# ============================================================
# BATCH
# ============================================================
def batch_generate(prompt, count=1, **kwargs):
    """Gera N variações. Resolve o bug 'multiple values for seed'."""
    results = []
    for i in range(int(count)):
        kw_copy = dict(kwargs)         # cópia independente
        kw_copy['seed'] = i            # injeta o seed aqui, não na chamada
        results.append(generate_pixel_art(prompt, **kw_copy))
    return results

# ============================================================
# CLI
# ============================================================
def main():
    print("🎨 PIXEL ART AI GENERATOR")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Gera pixel art procedural")
    parser.add_argument('--prompt', required=True, help="Descrição da arte")
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--palette-size', type=int, default=16)
    parser.add_argument('--dithering', default='none',
                        choices=list(DITHER_MATRICES.keys()))
    parser.add_argument('--outline', default='none',
                        help="none | black | white | dark | #rrggbb")
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--style', default='auto',
                        choices=list(STYLES.keys()) + ['auto'])
    parser.add_argument('--batch', action='store_true',
                        help="Ativa modo batch (--count > 1)")
    parser.add_argument('--archetype', default=None,
                        choices=list(ARCHETYPE_PARAMS.keys()) + [None],
                        help="Força tipo de sprite (auto-detectado do prompt se omitido)")
    parser.add_argument('--outdir', default='pixel_art_output')
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed global (ignorado em batch)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    kwargs = dict(
        width=args.width,
        height=args.height,
        palette_size=args.palette_size,
        dithering=args.dithering,
        outline=args.outline,
        style=args.style,
        scale=args.scale,
    )
    if args.archetype is not None:
        kwargs['archetype'] = args.archetype

    use_batch = args.batch or args.count > 1
    if use_batch:
        images = batch_generate(args.prompt, count=args.count, **kwargs)
    else:
        if args.seed is not None:
            kwargs['seed'] = args.seed
        images = [generate_pixel_art(args.prompt, **kwargs)]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_prompt = "".join(c for c in (args.prompt or '') if c.isalnum() or c in ' _-')
    safe_prompt = safe_prompt.strip().replace(' ', '_')[:40] or 'pixel'

    paths = []
    for i, img in enumerate(images):
        style_used = img.info.get('style', args.style)
        arch_used  = img.info.get('archetype', '?')
        fname = f"{safe_prompt}_{style_used}_{arch_used}_{ts}_{i:03d}.png"
        path = os.path.join(args.outdir, fname)
        img.save(path, optimize=True)
        paths.append(path)
        print(f"  ✓ [{i+1}/{len(images)}] {fname}  (style={style_used}, arch={arch_used})")

    print(f"\n✨ {len(images)} imagem(ns) gerada(s) em '{args.outdir}/'")
    return paths

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ cancelado")
        sys.exit(130)
    except Exception as e:
        print(f"❌ erro: {e}", file=sys.stderr)
        raise
