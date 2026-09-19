#!/usr/bin/env python3
"""
🎨 PIXEL ART AI GENERATOR v2 — com silhuetas compostas por archetype
"""
import argparse
import hashlib
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# ============================================================
# PALETAS REAIS DE CONSOLES
# ============================================================
STYLES = {
    'gameboy': ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'],
    'gameboy-pocket': ['#000000', '#555555', '#aaaaaa', '#ffffff'],
    'gameboy-light':  ['#2b2b2b', '#5a5a5a', '#9a9a9a', '#dadada'],
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
    'snes':  ['#000000','#1a1a2e','#16213e','#0f3460','#533483','#e94560','#f5b461','#f5d76e'],
    'pico8': ['#000000','#1d2b53','#7e2553','#008751','#ab5236','#5f574f',
              '#c2c3c7','#fff1e8','#ff004d','#ffa300','#ffec27','#00e436',
              '#29adff','#83769c','#ff77a8','#ffccaa'],
    'cga': ['#000000','#555555','#0000aa','#5555ff','#00aa00','#55ff55',
            '#00aaaa','#55ffff','#aa0000','#ff5555','#aa00aa','#ff55ff',
            '#aa5500','#ffff55','#aaaaaa','#ffffff'],
    'cga-high': ['#000000','#55ffff','#ff55ff','#ffffff'],
    'cga-low':  ['#000000','#ff5555','#ffff55','#ffffff'],
    'ega': ['#000000','#0000aa','#00aa00','#00aaaa','#aa0000','#aa00aa','#aa5500','#aaaaaa',
            '#555555','#5555ff','#55ff55','#55ffff','#ff5555','#ff55ff','#ffff55','#ffffff'],
    'grayscale': [f'#{i:02x}{i:02x}{i:02x}' for i in range(0, 256, 32)],
    'monochrome': ['#000000', '#ffffff'],
    'neon':  ['#0d0221','#ff006e','#fb5607','#ffbe0b','#8338ec','#3a86ff','#ffffff'],
    'pastel':['#f8f9fa','#ffd6e0','#d4f1f4','#b5ead7','#c7ceea','#fff5ba','#f0e6ff','#ffffff'],
    'autumn':['#2b170a','#5c2e0a','#8b3a0a','#b84a0a','#d96c0a','#f59e0a','#f8c471','#fae3c7'],
    'cyber': ['#000000','#1a0b2e','#ff00ff','#00ffff','#ff0080','#8000ff','#ffffff','#ffff00'],
}

# ============================================================
# DITHERING (Bayer)
# ============================================================
DITHER_MATRICES = {
    'none':    None,
    'bayer2x2': np.array([[0, 2],[3, 1]], dtype=float) / 4.0 - 0.5,
    'bayer4x4': np.array([[ 0, 8, 2,10],[12, 4,14, 6],
                          [ 3,11, 1, 9],[15, 7,13, 5]], dtype=float) / 16.0 - 0.5,
    'bayer8x8': np.array([
        [ 0,32, 8,40, 2,34,10,42],[48,16,56,24,50,18,58,26],
        [12,44, 4,36,14,46, 6,38],[60,28,52,20,62,30,54,22],
        [ 3,35,11,43, 1,33, 9,41],[51,19,59,27,49,17,57,25],
        [15,47, 7,39,13,45, 5,37],[63,31,55,23,61,29,53,21]
    ], dtype=float) / 64.0 - 0.5,
}

# ============================================================
# HELPERS
# ============================================================
def hex_to_rgb(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def seed_from_prompt(prompt, idx=0):
    h = hashlib.md5(f"{prompt}::{idx}".encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'big')

def make_rng(seed):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)

def detect_archetype(prompt):
    p = (prompt or '').lower()
    if any(k in p for k in ['dragão','dragon','monstro','monster','beast','demon','slime',
                            'orc','goblin','serpent','snake','spider','boss','criatura',
                            'grifo','griffin','lobo','wolf','urso','bear','troll']):
        return 'creature'
    if any(k in p for k in ['personagem','character','hero','herói','knight','guerreiro',
                            'warrior','mage','mago','elf','elfo','samurai','ninja','girl',
                            'boy','man','woman','pirate','king','queen','rei','rainha',
                            'garoto','garota','cavaleiro','princesa','príncipe']):
        return 'character'
    if any(k in p for k in ['espada','sword','potion','poção','shield','escudo','bow','arco',
                            'axe','machado','key','chave','coin','moeda','gem','joia',
                            'item','arma','weapon','staff','ring','anel','heart','coração',
                            'livro','book','caixa','chest','bottle','garrafa']):
        return 'item'
    if any(k in p for k in ['castelo','castle','torre','tower','casa','house','dungeon',
                            'cidade','city','building','prédio','templo','temple','igreja',
                            'church','cabana','hut']):
        return 'building'
    if any(k in p for k in ['árvore','tree','planta','plant','flower','flor','cogumelo',
                            'mushroom','grass','grama','bush','arbusto','leaf','folha']):
        return 'plant'
    if any(k in p for k in ['carro','car','ship','nave','spaceship','rocket','foguete',
                            'plane','avião','tank','tanque','ufo','ovni','mech','robô','robot']):
        return 'vehicle'
    return 'blob'

# ============================================================
# GERADORES DE SILHUETA POR ARCHETYPE (formas reconhecíveis!)
# ============================================================
def draw_ellipse(draw, cx, cy, rx, ry, fill=1):
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill)

def draw_rect(draw, cx, cy, w, h, fill=1, angle=0):
    if angle == 0:
        draw.rectangle([cx - w/2, cy - h/2, cx + w/2, cy + h/2], fill=fill)
    else:
        # Para retângulos rotacionados, cria um canvas temporário
        tmp = Image.new('L', (w + abs(h), h + abs(w)), 0)
        d = ImageDraw.Draw(tmp)
        d.rectangle([abs(h)/2, abs(w)/2, abs(h)/2 + w, abs(w)/2 + h], fill=fill)
        tmp = tmp.rotate(angle, expand=True, resample=Image.NEAREST)
        return tmp  # retorna a imagem para colar

def silhouette_creature(w, h, rng):
    """Dragão / monstro: corpo + cabeça + cauda + asas + pernas"""
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    
    # Corpo principal (elipse larga)
    body_rx = int(w * 0.28)
    body_ry = int(h * 0.22)
    body_cy = cy + int(h * 0.05)
    draw_ellipse(d, cx, body_cy, body_rx, body_ry, fill=1)
    
    # Cabeça (elipse à direita)
    head_rx = int(w * 0.14)
    head_ry = int(h * 0.13)
    head_cx = cx + int(w * 0.28)
    head_cy = body_cy - int(h * 0.08)
    draw_ellipse(d, head_cx, head_cy, head_rx, head_ry, fill=1)
    
    # Pescoço (retângulo conectando)
    neck_w = int(w * 0.08)
    neck_h = int(h * 0.12)
    neck_cx = (head_cx + cx) // 2
    neck_cy = (head_cy + body_cy) // 2
    d.rectangle([neck_cx - neck_w/2, neck_cy - neck_h/2, 
                 neck_cx + neck_w/2, neck_cy + neck_h/2], fill=1)
    
    # Cauda (triângulo longo à esquerda, curvando para cima)
    tail_points = [
        (cx - int(w * 0.18), body_cy),
        (cx - int(w * 0.42), body_cy - int(h * 0.05)),
        (cx - int(w * 0.40), body_cy + int(h * 0.08)),
    ]
    d.polygon(tail_points, fill=1)
    # Ponta da cauda (seta)
    tip_cx = cx - int(w * 0.42)
    tip_cy = body_cy - int(h * 0.05)
    d.polygon([
        (tip_cx, tip_cy),
        (tip_cx - int(w * 0.05), tip_cy - int(h * 0.05)),
        (tip_cx - int(w * 0.03), tip_cy + int(h * 0.03)),
    ], fill=1)
    
    # Asas (triângulos no topo do corpo)
    if rng.random() < 0.85:
        # Asa esquerda
        wing_pts_l = [
            (cx - int(w * 0.05), body_cy - int(h * 0.12)),
            (cx - int(w * 0.20), body_cy - int(h * 0.42)),
            (cx - int(w * 0.05), body_cy - int(h * 0.35)),
            (cx + int(w * 0.02), body_cy - int(h * 0.20)),
        ]
        d.polygon(wing_pts_l, fill=1)
        # Asa direita (espelhada)
        wing_pts_r = [(w - 1 - x, y) for (x, y) in wing_pts_l]
        d.polygon(wing_pts_r, fill=1)
    
    # Pernas (2 retângulos embaixo)
    leg_w = int(w * 0.09)
    leg_h = int(h * 0.18)
    leg_y = body_cy + int(h * 0.18)
    for lx in [cx - int(w * 0.15), cx + int(w * 0.08)]:
        d.rectangle([lx - leg_w/2, leg_y - leg_h/2, 
                     lx + leg_w/2, leg_y + leg_h/2], fill=1)
        # Pés
        d.rectangle([lx - leg_w*0.8, leg_y + leg_h/2, 
                     lx + leg_w*0.8, leg_y + leg_h/2 + int(h * 0.03)], fill=1)
    
    # Chifres (pequenos triângulos na cabeça)
    horn_base_y = head_cy - head_ry
    for hx in [head_cx - int(w * 0.06), head_cx + int(w * 0.03)]:
        d.polygon([
            (hx, horn_base_y),
            (hx - int(w * 0.02), horn_base_y - int(h * 0.08)),
            (hx + int(w * 0.02), horn_base_y - int(h * 0.08)),
        ], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def silhouette_character(w, h, rng):
    """Personagem humanoide: cabeça + corpo + braços + pernas (chibi)"""
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    
    # Cabeça (grande, estilo chibi — 40% da altura)
    head_ry = int(h * 0.22)
    head_rx = int(w * 0.18)
    head_cy = cy - int(h * 0.18)
    draw_ellipse(d, cx, head_cy, head_rx, head_ry, fill=1)
    
    # Corpo (retângulo)
    body_w = int(w * 0.28)
    body_h = int(h * 0.28)
    body_cy = cy + int(h * 0.05)
    d.rectangle([cx - body_w/2, body_cy - body_h/2, 
                 cx + body_w/2, body_cy + body_h/2], fill=1)
    
    # Braços (retângulos laterais, levemente angulados)
    arm_w = int(w * 0.09)
    arm_h = int(h * 0.22)
    for ax in [cx - body_w/2 - arm_w/2, cx + body_w/2 + arm_w/2]:
        d.rectangle([ax - arm_w/2, body_cy - int(h * 0.05),
                     ax + arm_w/2, body_cy - int(h * 0.05) + arm_h], fill=1)
    
    # Pernas (2 retângulos embaixo)
    leg_w = int(w * 0.11)
    leg_h = int(h * 0.28)
    leg_top = body_cy + body_h/2
    for lx in [cx - int(w * 0.07), cx + int(w * 0.07)]:
        d.rectangle([lx - leg_w/2, leg_top, 
                     lx + leg_w/2, leg_top + leg_h], fill=1)
    
    # Cabelo / topo da cabeça (detalhe simples)
    hair_w = int(w * 0.22)
    hair_h = int(h * 0.06)
    d.rectangle([cx - hair_w/2, head_cy - head_ry - hair_h/2,
                 cx + hair_w/2, head_cy - head_ry + hair_h/2], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def silhouette_item(w, h, rng):
    """Item genérico (espada, poção, etc) — escolhe aleatoriamente"""
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    
    item_type = rng.choice(['sword', 'potion', 'shield', 'gem'])
    
    if item_type == 'sword':
        # Lâmina (retângulo vertical longo)
        blade_w = int(w * 0.08)
        blade_h = int(h * 0.55)
        blade_cy = cy - int(h * 0.12)
        d.rectangle([cx - blade_w/2, blade_cy - blade_h/2,
                     cx + blade_w/2, blade_cy + blade_h/2], fill=1)
        # Ponta
        d.polygon([
            (cx - blade_w/2, blade_cy - blade_h/2),
            (cx + blade_w/2, blade_cy - blade_h/2),
            (cx, blade_cy - blade_h/2 - int(h * 0.05)),
        ], fill=1)
        # Guarda (retângulo horizontal)
        guard_w = int(w * 0.30)
        guard_h = int(h * 0.06)
        guard_cy = blade_cy + blade_h/2
        d.rectangle([cx - guard_w/2, guard_cy - guard_h/2,
                     cx + guard_w/2, guard_cy + guard_h/2], fill=1)
        # Cabo
        grip_w = int(w * 0.10)
        grip_h = int(h * 0.18)
        grip_cy = guard_cy + guard_h/2 + grip_h/2
        d.rectangle([cx - grip_w/2, grip_cy - grip_h/2,
                     cx + grip_w/2, grip_cy + grip_h/2], fill=1)
        # Pomo (círculo)
        d.ellipse([cx - int(w * 0.06), grip_cy + grip_h/2 - int(h * 0.02),
                   cx + int(w * 0.06), grip_cy + grip_h/2 + int(h * 0.08)], fill=1)
        
    elif item_type == 'potion':
        # Frasco (círculo grande)
        body_r = int(min(w, h) * 0.28)
        body_cy = cy + int(h * 0.05)
        draw_ellipse(d, cx, body_cy, body_r, body_r, fill=1)
        # Pescoço (retângulo)
        neck_w = int(w * 0.12)
        neck_h = int(h * 0.15)
        d.rectangle([cx - neck_w/2, body_cy - body_r - neck_h + int(h*0.03),
                     cx + neck_w/2, body_cy - body_r + int(h*0.03)], fill=1)
        # Tampa
        cap_w = int(w * 0.18)
        cap_h = int(h * 0.08)
        cap_y = body_cy - body_r - neck_h - int(h*0.02)
        d.rectangle([cx - cap_w/2, cap_y, cx + cap_w/2, cap_y + cap_h], fill=1)
        
    elif item_type == 'shield':
        # Escudo (forma de ponta arredondada)
        sh_w = int(w * 0.65)
        sh_h = int(h * 0.75)
        d.pieslice([cx - sh_w/2, cy - sh_h/2, cx + sh_w/2, cy + sh_h/2 + int(h*0.2)],
                   0, 360, fill=1)
        # Borda central
        d.rectangle([cx - int(w*0.03), cy - sh_h/2, 
                     cx + int(w*0.03), cy + sh_h/2 + int(h*0.1)], fill=1)
        
    else:  # gem
        # Gema (diamante)
        gem_w = int(w * 0.45)
        gem_h = int(h * 0.55)
        d.polygon([
            (cx, cy - gem_h/2),
            (cx + gem_w/2, cy),
            (cx, cy + gem_h/2),
            (cx - gem_w/2, cy),
        ], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def silhouette_building(w, h, rng):
    """Castelo / casa: base + torres + telhado"""
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    
    # Base (retângulo largo)
    base_w = int(w * 0.65)
    base_h = int(h * 0.45)
    base_cy = cy + int(h * 0.15)
    d.rectangle([cx - base_w/2, base_cy - base_h/2,
                 cx + base_w/2, base_cy + base_h/2], fill=1)
    
    # Telhado (triângulo no topo da base)
    d.polygon([
        (cx - base_w/2 - int(w*0.05), base_cy - base_h/2),
        (cx, base_cy - base_h/2 - int(h * 0.15)),
        (cx + base_w/2 + int(w*0.05), base_cy - base_h/2),
    ], fill=1)
    
    # 2 torres laterais
    tower_w = int(w * 0.15)
    tower_h = int(h * 0.65)
    for tx in [cx - base_w/2 - tower_w/2 + int(w*0.05), 
               cx + base_w/2 - tower_w/2 - int(w*0.05)]:
        ty = cy + int(h * 0.10)
        d.rectangle([tx - tower_w/2, ty - tower_h/2,
                     tx + tower_w/2, ty + tower_h/2], fill=1)
        # Topo da torre (pontudo)
        d.polygon([
            (tx - tower_w/2, ty - tower_h/2),
            (tx, ty - tower_h/2 - int(h * 0.10)),
            (tx + tower_w/2, ty - tower_h/2),
        ], fill=1)
    
    # Porta central (fica "cortada" como detalhe negativo mais tarde)
    door_w = int(w * 0.12)
    door_h = int(h * 0.25)
    door_cy = base_cy + base_h/2 - door_h/2
    d.rectangle([cx - door_w/2, door_cy - door_h/2,
                 cx + door_w/2, door_cy + door_h/2], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def silhouette_plant(w, h, rng):
    """Árvore / planta: caule + copa"""
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    
    # Caule (retângulo vertical)
    stem_w = int(w * 0.10)
    stem_h = int(h * 0.35)
    stem_cy = cy + int(h * 0.15)
    d.rectangle([cx - stem_w/2, stem_cy - stem_h/2,
                 cx + stem_w/2, stem_cy + stem_h/2], fill=1)
    
    # Copa (3 elipses sobrepostas formando nuvem)
    crown_cy = cy - int(h * 0.10)
    crown_r = int(min(w, h) * 0.22)
    offsets = [
        (cx, crown_cy - int(h * 0.05)),
        (cx - int(w * 0.12), crown_cy + int(h * 0.02)),
        (cx + int(w * 0.12), crown_cy + int(h * 0.02)),
    ]
    for (ox, oy) in offsets:
        draw_ellipse(d, ox, oy, crown_r, crown_r * 0.85, fill=1)
    
    return np.array(mask, dtype=np.uint8)

def silhouette_vehicle(w, h, rng):
    """Veículo: corpo principal + rodas"""
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    
    # Corpo principal (retângulo grande horizontal)
    body_w = int(w * 0.70)
    body_h = int(h * 0.30)
    d.rectangle([cx - body_w/2, cy - body_h/2,
                 cx + body_w/2, cy + body_h/2], fill=1)
    
    # Cabine (retângulo no topo)
    cab_w = int(w * 0.35)
    cab_h = int(h * 0.20)
    cab_cy = cy - body_h/2 - cab_h/2 + int(h * 0.02)
    d.rectangle([cx - cab_w/2 - int(w * 0.05), cab_cy - cab_h/2,
                 cx + cab_w/2 - int(w * 0.05), cab_cy + cab_h/2], fill=1)
    
    # 2 rodas (círculos embaixo)
    wheel_r = int(min(w, h) * 0.08)
    wheel_y = cy + body_h/2
    for wx in [cx - int(w * 0.22), cx + int(w * 0.15)]:
        draw_ellipse(d, wx, wheel_y, wheel_r, wheel_r, fill=1)
    
    return np.array(mask, dtype=np.uint8)

def silhouette_blob(w, h, rng):
    """Fallback: elipse simples"""
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    rx = int(w * 0.35)
    ry = int(h * 0.30)
    draw_ellipse(d, cx, cy, rx, ry, fill=1)
    return np.array(mask, dtype=np.uint8)

ARCHETYPE_BUILDERS = {
    'creature': silhouette_creature,
    'character': silhouette_character,
    'item': silhouette_item,
    'building': silhouette_building,
    'plant': silhouette_plant,
    'vehicle': silhouette_vehicle,
    'blob': silhouette_blob,
}

def generate_silhouette(w, h, rng, archetype):
    """Gera silhueta usando primitivos geométricos, não ruído."""
    builder = ARCHETYPE_BUILDERS.get(archetype, silhouette_blob)
    return builder(w, h, rng)

# ============================================================
# DETALHES (olhos, sombra, highlight)
# ============================================================
def compute_details(base_mask, rng, archetype):
    h, w = base_mask.shape
    
    # Sombras: metade inferior do corpo + bordas
    ys = np.linspace(0.0, 1.0, h)[:, None] * np.ones((h, w))
    shadow = (base_mask == 1) & (ys > 0.62)
    
    # Highlight: topo superior
    highlight = (base_mask == 1) & (ys < 0.25)
    
    # Olhos (apenas para creature e character)
    eye_mask = np.zeros_like(base_mask, dtype=bool)
    if archetype in ('creature', 'character') and rng.random() < 0.95:
        # Detecta centro horizontal da parte superior da máscara
        upper = base_mask.copy()
        upper[h // 2:, :] = 0
        cols = np.any(upper > 0, axis=0)
        if np.any(cols):
            left = np.argmax(cols)
            right = w - 1 - np.argmax(cols[::-1])
            center_x = (left + right) // 2
            width_span = right - left
            
            # Y dos olhos: ~35% da altura da parte superior
            rows = np.any(upper > 0, axis=1)
            row_indices = np.where(rows)[0]
            if len(row_indices) > 0:
                eye_y = int(row_indices[0] + (row_indices[-1] - row_indices[0]) * 0.45)
                eye_offset = int(width_span * 0.15)
                eye_r = max(1, int(width_span * 0.06))
                
                for ex in [center_x - eye_offset, center_x + eye_offset]:
                    y0 = max(0, eye_y - eye_r)
                    y1 = min(h, eye_y + eye_r + 1)
                    x0 = max(0, ex - eye_r)
                    x1 = min(w, ex + eye_r + 1)
                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            if (xx - ex)**2 + (yy - eye_y)**2 <= eye_r**2:
                                eye_mask[yy, xx] = True
    
    # Remover sobreposições
    highlight = highlight & ~eye_mask
    shadow = shadow & ~eye_mask & ~highlight
    
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
# DITHER + QUANTIZAÇÃO
# ============================================================
def apply_dither_and_quantize(img_rgb, palette_hex, dither_name):
    palette = np.array([hex_to_rgb(c) for c in palette_hex], dtype=float)
    h, w, _ = img_rgb.shape
    img_f = img_rgb.astype(float)
    
    mat = DITHER_MATRICES.get(dither_name)
    if mat is None:
        flat = img_f.reshape(-1, 3)
        diff = flat[:, None, :] - palette[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        return palette[idx].reshape(h, w, 3).astype(np.uint8)
    
    mh, mw = mat.shape
    th = np.tile(mat, (int(np.ceil(h / mh)), int(np.ceil(w / mw))))[:h, :w]
    dither_scale = 32.0  # menor que antes para menos ruído
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
    rng = make_rng(seed_from_prompt(prompt, seed))
    
    # 1) Paleta
    if style == 'auto' or style not in STYLES:
        available = list(STYLES.keys())
        style = available[seed_from_prompt(prompt, 0) % len(available)]
    palette = list(STYLES[style])
    while len(palette) < max(4, palette_size):
        palette.append(palette[len(palette) % len(STYLES[style])])
    palette = palette[:max(4, palette_size)]
    
    # 2) Archetype
    if archetype is None:
        archetype = detect_archetype(prompt)
    
    # 3) Silhueta (geométrica, não noise!)
    base_mask = generate_silhouette(width, height, rng, archetype)
    
    # 4) Detalhes
    shadow, highlight, eyes = compute_details(base_mask, rng, archetype)
    
    # 5) Cores:
    #    paleta[0] = cor mais escura (fundo + olhos)
    #    paleta[1] = sombra
    #    paleta[2] = base do corpo
    #    paleta[-1] = highlight
    bg_color        = hex_to_rgb(palette[0])
    shadow_color    = hex_to_rgb(palette[1 % len(palette)])
    base_color      = hex_to_rgb(palette[2 % len(palette)])
    highlight_color = hex_to_rgb(palette[-1])
    eye_color       = hex_to_rgb(palette[0])
    
    # 6) Monta imagem
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    body = (base_mask == 1) & (shadow == 0) & (highlight == 0) & (eyes == 0)
    img[body]          = base_color
    img[shadow == 1]   = shadow_color
    img[highlight == 1]= highlight_color
    img[eyes == 1]     = eye_color
    
    # 7) Outline (opcional)
    if outline not in (None, 'none', 'None', '', 'off', 'false'):
        if outline == 'black':    oc = (0, 0, 0)
        elif outline == 'white':  oc = (255, 255, 255)
        elif outline == 'dark':   oc = hex_to_rgb(palette[0])
        else:
            try: oc = hex_to_rgb(outline)
            except: oc = (0, 0, 0)
        img = apply_outline(img, base_mask, oc, thickness=1)
    
    # 8) Dither + quantização
    img = apply_dither_and_quantize(img, palette, dithering)
    
    # 9) Scale
    if scale and int(scale) > 1:
        pil = Image.fromarray(img)
        pil = pil.resize((width * int(scale), height * int(scale)), Image.NEAREST)
    else:
        pil = Image.fromarray(img)
    
    pil.info['style'] = style
    pil.info['archetype'] = archetype
    return pil

# ============================================================
# BATCH
# ============================================================
def batch_generate(prompt, count=1, **kwargs):
    results = []
    for i in range(int(count)):
        kw_copy = dict(kwargs)
        kw_copy['seed'] = i
        results.append(generate_pixel_art(prompt, **kw_copy))
    return results

# ============================================================
# CLI
# ============================================================
def main():
    print("🎨 PIXEL ART AI GENERATOR v2")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Gera pixel art procedural com silhuetas compostas")
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--palette-size', type=int, default=16)
    parser.add_argument('--dithering', default='none', choices=list(DITHER_MATRICES.keys()))
    parser.add_argument('--outline', default='none')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--style', default='auto', choices=list(STYLES.keys()) + ['auto'])
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--archetype', default=None, choices=list(ARCHETYPE_BUILDERS.keys()) + [None])
    parser.add_argument('--outdir', default='pixel_art_output')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    kwargs = dict(
        width=args.width, height=args.height,
        palette_size=args.palette_size,
        dithering=args.dithering, outline=args.outline,
        style=args.style, scale=args.scale,
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
