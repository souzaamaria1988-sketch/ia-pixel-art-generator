#!/usr/bin/env python3
"""
🎨 PIXEL ART AI GENERATOR v3 — com mutação de características entre seeds
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
# PALETAS
# ============================================================
STYLES = {
    'gameboy': ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'],
    'gameboy-pocket': ['#000000', '#555555', '#aaaaaa', '#ffffff'],
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
    'ega': ['#000000','#0000aa','#00aa00','#00aaaa','#aa0000','#aa00aa','#aa5500','#aaaaaa',
            '#555555','#5555ff','#55ff55','#55ffff','#ff5555','#ff55ff','#ffff55','#ffffff'],
    'grayscale': [f'#{i:02x}{i:02x}{i:02x}' for i in range(0, 256, 32)],
    'monochrome': ['#000000', '#ffffff'],
    'neon':  ['#0d0221','#ff006e','#fb5607','#ffbe0b','#8338ec','#3a86ff','#ffffff'],
    'cyber': ['#000000','#1a0b2e','#ff00ff','#00ffff','#ff0080','#8000ff','#ffffff','#ffff00'],
}

DITHER_MATRICES = {
    'none':    None,
    'bayer2x2': np.array([[0,2],[3,1]], dtype=float)/4.0 - 0.5,
    'bayer4x4': np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]], dtype=float)/16.0 - 0.5,
    'bayer8x8': np.array([
        [0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],
        [12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
        [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],
        [15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]
    ], dtype=float)/64.0 - 0.5,
}

def hex_to_rgb(h):
    h = h.lstrip('#')
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

def seed_from_prompt(prompt, idx=0):
    h = hashlib.md5(f"{prompt}::{idx}".encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'big')

def make_rng(seed):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)

def detect_archetype(prompt):
    p = (prompt or '').lower()
    if any(k in p for k in ['dragão','dragon','monstro','monster','beast','demon','slime',
                            'orc','goblin','serpent','snake','spider','boss','criatura',
                            'lobo','wolf','urso','bear','grifo']):
        return 'creature'
    if any(k in p for k in ['personagem','character','hero','herói','knight','guerreiro',
                            'warrior','mage','mago','elf','elfo','samurai','ninja','girl',
                            'boy','man','woman','pirate','king','queen','rei','rainha']):
        return 'character'
    if any(k in p for k in ['espada','sword','potion','poção','shield','escudo','bow','arco',
                            'axe','machado','key','chave','coin','moeda','gem','joia','item']):
        return 'item'
    if any(k in p for k in ['castelo','castle','torre','tower','casa','house','dungeon',
                            'cidade','building','prédio','templo','church']):
        return 'building'
    if any(k in p for k in ['árvore','tree','planta','plant','flower','flor','cogumelo']):
        return 'plant'
    if any(k in p for k in ['carro','car','ship','nave','spaceship','rocket','foguete',
                            'plane','avião','tank','ufo','robô','robot']):
        return 'vehicle'
    return 'blob'

# ============================================================
# FEATURES MUTÁVEIS — cada seed escolhe proporções diferentes!
# ============================================================
def roll_features(rng, archetype, variation='medium'):
    """Gera características aleatórias baseadas no seed + nível de variação."""
    intensity = {'low': 0.3, 'medium': 0.7, 'heavy': 1.0}[variation]
    f = {}
    
    if archetype == 'creature':
        f['body_width']  = rng.uniform(0.55, 0.85) * intensity + rng.uniform(0.15, 0.3)
        f['body_height'] = rng.uniform(0.35, 0.65) * intensity + rng.uniform(0.15, 0.25)
        f['head_size']   = rng.uniform(0.10, 0.22) * intensity + rng.uniform(0.08, 0.12)
        f['has_wings']   = rng.random() < 0.75 * intensity
        f['has_horns']   = rng.random() < 0.60 * intensity
        f['has_tail']    = rng.random() < 0.90 * intensity
        f['tail_curve']  = rng.uniform(-0.3, 0.5)
        f['leg_count']   = rng.choice([2, 4], p=[0.4, 0.6])
        f['leg_length']  = rng.uniform(0.12, 0.28) * intensity + rng.uniform(0.08, 0.15)
        f['neck_angle']  = rng.uniform(-0.25, 0.15)
        f['head_pos']    = rng.uniform(0.15, 0.35)
        f['body_pos_y']  = rng.uniform(0.45, 0.60)
        
    elif archetype == 'character':
        f['head_size']   = rng.uniform(0.16, 0.26) * intensity + rng.uniform(0.12, 0.18)
        f['body_width']  = rng.uniform(0.20, 0.38) * intensity + rng.uniform(0.15, 0.22)
        f['body_height'] = rng.uniform(0.22, 0.38) * intensity + rng.uniform(0.18, 0.25)
        f['arm_length']  = rng.uniform(0.15, 0.32) * intensity + rng.uniform(0.12, 0.18)
        f['leg_length']  = rng.uniform(0.22, 0.38) * intensity + rng.uniform(0.18, 0.25)
        f['has_hat']     = rng.random() < 0.55 * intensity
        f['has_cape']    = rng.random() < 0.30 * intensity
        f['head_y']      = rng.uniform(0.20, 0.32)
        
    elif archetype == 'item':
        f['item_type']   = rng.choice(['sword', 'potion', 'shield', 'gem', 'axe'])
        f['size_scale']  = rng.uniform(0.75, 1.15)
        f['angle']       = rng.uniform(-15, 15)
        f['has_glow']    = rng.random() < 0.40 * intensity
        
    elif archetype == 'building':
        f['tower_count'] = rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        f['base_width']  = rng.uniform(0.55, 0.75) * intensity + rng.uniform(0.35, 0.45)
        f['base_height'] = rng.uniform(0.35, 0.55) * intensity + rng.uniform(0.25, 0.35)
        f['roof_angle']  = rng.uniform(0.15, 0.35)
        f['has_flag']    = rng.random() < 0.50 * intensity
        
    elif archetype == 'plant':
        f['crown_shape'] = rng.choice(['round', 'pointed', 'wide'])
        f['stem_width']  = rng.uniform(0.06, 0.15) * intensity + rng.uniform(0.05, 0.08)
        f['crown_size']  = rng.uniform(0.25, 0.40) * intensity + rng.uniform(0.20, 0.28)
        f['leaf_count']  = rng.choice([2, 3, 4])
        
    elif archetype == 'vehicle':
        f['body_width']  = rng.uniform(0.55, 0.80) * intensity + rng.uniform(0.35, 0.45)
        f['body_height'] = rng.uniform(0.20, 0.35) * intensity + rng.uniform(0.15, 0.22)
        f['wheel_count'] = rng.choice([2, 4], p=[0.4, 0.6])
        f['has_cabin']   = rng.random() < 0.70 * intensity
        f['cabin_width'] = rng.uniform(0.25, 0.45) * intensity + rng.uniform(0.15, 0.25)
        
    else:  # blob
        f['width']  = rng.uniform(0.35, 0.60) * intensity + rng.uniform(0.20, 0.30)
        f['height'] = rng.uniform(0.30, 0.55) * intensity + rng.uniform(0.20, 0.28)
        f['spikes'] = rng.random() < 0.30 * intensity
        f['spike_count'] = rng.randint(3, 9)
    
    return f

# ============================================================
# DESENHADORES DE SILHUETA COM FEATURES
# ============================================================
def draw_creature(w, h, rng, f):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx = w // 2
    
    body_rx = int(w * f['body_width'] / 2)
    body_ry = int(h * f['body_height'] / 2)
    body_cy = int(h * f['body_pos_y'])
    
    # Corpo
    d.ellipse([cx-body_rx, body_cy-body_ry, cx+body_rx, body_cy+body_ry], fill=1)
    
    # Cabeça
    head_rx = int(w * f['head_size'])
    head_ry = int(h * f['head_size'] * 0.9)
    head_cx = cx + int(w * f['head_pos'])
    head_cy = body_cy - int(h * 0.08) + int(h * f['neck_angle'])
    d.ellipse([head_cx-head_rx, head_cy-head_ry, head_cx+head_rx, head_cy+head_ry], fill=1)
    
    # Pescoço
    neck_w = int(w * 0.06)
    neck_h = int(h * 0.10)
    neck_cx = (head_cx + cx) // 2
    neck_cy = (head_cy + body_cy) // 2
    d.rectangle([neck_cx-neck_w/2, neck_cy-neck_h/2, neck_cx+neck_w/2, neck_cy+neck_h/2], fill=1)
    
    # Cauda
    if f['has_tail']:
        tail_len = int(w * 0.35 * rng.uniform(0.8, 1.2))
        tail_curve = f['tail_curve']
        pts = []
        for i in range(8):
            t = i / 7.0
            tx = cx - int(w * 0.20) - int(tail_len * t)
            ty = body_cy + int(h * tail_curve * t) - int(h * 0.02 * t)
            pts.append((tx, ty))
        pts.append((pts[-1][0] - int(w*0.05), pts[-1][1] - int(h*0.05)))
        pts.append((pts[-1][0], pts[-1][1] + int(h*0.04)))
        pts.append((pts[0][0], pts[0][1] + int(h*0.06)))
        d.polygon(pts, fill=1)
    
    # Asas
    if f['has_wings']:
        wing_span = int(w * 0.35 * rng.uniform(0.7, 1.3))
        wing_height = int(h * 0.35 * rng.uniform(0.7, 1.3))
        wing_y = body_cy - int(h * 0.12)
        # Asa esquerda
        d.polygon([
            (cx - int(w*0.05), wing_y),
            (cx - wing_span, wing_y - wing_height),
            (cx - int(w*0.05), wing_y - int(h*0.25)),
            (cx + int(w*0.02), wing_y - int(h*0.10)),
        ], fill=1)
        # Asa direita (espelhada)
        d.polygon([
            (cx + int(w*0.05), wing_y),
            (cx + wing_span, wing_y - wing_height),
            (cx + int(w*0.05), wing_y - int(h*0.25)),
            (cx - int(w*0.02), wing_y - int(h*0.10)),
        ], fill=1)
    
    # Pernas
    leg_count = f['leg_count']
    leg_w = int(w * 0.08)
    leg_h = int(h * f['leg_length'])
    leg_y = body_cy + int(h * 0.15)
    if leg_count == 2:
        positions = [cx - int(w*0.12), cx + int(w*0.08)]
    else:
        positions = [cx - int(w*0.18), cx - int(w*0.06), cx + int(w*0.04), cx + int(w*0.16)]
    for lx in positions:
        d.rectangle([lx-leg_w/2, leg_y, lx+leg_w/2, leg_y+leg_h], fill=1)
        d.rectangle([lx-leg_w*0.9, leg_y+leg_h, lx+leg_w*0.9, leg_y+leg_h+int(h*0.03)], fill=1)
    
    # Chifres
    if f['has_horns']:
        horn_count = rng.choice([2, 3])
        for i in range(horn_count):
            hx = head_cx - int(w*0.05) + int(w*0.05) * i
            hy = head_cy - head_ry
            d.polygon([
                (hx, hy),
                (hx - int(w*0.02), hy - int(h*0.08)),
                (hx + int(w*0.02), hy - int(h*0.08)),
            ], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def draw_character(w, h, rng, f):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx = w // 2
    
    head_ry = int(h * f['head_size'])
    head_rx = int(w * f['head_size'] * 0.85)
    head_cy = int(h * f['head_y'])
    d.ellipse([cx-head_rx, head_cy-head_ry, cx+head_rx, head_cy+head_ry], fill=1)
    
    body_w = int(w * f['body_width'])
    body_h = int(h * f['body_height'])
    body_cy = head_cy + int(h * 0.20)
    d.rectangle([cx-body_w/2, body_cy-body_h/2, cx+body_w/2, body_cy+body_h/2], fill=1)
    
    arm_w = int(w * 0.08)
    arm_h = int(h * f['arm_length'])
    for ax in [cx-body_w/2-arm_w/2, cx+body_w/2+arm_w/2]:
        d.rectangle([ax-arm_w/2, body_cy-int(h*0.03), ax+arm_w/2, body_cy-int(h*0.03)+arm_h], fill=1)
    
    leg_w = int(w * 0.10)
    leg_h = int(h * f['leg_length'])
    leg_top = body_cy + body_h/2
    for lx in [cx-int(w*0.06), cx+int(w*0.06)]:
        d.rectangle([lx-leg_w/2, leg_top, lx+leg_w/2, leg_top+leg_h], fill=1)
    
    if f['has_hat']:
        hat_w = int(w * 0.24)
        hat_h = int(h * 0.08)
        d.rectangle([cx-hat_w/2, head_cy-head_ry-hat_h/2, cx+hat_w/2, head_cy-head_ry+hat_h/2], fill=1)
    
    if f['has_cape']:
        cape_w = int(w * 0.30)
        cape_h = int(h * 0.35)
        d.rectangle([cx-cape_w/2, body_cy-int(h*0.05), cx+cape_w/2, body_cy-int(h*0.05)+cape_h], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def draw_item(w, h, rng, f):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w//2, h//2
    scale = f['size_scale']
    
    if f['item_type'] == 'sword':
        blade_w = int(w * 0.07 * scale)
        blade_h = int(h * 0.50 * scale)
        blade_cy = cy - int(h * 0.10)
        d.rectangle([cx-blade_w/2, blade_cy-blade_h/2, cx+blade_w/2, blade_cy+blade_h/2], fill=1)
        d.polygon([(cx-blade_w/2, blade_cy-blade_h/2), (cx+blade_w/2, blade_cy-blade_h/2),
                   (cx, blade_cy-blade_h/2-int(h*0.05*scale))], fill=1)
        guard_w = int(w * 0.28 * scale)
        d.rectangle([cx-guard_w/2, blade_cy+blade_h/2, cx+guard_w/2, blade_cy+blade_h/2+int(h*0.05)], fill=1)
        grip_h = int(h * 0.18 * scale)
        d.rectangle([cx-int(w*0.05), blade_cy+blade_h/2+int(h*0.05), cx+int(w*0.05), blade_cy+blade_h/2+int(h*0.05)+grip_h], fill=1)
    elif f['item_type'] == 'potion':
        r = int(min(w,h) * 0.25 * scale)
        d.ellipse([cx-r, cy+r*0.3, cx+r, cy+r*1.3], fill=1)
        d.rectangle([cx-int(w*0.06), cy-r*0.5, cx+int(w*0.06), cy+r*0.4], fill=1)
        d.rectangle([cx-int(w*0.09), cy-r*0.7, cx+int(w*0.09), cy-r*0.5], fill=1)
    elif f['item_type'] == 'shield':
        sw, sh = int(w*0.55*scale), int(h*0.65*scale)
        d.ellipse([cx-sw/2, cy-sh/2, cx+sw/2, cy+sh/2], fill=1)
        d.rectangle([cx-int(w*0.03), cy-sh/2, cx+int(w*0.03), cy+sh/2], fill=1)
    elif f['item_type'] == 'gem':
        gw, gh = int(w*0.40*scale), int(h*0.50*scale)
        d.polygon([(cx, cy-gh/2), (cx+gw/2, cy), (cx, cy+gh/2), (cx-gw/2, cy)], fill=1)
    else:  # axe
        d.rectangle([cx-int(w*0.04), cy-int(h*0.25), cx+int(w*0.04), cy+int(h*0.25)], fill=1)
        d.polygon([(cx+int(w*0.04), cy-int(h*0.15)), (cx+int(w*0.20), cy),
                   (cx+int(w*0.04), cy+int(h*0.15))], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def draw_building(w, h, rng, f):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx = w // 2
    
    base_w = int(w * f['base_width'])
    base_h = int(h * f['base_height'])
    base_cy = h - int(h * 0.15) - base_h // 2
    d.rectangle([cx-base_w/2, base_cy-base_h/2, cx+base_w/2, base_cy+base_h/2], fill=1)
    
    roof_h = int(h * f['roof_angle'])
    d.polygon([(cx-base_w/2-int(w*0.05), base_cy-base_h/2),
               (cx, base_cy-base_h/2-roof_h),
               (cx+base_w/2+int(w*0.05), base_cy-base_h/2)], fill=1)
    
    tower_w = int(w * 0.13)
    tower_h = int(h * 0.55)
    tower_count = f['tower_count']
    if tower_count == 1:
        positions = [cx]
    elif tower_count == 2:
        positions = [cx-base_w/2+tower_w/2, cx+base_w/2-tower_w/2]
    else:
        positions = [cx-base_w/2+tower_w/2, cx, cx+base_w/2-tower_w/2]
    
    for tx in positions:
        ty = base_cy - base_h/2 - tower_h/2 + int(h*0.10)
        d.rectangle([tx-tower_w/2, ty-tower_h/2, tx+tower_w/2, ty+tower_h/2], fill=1)
        d.polygon([(tx-tower_w/2, ty-tower_h/2), (tx, ty-tower_h/2-int(h*0.10)),
                   (tx+tower_w/2, ty-tower_h/2)], fill=1)
    
    if f['has_flag']:
        flag_x = positions[0] if tower_count >= 1 else cx
        flag_y = base_cy - base_h/2 - tower_h - int(h*0.08)
        d.rectangle([flag_x-1, flag_y, flag_x+1, flag_y+int(h*0.15)], fill=1)
        d.rectangle([flag_x+1, flag_y, flag_x+int(w*0.08), flag_y+int(h*0.05)], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def draw_plant(w, h, rng, f):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx = w // 2
    
    stem_w = int(w * f['stem_width'])
    stem_h = int(h * 0.30)
    stem_cy = h - int(h * 0.10) - stem_h // 2
    d.rectangle([cx-stem_w/2, stem_cy-stem_h/2, cx+stem_w/2, stem_cy+stem_h/2], fill=1)
    
    crown_r = int(min(w, h) * f['crown_size'])
    crown_cy = stem_cy - stem_h/2 - crown_r * 0.4
    
    if f['crown_shape'] == 'round':
        d.ellipse([cx-crown_r, crown_cy-crown_r, cx+crown_r, crown_cy+crown_r], fill=1)
    elif f['crown_shape'] == 'pointed':
        d.polygon([(cx, crown_cy-crown_r*1.3), (cx+crown_r, crown_cy+crown_r*0.5),
                   (cx-crown_r, crown_cy+crown_r*0.5)], fill=1)
    else:  # wide
        for i in range(f['leaf_count']):
            angle = (i / f['leaf_count']) * 360
            lx = cx + int(crown_r * 0.7 * np.cos(np.radians(angle)))
            ly = crown_cy + int(crown_r * 0.5 * np.sin(np.radians(angle)))
            d.ellipse([lx-crown_r*0.6, ly-crown_r*0.5, lx+crown_r*0.6, ly+crown_r*0.5], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def draw_vehicle(w, h, rng, f):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx = w // 2
    cy = h // 2 + int(h * 0.05)
    
    body_w = int(w * f['body_width'])
    body_h = int(h * f['body_height'])
    d.rectangle([cx-body_w/2, cy-body_h/2, cx+body_w/2, cy+body_h/2], fill=1)
    
    if f['has_cabin']:
        cab_w = int(w * f['cabin_width'])
        cab_h = int(h * 0.18)
        cab_cy = cy - body_h/2 - cab_h/2 + int(h*0.02)
        d.rectangle([cx-cab_w/2-int(w*0.05), cab_cy-cab_h/2, cx+cab_w/2-int(w*0.05), cab_cy+cab_h/2], fill=1)
    
    wheel_r = int(min(w, h) * 0.07)
    wheel_y = cy + body_h/2
    wheel_count = f['wheel_count']
    if wheel_count == 2:
        positions = [cx-int(w*0.22), cx+int(w*0.15)]
    else:
        positions = [cx-int(w*0.25), cx-int(w*0.08), cx+int(w*0.05), cx+int(w*0.22)]
    for wx in positions:
        d.ellipse([wx-wheel_r, wheel_y-wheel_r, wx+wheel_r, wheel_y+wheel_r], fill=1)
    
    return np.array(mask, dtype=np.uint8)

def draw_blob(w, h, rng, f):
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = w//2, h//2
    
    rx = int(w * f['width'])
    ry = int(h * f['height'])
    d.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=1)
    
    if f['spikes']:
        for i in range(f['spike_count']):
            angle = (i / f['spike_count']) * 2 * np.pi
            sx = cx + int(rx * 1.1 * np.cos(angle))
            sy = cy + int(ry * 1.1 * np.sin(angle))
            d.polygon([(sx, sy),
                       (sx + int(rx*0.15*np.cos(angle+1.5)), sy + int(ry*0.15*np.sin(angle+1.5))),
                       (sx + int(rx*0.15*np.cos(angle-1.5)), sy + int(ry*0.15*np.sin(angle-1.5)))],
                      fill=1)
    
    return np.array(mask, dtype=np.uint8)

ARCHETYPE_DRAWERS = {
    'creature': draw_creature,
    'character': draw_character,
    'item': draw_item,
    'building': draw_building,
    'plant': draw_plant,
    'vehicle': draw_vehicle,
    'blob': draw_blob,
}

# ============================================================
# DETALHES (olhos, sombra, highlight)
# ============================================================
def compute_details(base_mask, rng, archetype):
    h, w = base_mask.shape
    ys = np.linspace(0.0, 1.0, h)[:, None] * np.ones((h, w))
    shadow = (base_mask == 1) & (ys > rng.uniform(0.55, 0.70))
    highlight = (base_mask == 1) & (ys < rng.uniform(0.20, 0.32))
    
    eye_mask = np.zeros_like(base_mask, dtype=bool)
    if archetype in ('creature', 'character') and rng.random() < 0.90:
        upper = base_mask.copy()
        upper[h//2:, :] = 0
        cols = np.any(upper > 0, axis=0)
        if np.any(cols):
            left = np.argmax(cols)
            right = w - 1 - np.argmax(cols[::-1])
            center_x = (left + right) // 2
            width_span = right - left
            rows = np.where(np.any(upper > 0, axis=1))[0]
            if len(rows) > 0:
                eye_y = int(rows[0] + (rows[-1] - rows[0]) * rng.uniform(0.35, 0.50))
                eye_offset = int(width_span * rng.uniform(0.12, 0.20))
                eye_r = max(1, int(width_span * rng.uniform(0.05, 0.08)))
                for ex in [center_x - eye_offset, center_x + eye_offset]:
                    y0, y1 = max(0, eye_y-eye_r), min(h, eye_y+eye_r+1)
                    x0, x1 = max(0, ex-eye_r), min(w, ex+eye_r+1)
                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            if (xx-ex)**2 + (yy-eye_y)**2 <= eye_r**2:
                                eye_mask[yy, xx] = True
    
    highlight = highlight & ~eye_mask
    shadow = shadow & ~eye_mask & ~highlight
    return shadow.astype(np.uint8), highlight.astype(np.uint8), eye_mask.astype(np.uint8)

# ============================================================
# OUTLINE + DITHER + QUANTIZE
# ============================================================
def apply_outline(img_rgb, base_mask, outline_color, thickness=1):
    struct = np.ones((3,3), dtype=bool)
    dilated = ndimage.binary_dilation(base_mask.astype(bool), struct, iterations=thickness)
    outline = dilated & ~base_mask.astype(bool)
    result = img_rgb.copy()
    result[outline] = outline_color
    return result

def apply_dither_and_quantize(img_rgb, palette_hex, dither_name):
    palette = np.array([hex_to_rgb(c) for c in palette_hex], dtype=float)
    h, w, _ = img_rgb.shape
    img_f = img_rgb.astype(float)
    mat = DITHER_MATRICES.get(dither_name)
    
    if mat is None:
        flat = img_f.reshape(-1, 3)
        diff = flat[:, None, :] - palette[None, :, :]
        idx = np.argmin(np.sum(diff*diff, axis=2), axis=1)
        return palette[idx].reshape(h, w, 3).astype(np.uint8)
    
    mh, mw = mat.shape
    th = np.tile(mat, (int(np.ceil(h/mh)), int(np.ceil(w/mw))))[:h, :w]
    perturbed = np.clip(img_f + th[..., None] * 32.0, 0, 255)
    flat = perturbed.reshape(-1, 3)
    diff = flat[:, None, :] - palette[None, :, :]
    idx = np.argmin(np.sum(diff*diff, axis=2), axis=1)
    return palette[idx].reshape(h, w, 3).astype(np.uint8)

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def generate_pixel_art(prompt, width=64, height=64, palette_size=16,
                       dithering='none', outline='none', style='auto',
                       seed=0, scale=1, archetype=None, variation='medium'):
    rng = make_rng(seed_from_prompt(prompt, seed))
    
    if style == 'auto' or style not in STYLES:
        available = list(STYLES.keys())
        style = available[seed_from_prompt(prompt, seed) % len(available)]
    palette = list(STYLES[style])
    while len(palette) < max(4, palette_size):
        palette.append(palette[len(palette) % len(STYLES[style])])
    palette = palette[:max(4, palette_size)]
    
    if archetype is None:
        archetype = detect_archetype(prompt)
    
    # FEATURES MUTÁVEIS — cada seed gera forma diferente!
    features = roll_features(rng, archetype, variation)
    drawer = ARCHETYPE_DRAWERS.get(archetype, draw_blob)
    base_mask = drawer(width, height, rng, features)
    
    shadow, highlight, eyes = compute_details(base_mask, rng, archetype)
    
    bg_color        = hex_to_rgb(palette[0])
    shadow_color    = hex_to_rgb(palette[1 % len(palette)])
    base_color      = hex_to_rgb(palette[2 % len(palette)])
    highlight_color = hex_to_rgb(palette[-1])
    eye_color       = hex_to_rgb(palette[0])
    
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    body = (base_mask == 1) & (shadow == 0) & (highlight == 0) & (eyes == 0)
    img[body] = base_color
    img[shadow == 1] = shadow_color
    img[highlight == 1] = highlight_color
    img[eyes == 1] = eye_color
    
    if outline not in (None, 'none', 'None', '', 'off', 'false'):
        if outline == 'black':    oc = (0, 0, 0)
        elif outline == 'white':  oc = (255, 255, 255)
        elif outline == 'dark':   oc = hex_to_rgb(palette[0])
        else:
            try: oc = hex_to_rgb(outline)
            except: oc = (0, 0, 0)
        img = apply_outline(img, base_mask, oc, thickness=1)
    
    img = apply_dither_and_quantize(img, palette, dithering)
    
    if scale and int(scale) > 1:
        pil = Image.fromarray(img).resize((width*int(scale), height*int(scale)), Image.NEAREST)
    else:
        pil = Image.fromarray(img)
    
    pil.info['style'] = style
    pil.info['archetype'] = archetype
    pil.info['variation'] = variation
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
    print("🎨 PIXEL ART AI GENERATOR v3 — com mutação de características")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--archetype', default=None, choices=list(ARCHETYPE_DRAWERS.keys()) + [None])
    parser.add_argument('--variation', default='medium', choices=['low', 'medium', 'heavy'],
                        help="low=varia pouco, medium=equilibrado, heavy=máxima variação")
    parser.add_argument('--outdir', default='pixel_art_output')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    kwargs = dict(width=args.width, height=args.height,
                  palette_size=args.palette_size, dithering=args.dithering,
                  outline=args.outline, style=args.style, scale=args.scale,
                  variation=args.variation)
    if args.archetype is not None:
        kwargs['archetype'] = args.archetype
    
    if args.batch or args.count > 1:
        images = batch_generate(args.prompt, count=args.count, **kwargs)
    else:
        if args.seed is not None:
            kwargs['seed'] = args.seed
        images = [generate_pixel_art(args.prompt, **kwargs)]
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_prompt = "".join(c for c in (args.prompt or '') if c.isalnum() or c in ' _-').strip().replace(' ', '_')[:40] or 'pixel'
    
    for i, img in enumerate(images):
        style_used = img.info.get('style', args.style)
        arch_used = img.info.get('archetype', '?')
        fname = f"{safe_prompt}_{style_used}_{arch_used}_{ts}_{i:03d}.png"
        path = os.path.join(args.outdir, fname)
        img.save(path, optimize=True)
        print(f"  ✓ [{i+1}/{len(images)}] {fname}  (style={style_used}, arch={arch_used})")
    
    print(f"\n✨ {len(images)} imagem(ns) gerada(s) em '{args.outdir}/'")
    return [os.path.join(args.outdir, f) for f in os.listdir(args.outdir) if f.endswith('.png')]

if __name__ == "__main__":
    main()
