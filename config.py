"""
Configuração compartilhada entre treino e geração.
Única fonte da verdade para dimensões, paletas, tipos e condições.

ARQUITETURA v2.0 — se alterar dimensões, incremente ARCH_VERSION
e antigos checkpoints serão invalidados com mensagem clara.
"""
import numpy as np

# ============================================================
# VERSIONAMENTO
# ============================================================
ARCH_VERSION = "2.0"

# ============================================================
# DIMENSÕES DO MODELO
# ============================================================
INPUT_DIM = 256
LATENT_DIM = 128
OUTPUT_DIM = 192       # 8x8x3
HIDDEN_DIM = 512
CONDITION_DIM = 64
PATCH_SIZE = 8

# ============================================================
# PALETAS (compartilhadas treino <-> geracao)
# A ORDEM define o indice one-hot na condicao [0:6]
# ============================================================
PALETTE_NAMES = ["nes", "gameboy", "dark", "neon", "cga", "nature"]

PALETTES_HEX = {
    "nes": ["#7C7C7C","#0000FC","#0000BC","#4428BC","#940084","#A80020","#A81000",
            "#881400","#503000","#007800","#006800","#005800","#004058","#000000",
            "#BCBCBC","#0078F8","#0058F8","#6844FC","#D800CC","#E40058","#F83800",
            "#E45C10","#AC7C00","#00B800","#00A800","#00A844","#008888","#F8F8F8",
            "#3CBCFC","#6888FC","#9878F8","#F878F8","#F85898","#F87858","#FCA044",
            "#F8B800","#B8F818","#58D854","#58F898","#00E8D8","#787878","#FCFCFC",
            "#A4E4FC","#B8B8F8","#D8B8F8","#F8B8F8","#F8A4C0","#F0D0B0","#FCE0A8",
            "#F8D878","#D8F878","#B8F8B8","#B8F8D8","#00FCFC"],
    "gameboy": ["#0f380f","#306230","#8bac0f","#9bbc0e"],
    "dark": ["#0A0A0A","#1A1A1A","#2C2C2C","#3D3D3D","#4F4F4F",
             "#1A0000","#2A0000","#3A0000","#00001A","#00002A","#00003A",
             "#1A001A","#2A002A","#3A003A","#0A1A0A","#1A2A1A",
             "#800000","#A02020"],
    "neon": ["#FF006E","#FF4DA6","#FF00FF","#CC00FF","#8B00FF","#00F0FF",
             "#00FFFF","#39FF14","#CCFF00","#FFF200","#FF8C00","#FF1744",
             "#D500F9","#651FFF","#00E5FF","#76FF03","#FF8800"],
    "cga": ["#000000","#55FFFF","#FF55FF","#FFFFFF"],
    "nature": ["#2C7B2C","#4CA84C","#6BC86B","#8B5A2B","#A0826F","#E8C170",
               "#1A5A1A","#3A8A3A","#5AAA5A","#7ACA7A"],
    # Extras apenas para o gerador (nao usados em treino)
    "monochrome": ["#000000","#555555","#AAAAAA","#FFFFFF"],
    "sepia": ["#3B2414","#6B4423","#A67B5B","#D4A574","#E8C8A0","#F5E6D3","#F8F0E3","#FFFFFF"],
    "pastel": ["#FFD1DC","#FFB7C5","#FDFD96","#B5EAD7","#C7CEEA","#E2F0CB","#FFDAC1","#B5D8EB","#F0E6FF","#FFE4E1","#DCD0FF","#C1E1C1","#FFEBCD","#E0FFFF","#F5DEB3","#FFFACD"],
    "bright": ["#FFFFFF","#FFFFAA","#AAFFFF","#FFAAFF","#AAFFAA","#FFAAAA","#AAAAFF","#FFCC00","#00CCFF","#FF00CC","#00FFCC","#CCFF00","#CC00FF","#00FF00","#FF0000","#0000FF"],
}

# Subconjuntos usados em treino (RGB normalizado)
TRAIN_PALETTES_RGB = {
    'nes':    np.array([[0x7C,0x7C,0x7C],[0x00,0x00,0xFC],[0xA8,0x10,0x00],
                        [0x00,0xB8,0x00],[0xF8,0x38,0x00],[0xFC,0xFC,0xFC],
                        [0x58,0xD8,0x54],[0xF8,0xB8,0x00]], dtype=np.float32)/255.,
    'gameboy': np.array([[0x0f,0x38,0x0f],[0x30,0x62,0x30],
                         [0x8b,0xac,0x0f],[0x9b,0xbc,0x0e]], dtype=np.float32)/255.,
    'dark':    np.array([[0x0A,0x0A,0x0A],[0x1A,0x1A,0x1A],[0x2C,0x2C,0x2C],
                         [0x3D,0x3D,0x3D],[0x4F,0x4F,0x4F],[0x80,0x00,0x00],
                         [0xA0,0x20,0x20]], dtype=np.float32)/255.,
    'neon':    np.array([[0xFF,0x00,0x6E],[0x00,0xF0,0xFF],[0x39,0xFF,0x14],
                         [0xFF,0xF2,0x00],[0xBC,0x13,0xFE],[0xFF,0x88,0x00]],
                        dtype=np.float32)/255.,
    'cga':     np.array([[0,0,0],[0x55,0xFF,0xFF],[0xFF,0x55,0xFF],
                         [0xFF,0xFF,0xFF]], dtype=np.float32)/255.,
    'nature':  np.array([[0x2C,0x7B,0x2C],[0x4C,0xA8,0x4C],[0x6B,0xC8,0x6B],
                         [0x8B,0x5A,0x2B],[0xA0,0x82,0x6F],[0xE8,0xC1,0x70]],
                        dtype=np.float32)/255.,
}

# ============================================================
# TIPOS DE PIXEL ART (12 tipos)
# A ORDEM define o indice one-hot na condicao [6:18]
# ============================================================
PIXEL_ART_TYPES = [
    "sprite",       # 0
    "character",    # 1
    "grass",        # 2
    "water",        # 3
    "stone",        # 4
    "sword",        # 5
    "potion",       # 6
    "tree",         # 7
    "cloud",        # 8
    "outline",      # 9
    "gradient",     # 10
    "geometric",    # 11
]
PIXEL_ART_TYPE_TO_IDX = {name: i for i, name in enumerate(PIXEL_ART_TYPES)}

# ============================================================
# HUMORES / AMBIENTES (6) - indices one-hot em [18:24]
# ============================================================
MOODS = ["dark", "neon", "nature", "medieval", "bright", "retro"]
MOOD_TO_IDX = {m: i for i, m in enumerate(MOODS)}

# ============================================================
# CARACTERISTICAS (multi-hot em [24:32])
# ============================================================
TRAITS = ["outline", "small", "large", "bright", "symmetric", "textured", "glow", "wet"]
TRAIT_TO_IDX = {t: i for i, t in enumerate(TRAITS)}

# ============================================================
# ESTRUTURA DA CONDICAO (CONDITION_DIM = 64)
# ============================================================
COND_SLICE_PAL   = slice(0, 6)
COND_SLICE_TYPE  = slice(6, 18)
COND_SLICE_MOOD  = slice(18, 24)
COND_SLICE_TRAIT = slice(24, 32)
COND_SLICE_FEAT  = slice(32, 64)

assert CONDITION_DIM == 64
assert len(PALETTE_NAMES) == 6
assert len(PIXEL_ART_TYPES) == 12
assert len(MOODS) == 6
assert len(TRAITS) == 8

# ============================================================
# HIPERPARAMETROS DE TREINAMENTO
# ============================================================
BETA_KL = 0.001           # peso maximo da KL
BETA_KL_WARMUP = 500      # epocas de warm-up linear de 0 -> BETA_KL
BETA_EDGE = 0.05          # peso da edge loss
BETA_PALETTE = 0.02       # peso da palette loss
LEARNING_RATE = 0.0008
VAL_SPLIT = 0.1
