# 🧠 Pixel Art VAE Generator

**Variational Autoencoder** real (KL divergence + reparameterization trick) rodando em GitHub Actions Linux.

## ⚠️ ORDEM DE USO

1. **TREINAR**: Actions → 🧠 Treinar VAE (~30-50 min para 5000 épocas)
2. **GERAR**: Actions → 🎨 Gerar Pixel Art (~1-2 min)

## 🧠 Arquitetura VAE

```
INPUT (256d) ──→ ENCODER ──→ (μ, σ) ──→ z = μ + σ * ε ──→ DECODER ──→ PATCH 8x8x3
                                  ↑                          ↓
                                  └──── KL divergence ───────┘
                                        (organiza espaço)
```

## 📦 Dataset Realista (8000 patches)

O modelo é treinado em **patches que parecem pixel art de verdade**:
- ✓ Sprites simétricos (estilo Space Invader)
- ✓ Humanoides (cabeça + corpo)
- ✓ Tiles de grama com textura
- ✓ Tiles de água com ondas
- ✓ Tiles de pedra/tijolos
- ✓ Itens (espadas, poções, escudos)
- ✓ Árvores e vegetação
- ✓ Nuvens e céu
- ✓ Bordas/outline patterns
- ✓ Gradientes e formas geométricas

6 paletas: NES, Game Boy, dark, neon, CGA, nature

## 🎨 Parâmetros de Geração

| Parâmetro | Descrição | Recomendado |
|---|---|---|
| noise_scale | Variabilidade | 1.0 |
| coherence | Coerência entre patches | 0.3-0.5 |
| palette_size | Cores na paleta | 16 |
| dithering | Método | bayer4x4 |

## 💡 Prompts Testados

- "herói com espada estilo NES"
- "dragão 16-bit SNES"
- "floresta Game Boy"
- "nave cyberpunk neon"
- "dungeon monocromática"
- "cidade à noite"

## 📊 Tempos (Linux)

| Operação | Tempo |
|---|---|
| Treino 2500 épocas | ~15 min |
| Treino 5000 épocas | ~30 min |
| Treino 10000 épocas | ~60 min |
| Geração 1 imagem | ~30s |
| Geração 5 imagens | ~2 min |

## 🎯 Diferença do Autoencoder anterior

| Autoencoder (antes) | VAE (agora) |
|---|---|
| Espaço latente caótico | Espaço latente **organizado** |
| Gera ruído fora da distribuição | Gera samples **coerentes** |
| Loss ~0.024 (MSE só) | Loss ~0.08 (MSE + KL honesto) |
| Sem interpolação | Interpolável entre imagens |

MIT © Valentin