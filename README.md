# 🧠 IA Pixel Art Generator

Autoencoder condicional real (backprop em numpy) rodando em GitHub Actions Linux.

## ⚠️ ORDEM DE USO

1. **PRIMEIRO**: Treine o modelo (Actions → 🧠 Treinar) — 30-60 min
2. **DEPOIS**: Gere pixel art (Actions → 🎨 Gerar) — 1-2 min

## 🧠 Arquitetura

```
PROMPT → [Condition 64d] → ENCODER → LATENT 128d → DECODER → PATCH 8x8x3 → IMAGEM
```

- Autoencoder condicional com backpropagation manual
- Otimizador Adam (lr=0.001)
- 10.000 épocas, 5000 patches sintéticos
- Geração usa pesos treinados (não procedural)

## 🚀 Uso

### Treinar
Actions → 🧠 Treinar Modelo → Run (10000 épocas, ~40 min)

### Gerar
Actions → 🎨 Gerar Pixel Art → Run (prompt, estilo, resolução)

## 💻 Local

```bash
pip install -r requirements.txt
python train.py --epochs 10000
python pixel_art_generator.py --batch --prompt "dragão 16-bit" --count 5
```

## 📁 Estrutura

```
├── .github/workflows/
├── pixel_art_output/
├── models/checkpoints/   (latest.npz)
├── knowledge_base.json
├── pixel_art_generator.py
├── train.py
├── requirements.txt
└── README.md
```

## 💡 Prompts

- "herói com espada estilo NES"
- "dragão 16-bit SNES"
- "floresta Game Boy"
- "nave cyberpunk neon"
- "dungeon monocromática"

## 📊 Tempo (Linux)

| Operação | Tempo |
|---|---|
| Treino 10000 épocas | ~40 min |
| Geração 1 imagem | ~30s |
| Geração 5 imagens | ~2 min |

MIT © Valentin