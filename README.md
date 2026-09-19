# 🎨 IA Pixel Art Generator

Gerador de **Pixel Art** com IA usando MoE (64 experts), RAG e algoritmos procedurais, rodando em **GitHub Actions (Linux)**.

## ⚡ Linux vs macOS

Este projeto roda em **ubuntu-latest** (Linux) em vez de macOS:
- ~3x mais rápido em workloads CPU-bound
- ~10x mais barato em minutos de CI
- 7GB RAM suficientes para nosso modelo
- Treino completo: **5-15 min** (vs 3h+ em macOS)

## 🚀 Como usar

1. Vá em **Actions** → **🎨 Gerar Pixel Art**
2. Clique em **Run workflow**
3. Digite o prompt: `um herói com espada estilo NES`
4. Aguarde 2-5 min → imagens aparecem em `pixel_art_output/`

## 💻 Localmente

```bash
pip install -r requirements.txt
python pixel_art_generator.py
```

## 🎓 Treinar o modelo

1. Actions → **🎓 Treinar Modelo**
2. Escolha épocas (1000 padrão) e ative "fast mode"
3. Aguarde ~5-15 min

## 🎨 64 Experts

NES, SNES, Game Boy, GBA, CGA, Atari, Commodore 64, Amiga, isométrico, top-down, side-scroller, plataforma, RPG, ação, aventura, puzzle, shooter, luta, corrida, horror, fantasia, sci-fi, medieval, steampunk, cyberpunk, pós-apocalíptico, subaquático, espaço, faroeste, noir, cartoon, anime, minimalista, monocromático, sépia, neon, pastel, personagens, inimigos, itens, tilesets e mais.

## 💡 Exemplos de prompts

- "um herói com espada estilo NES 8-bit"
- "dragão cuspindo fogo 16-bit SNES"
- "floresta com árvores pixel art Game Boy"
- "dungeon com paredes de pedra monocromático"
- "nave espacial cyberpunk neon"
- "castelo medieval isométrico"
- "slime verde simples 32x32"
- "tileset de terreno com grama e água"

## 🔧 Técnicas

- **Paletas**: NES (54 cores), Game Boy (4), CGA (4), monocromática, sépia, pastel, neon, escura, clara
- **Dithering**: Bayer 2x2/4x4/8x8, Floyd-Steinberg, Atkinson
- **Algoritmos**: Perlin Noise, Cellular Automata, Midpoint Displacement
- **Pós**: Outline, nearest-neighbor scaling

## 📁 Estrutura

```
ia-pixel-art-generator/
├── .github/workflows/
│   ├── train_model.yml
│   └── generate_pixel_art.yml
├── pixel_art_output/
├── pixel_art_input/
├── models/layers/
├── knowledge_base.json
├── pixel_art_generator.py
├── train.py
├── requirements.txt
└── README.md
```

## ⚙️ Parâmetros

| Parâmetro | Valores | Default |
|---|---|---|
| width/height | 16, 32, 64, 128, 256 | 64 |
| palette_size | 2, 4, 8, 16, 32, 64 | 16 |
| dithering | none, bayer2x2, bayer4x4, bayer8x8, floyd_steinberg, atkinson | bayer4x4 |
| outline | none, dark, light, auto | none |
| scale | 1, 2, 4, 8 | 4 |

## 💾 Economia de RAM

- Layer-wise processing (1 camada por vez)
- INT8 quantization (4x menor)
- gc.collect() após cada camada
- ~300MB RAM ativa

## 📜 Licença

MIT © Valentin