# 🧠 Pixel Art VAE v2.0

VAE condicional real (numpy puro) para geração de pixel art, com melhorias robustas sobre a v1.0.

## ⚠️ QUEBRA DE COMPATIBILIDADE

A v2.0 tem `arch_version="2.0"`. Checkpoints antigos da v1.0 serão rejeitados com mensagem clara. Treine novamente para obter um modelo compatível.

## 🎯 Melhorias sobre v1.0

### 1. Gradiente KL corrigido
O gradiente KL agora recebe `beta_kl` como parâmetro e é aplicado **apenas** aos componentes `mu` e `log_var`, sem contaminar o gradiente de reconstrução do decoder.

### 2. `config.py` - fonte única da verdade
Dimensões, paletas, tipos, humores e traits definidos em um único lugar. `train.py` e `pixel_art_generator.py` importam o mesmo config.

### 3. 12 tipos de pixel art padronizados
`sprite, character, grass, water, stone, sword, potion, tree, cloud, outline, gradient, geometric`

Índices compartilhados entre dataset de treino, interpretador de prompts e gerador.

### 4. `interpret_prompt()` estruturado
Detecta **paleta + tipo + humor + traits** em uma estrutura organizada. Exemplo:
- `"espada neon"` → palette=neon, type=sword, mood=neon
- `"monstro dark"` → palette=dark, type=character, mood=dark, traits=[symmetric, outline]

### 5. Dimensões não-múltiplas de 8
`math.ceil` substituiu `//` — imagens como 70x70, 100x64 agora são preenchidas completamente.

### 6. Overlap real com blending linear
Parâmetro `--overlap` (0-4) agora faz blending real:
- Acumulador de pixels + matriz de pesos
- Máscara senoidal (peso maior no centro do patch)
- Bordas tratadas com crop

### 7. Validação + best checkpoint
- Split 10% para validação
- Salva `best.npz` quando val_loss melhora
- `latest.npz` sempre atualizado
- Logs de train/val/recon/kl/edge/palette

### 8. Loss enriquecida
```
total = MSE + 0.05 * edge_loss + 0.02 * palette_loss + beta_kl(t) * KL
```
- **Edge loss**: Sobel entre pred e target (preserva outlines)
- **Palette loss**: distância média à paleta mais próxima
- **KL warm-up**: linear 0 → BETA_KL em 500 épocas

### 9. Versionamento e segurança
- Campo `arch_version` em cada checkpoint
- Mensagem clara quando checkpoint é incompatível
- Validação de campos obrigatórios no load

### 10. 10 testes automáticos
`tests.py` cobre:
1. train.py inicia
2. Forward executa
3. Gradientes com dimensões corretas
4. Save/load de checkpoint
5. Prompts diferentes → condições diferentes
6. 6 paletas reconhecidas
7. Imagens 64x64, 70x70, 100x64 sem áreas vazias
8. Overlap altera blending
9. Carrega latest.npz
10. beta_kl afeta só gradientes KL

## 📦 Instalação

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 🧪 Testes

\`\`\`bash
python -m pytest tests.py -v
\`\`\`

## 🧠 Treinar

\`\`\`bash
python train.py --epochs 5000 --batch-size 64
\`\`\`

Ou via Actions: 🧠 Treinar VAE Pixel Art (30-50 min em ubuntu-latest).

## 🎨 Gerar

\`\`\`bash
python pixel_art_generator.py --batch --prompt "dragao estilo NES" \
  --width 64 --height 64 --palette-size 16 \
  --noise-scale 1.0 --coherence 0.3 --overlap 2
\`\`\`

Ou via Actions: 🎨 Gerar Pixel Art.

## 📊 Estrutura da condição (64d)

| Slice | Conteúdo | Bits |
|-------|----------|------|
| [0:6]   | paleta (one-hot)       | 6  |
| [6:18]  | tipo (one-hot)         | 12 |
| [18:24] | humor (one-hot)        | 6  |
| [24:32] | traits (multi-hot)     | 8  |
| [32:64] | features contínuas     | 32 |

## 📁 Estrutura

\`\`\`
ia-pixel-art-generator/
├── config.py              # fonte única da verdade
├── train.py               # treinamento VAE v2.0
├── pixel_art_generator.py # gerador v2.0
├── tests.py               # 10 testes
├── knowledge_base.json    # RAG
├── requirements.txt
├── README.md
├── .github/workflows/
│   ├── generate_pixel_art.yml
│   ├── train_model.yml
│   └── tests.yml
├── pixel_art_output/
├── pixel_art_input/
└── models/
    ├── checkpoints/   (latest.npz + best.npz + model_epoch_*.npz)
    ├── samples/       (PNG de inspeção visual)
    ├── manifest.json
    └── training_log.json
\`\`\`

## ⏱️ Tempos (ubuntu-latest)

| Operação | Tempo |
|---|---|
| Treino 2500 épocas | ~15 min |
| Treino 5000 épocas | ~30 min |
| Treino 10000 épocas | ~60 min |
| Testes | ~2 min |
| Geração 1 imagem | ~30s |
| Geração 5 imagens | ~2 min |

## 🎯 Hiperparâmetros recomendados

- `noise_scale`: 1.0 (respeita distribuição N(0,1))
- `coherence`: 0.3-0.5 (patches vizinhos compartilham z_base)
- `overlap`: 2 (bom tradeoff qualidade/velocidade)
- `palette_size`: 16 (padrão NES)
- `dithering`: bayer4x4

## ⚠️ Limitações

1. **Dataset sintético**: os patches são gerados proceduralmente, não a partir de imagens reais. O modelo aprende padrões estruturais, mas não "reconhece" objetos semânticos complexos.
2. **Decoder pequeno**: 2 camadas MLP. Capacidade limitada para alta complexidade.
3. **Sem atenção**: não há mecanismo de atenção global sobre a imagem.
4. **Paleta só no pós-processamento**: a paleta não é forçada durante a geração — só aplicada via dithering/quantização.
5. **Prompt não é embedding**: palavras raras que não estão em `KEYWORDS` são ignoradas.
6. **Não há fine-tuning incremental**: retreinar do zero ainda é a melhor opção.

Para resultados profissionais, o próximo passo é adicionar imagens reais em `pixel_art_input/` e treinar com elas (reconstrução supervisionada).

## 📜 Licença

MIT © Valentin
