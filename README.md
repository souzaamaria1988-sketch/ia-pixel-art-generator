# 🎨 IA Pixel Art Generator - VAE Condicional CORRIGIDO

Sistema de geração de pixel art usando **Variational Autoencoder (VAE) Condicional** com condicionamento REAL do prompt.

## ⚠️ CORREÇÕES APLICADAS

### 1. Decoder Condicional Real ✅
- **Antes**: Decoder ignorava condição (apenas `z`)
- **Depois**: Decoder recebe `concat(z, cond_proj)` igual ao encoder
- **Resultado**: Prompt agora controla o conteúdo gerado!

### 2. Prompt → Condição Real ✅
- `interpret_prompt()` converte texto em vetor de condição
- Condição usada no decoder para controlar geração
- Paleta e tipo de conteúdo são respeitados

### 3. Checkpoint Completo ✅
- Salva pesos + estados do Adam (m, v, t)
- Retomada perfeita do treino

### 4. KL Warm-up ✅
- Beta aumenta gradualmente: 0.0001 → 0.001 em 1000 épocas
- Evita posterior collapse

### 5. Coerência Global ✅
- Todos os patches usam mesma condição
- z_base compartilhado + variações locais
- Overlap entre patches para blending

## 🚀 Como Usar

### 1. Treinar o Modelo

```bash
python train.py --epochs 5000
```

### 2. Gerar Pixel Art

```bash
python pixel_art_generator.py --batch \
  --prompt "herói com espada estilo NES" \
  --width 64 --height 64 \
  --count 5
```

## 🎯 Exemplos de Prompts

```bash
--prompt "herói estilo NES"
--prompt "espada neon"
--prompt "monstro dark"
--prompt "grama e árvore"
```

## 📊 Comparação: Antes vs Depois

| Aspecto | Antes | Depois |
|---|---|---|
| **Decoder** | `decode(z)` | `decode(z, cond)` |
| **Prompt** | Ignorado | Usado como condição real |
| **KL** | Beta fixo 0.001 | Warm-up 0.0001→0.001 |
| **Checkpoint** | Só pesos | Pesos + Adam states |
| **Coerência** | Patches independentes | z_base + overlap |
| **Controle** | Aleatório | Prompt controla paleta/tipo |

---

**Versão:** 2.0 (Correção Arquitetural)  
**Status:** ✅ Funcional e Testado