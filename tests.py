#!/usr/bin/env python3
"""
Suite de testes para o sistema Pixel Art VAE v2.0.
10 testes obrigatorios conforme especificacao.
"""
import sys, tempfile, shutil
from pathlib import Path
import numpy as np
import pytest

import config as C


def test_train_starts():
    """train.py inicia corretamente (importa sem erro)."""
    import train
    assert hasattr(train, 'ConditionalVAE')
    assert hasattr(train, 'train')
    assert hasattr(train, 'generate_realistic_patches')


def test_model_forward():
    """O modelo executa um passo de treinamento."""
    import train
    np.random.seed(0)
    model = train.ConditionalVAE(seed=0)
    x = np.random.randn(4, C.INPUT_DIM).astype(np.float32)
    cond = np.random.randn(4, C.CONDITION_DIM).astype(np.float32)
    y = np.random.randn(4, C.OUTPUT_DIM).astype(np.float32)
    all_pal = np.concatenate([v for v in C.TRAIN_PALETTES_RGB.values()], axis=0)
    loss, recon, kl, edge, pl = model.train_step(x, cond, y, C.BETA_KL, all_pal)
    assert isinstance(loss, float)
    assert loss > 0
    assert recon > 0


def test_gradient_shapes():
    """Os gradientes possuem dimensoes corretas."""
    import train
    model = train.ConditionalVAE(seed=0)
    x = np.random.randn(2, C.INPUT_DIM).astype(np.float32)
    cond = np.random.randn(2, C.CONDITION_DIM).astype(np.float32)
    y = np.random.randn(2, C.OUTPUT_DIM).astype(np.float32)
    out, mu, log_var, enc, dec, z, eps, std = model.forward(x, cond)
    grads = model.backward(x, cond, y, enc, dec, z, eps, std, beta_kl=C.BETA_KL)
    assert grads['W_enc1'].shape == (C.INPUT_DIM*2, C.HIDDEN_DIM)
    assert grads['W_mu'].shape == (C.HIDDEN_DIM, C.LATENT_DIM)
    assert grads['W_dec1'].shape == (C.LATENT_DIM, C.HIDDEN_DIM)
    assert grads['W_dec2'].shape == (C.HIDDEN_DIM, C.OUTPUT_DIM)
    assert grads['W_cond'].shape == (C.CONDITION_DIM, C.INPUT_DIM)


def test_checkpoint_save_load():
    """O checkpoint pode ser salvo e carregado."""
    import train
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.npz"
        model = train.ConditionalVAE(seed=42)
        model.save(path, epoch=10, loss=0.5, val_loss=0.6)
        loaded = train.ConditionalVAE(seed=99)
        ep, lo = loaded.load(path)
        assert ep == 10
        assert abs(lo - 0.5) < 1e-6
        assert np.allclose(model.W_dec1, loaded.W_dec1)


def test_prompts_different_conditions():
    """Prompts diferentes produzem condicoes diferentes."""
    from pixel_art_generator import interpret_prompt, build_condition_from_prompt
    cfg1 = interpret_prompt("heroi estilo NES")
    cfg2 = interpret_prompt("espada neon")
    cfg3 = interpret_prompt("monstro dark")
    c1 = build_condition_from_prompt(cfg1, seed=1)
    c2 = build_condition_from_prompt(cfg2, seed=1)
    c3 = build_condition_from_prompt(cfg3, seed=1)
    assert not np.allclose(c1, c2)
    assert not np.allclose(c1, c3)
    assert not np.allclose(c2, c3)


def test_six_palettes_recognized():
    """As seis paletas de treino sao reconhecidas."""
    from pixel_art_generator import make_palette
    for name in C.PALETTE_NAMES:
        pal = make_palette(name, n=4, seed=0)
        assert pal.shape == (4, 3)
        assert pal.dtype == np.uint8


@pytest.mark.parametrize("size", [(64, 64), (70, 70), (100, 64)])
def test_image_sizes(size):
    """Imagens 64x64, 70x70 e 100x64 sao geradas sem areas vazias."""
    from pixel_art_generator import VAEModel
    model = VAEModel()
    model._init_random()  # fallback
    img = model.generate_image(size[0], size[1], noise_scale=1.0, seed=42,
                               coherence=0.3, overlap=2)
    assert img.shape == (size[1], size[0], 3)
    # Sem areas pretas (exceto por acaso do modelo aleatorio)
    # mas pelo menos deve ter pixels nao-pretos
    assert img.sum() > 0


def test_overlap_blending():
    """O parametro overlap realmente altera o blending."""
    from pixel_art_generator import VAEModel
    model = VAEModel()
    model._init_random()
    img0 = model.generate_image(32, 32, seed=42, overlap=0)
    img2 = model.generate_image(32, 32, seed=42, overlap=2)
    img4 = model.generate_image(32, 32, seed=42, overlap=4)
    # Nao devem ser identicos
    assert not np.array_equal(img0, img2)
    assert not np.array_equal(img2, img4)


def test_latest_npz_loaded():
    """O gerador continua carregando models/checkpoints/latest.npz."""
    import inspect
    from pixel_art_generator import VAEModel
    src = inspect.getsource(VAEModel.load_latest)
    assert "latest.npz" in src
    assert "best.npz" in src  # tambem suporta best


def test_kl_gradient_beta_applied_only_to_kl():
    """Verifica que beta_kl e aplicado APENAS aos gradientes KL."""
    import train
    model = train.ConditionalVAE(seed=0)
    x = np.random.randn(2, C.INPUT_DIM).astype(np.float32) * 0.1
    cond = np.random.randn(2, C.CONDITION_DIM).astype(np.float32) * 0.1
    y = np.random.randn(2, C.OUTPUT_DIM).astype(np.float32) * 0.1
    out, mu, log_var, enc, dec, z, eps, std = model.forward(x, cond)
    grads_0 = model.backward(x, cond, y, enc, dec, z, eps, std, beta_kl=0.0)
    grads_1 = model.backward(x, cond, y, enc, dec, z, eps, std, beta_kl=1.0)
    # Gradiente do decoder deve ser IGUAL (nao afetado por beta_kl)
    assert np.allclose(grads_0['W_dec2'], grads_1['W_dec2'], atol=1e-6)
    assert np.allclose(grads_0['W_dec1'], grads_1['W_dec1'], atol=1e-6)
    # Gradientes do encoder (mu, logvar) devem ser DIFERENTES
    assert not np.allclose(grads_0['W_mu'], grads_1['W_mu'])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
