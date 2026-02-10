"""Tests for the full Transformer model."""

import numpy as np
import pytest
import torch
import torch.nn

from rawformer.exceptions import ForwardNotCalledError
from rawformer.transformer.transformer import Transformer


class TestTransformer:
    def test_forward_output_shape(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=64,
            max_len=100,
            rng=np.random.default_rng(0),
            dropout_rate=0.0,
        )
        src = np.array([[1, 5, 10, 3], [2, 7, 0, 4]], dtype=np.intp)
        tgt = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.intp)
        logits = model.forward(src, tgt)
        assert logits.shape == (2, 3, 50)

    def test_backward_runs_without_error(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=64,
            max_len=100,
            rng=np.random.default_rng(0),
            dropout_rate=0.0,
        )
        src = np.array([[1, 5, 10, 3]], dtype=np.intp)
        tgt = np.array([[1, 3, 5]], dtype=np.intp)
        logits = model.forward(src, tgt)
        grad = np.ones_like(logits)
        model.backward(grad)

    def test_backward_raises_without_forward(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
            dropout_rate=0.0,
        )
        with pytest.raises(ForwardNotCalledError):
            model.backward(np.ones((1, 3, 50)))

    def test_update_params_changes_output(self) -> None:
        model = Transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
            dropout_rate=0.0,
        )
        src = np.array([[1, 5, 10]], dtype=np.intp)
        tgt = np.array([[1, 3]], dtype=np.intp)
        logits_before = model.forward(src, tgt).copy()
        model.backward(np.ones_like(logits_before))
        model.update_params(learning_rate=0.01)
        logits_after = model.forward(src, tgt)
        assert not np.array_equal(logits_before, logits_after)

    def test_different_src_tgt_vocab_sizes(self) -> None:
        model = Transformer(
            src_vocab_size=30,
            tgt_vocab_size=40,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
            dropout_rate=0.0,
        )
        src = np.array([[1, 5, 10]], dtype=np.intp)
        tgt = np.array([[1, 3]], dtype=np.intp)
        logits = model.forward(src, tgt)
        assert logits.shape == (1, 2, 40)

    def test_forward_matches_pytorch(self) -> None:
        """End-to-end cross-verification against torch.nn.Transformer."""
        d_model, n_heads, d_ff = 16, 4, 32
        src_vocab, tgt_vocab = 20, 20
        n_enc, n_dec = 1, 1

        rng = np.random.default_rng(42)
        model = Transformer(
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_enc,
            n_decoder_layers=n_dec,
            d_ff=d_ff,
            max_len=50,
            rng=rng,
            dropout_rate=0.0,
        )

        # build matching PyTorch model
        torch_src_embed = torch.nn.Embedding(src_vocab, d_model, dtype=torch.float64)
        torch_tgt_embed = torch.nn.Embedding(tgt_vocab, d_model, dtype=torch.float64)
        torch_src_embed.weight.data = torch.from_numpy(model.src_embed.weight.copy())
        torch_tgt_embed.weight.data = torch.from_numpy(model.tgt_embed.weight.copy())

        torch_tf = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
            dtype=torch.float64,
        )

        # copy encoder block weights
        enc_block = model.encoder.blocks[0]
        t_enc = torch_tf.encoder.layers[0]

        # self-attention
        wq = enc_block.self_attn.w_q.weights.T
        wk = enc_block.self_attn.w_k.weights.T
        wv = enc_block.self_attn.w_v.weights.T
        t_enc.self_attn.in_proj_weight.data = torch.from_numpy(
            np.concatenate([wq, wk, wv], axis=0).copy()
        )
        bq = enc_block.self_attn.w_q.biases
        bk = enc_block.self_attn.w_k.biases
        bv = enc_block.self_attn.w_v.biases
        t_enc.self_attn.in_proj_bias.data = torch.from_numpy(np.concatenate([bq, bk, bv]).copy())
        t_enc.self_attn.out_proj.weight.data = torch.from_numpy(
            enc_block.self_attn.w_o.weights.T.copy()
        )
        t_enc.self_attn.out_proj.bias.data = torch.from_numpy(enc_block.self_attn.w_o.biases.copy())

        # encoder layer norms
        t_enc.norm1.weight.data = torch.from_numpy(enc_block.norm1.gamma.copy())
        t_enc.norm1.bias.data = torch.from_numpy(enc_block.norm1.beta.copy())
        t_enc.norm2.weight.data = torch.from_numpy(enc_block.norm2.gamma.copy())
        t_enc.norm2.bias.data = torch.from_numpy(enc_block.norm2.beta.copy())

        # encoder FFN
        t_enc.linear1.weight.data = torch.from_numpy(enc_block.ffn.linear1.weights.T.copy())
        t_enc.linear1.bias.data = torch.from_numpy(enc_block.ffn.linear1.biases.copy())
        t_enc.linear2.weight.data = torch.from_numpy(enc_block.ffn.linear2.weights.T.copy())
        t_enc.linear2.bias.data = torch.from_numpy(enc_block.ffn.linear2.biases.copy())

        # copy decoder block weights
        dec_block = model.decoder.blocks[0]
        t_dec = torch_tf.decoder.layers[0]

        # masked self-attention
        wq = dec_block.self_attn.w_q.weights.T
        wk = dec_block.self_attn.w_k.weights.T
        wv = dec_block.self_attn.w_v.weights.T
        t_dec.self_attn.in_proj_weight.data = torch.from_numpy(
            np.concatenate([wq, wk, wv], axis=0).copy()
        )
        bq = dec_block.self_attn.w_q.biases
        bk = dec_block.self_attn.w_k.biases
        bv = dec_block.self_attn.w_v.biases
        t_dec.self_attn.in_proj_bias.data = torch.from_numpy(np.concatenate([bq, bk, bv]).copy())
        t_dec.self_attn.out_proj.weight.data = torch.from_numpy(
            dec_block.self_attn.w_o.weights.T.copy()
        )
        t_dec.self_attn.out_proj.bias.data = torch.from_numpy(dec_block.self_attn.w_o.biases.copy())

        # cross-attention
        wq = dec_block.cross_attn.w_q.weights.T
        wk = dec_block.cross_attn.w_k.weights.T
        wv = dec_block.cross_attn.w_v.weights.T
        t_dec.multihead_attn.in_proj_weight.data = torch.from_numpy(
            np.concatenate([wq, wk, wv], axis=0).copy()
        )
        bq = dec_block.cross_attn.w_q.biases
        bk = dec_block.cross_attn.w_k.biases
        bv = dec_block.cross_attn.w_v.biases
        t_dec.multihead_attn.in_proj_bias.data = torch.from_numpy(
            np.concatenate([bq, bk, bv]).copy()
        )
        t_dec.multihead_attn.out_proj.weight.data = torch.from_numpy(
            dec_block.cross_attn.w_o.weights.T.copy()
        )
        t_dec.multihead_attn.out_proj.bias.data = torch.from_numpy(
            dec_block.cross_attn.w_o.biases.copy()
        )

        # decoder layer norms
        t_dec.norm1.weight.data = torch.from_numpy(dec_block.norm1.gamma.copy())
        t_dec.norm1.bias.data = torch.from_numpy(dec_block.norm1.beta.copy())
        t_dec.norm2.weight.data = torch.from_numpy(dec_block.norm2.gamma.copy())
        t_dec.norm2.bias.data = torch.from_numpy(dec_block.norm2.beta.copy())
        t_dec.norm3.weight.data = torch.from_numpy(dec_block.norm3.gamma.copy())
        t_dec.norm3.bias.data = torch.from_numpy(dec_block.norm3.beta.copy())

        # decoder FFN
        t_dec.linear1.weight.data = torch.from_numpy(dec_block.ffn.linear1.weights.T.copy())
        t_dec.linear1.bias.data = torch.from_numpy(dec_block.ffn.linear1.biases.copy())
        t_dec.linear2.weight.data = torch.from_numpy(dec_block.ffn.linear2.weights.T.copy())
        t_dec.linear2.bias.data = torch.from_numpy(dec_block.ffn.linear2.biases.copy())

        # output projection
        torch_output_proj = torch.nn.Linear(d_model, tgt_vocab, dtype=torch.float64)
        torch_output_proj.weight.data = torch.from_numpy(model.output_proj.weights.T.copy())
        torch_output_proj.bias.data = torch.from_numpy(model.output_proj.biases.copy())

        # forward through rawformer
        src = np.array([[1, 5, 10]], dtype=np.intp)
        tgt = np.array([[1, 3]], dtype=np.intp)
        logits_np = model.forward(src, tgt)

        # forward through PyTorch (manual embedding + PE + transformer + proj)
        scale = np.sqrt(np.float64(d_model))
        pe = model.pos_enc.encoding_table

        src_t = torch.from_numpy(src).long()
        tgt_t = torch.from_numpy(tgt).long()

        src_emb_t = torch_src_embed(src_t) * scale + torch.from_numpy(pe[: src.shape[1]])
        tgt_emb_t = torch_tgt_embed(tgt_t) * scale + torch.from_numpy(pe[: tgt.shape[1]])

        tgt_mask_t = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt_mask_t = tgt_mask_t.to(torch.float64)

        torch_tf.eval()
        with torch.no_grad():
            tf_out = torch_tf(src_emb_t, tgt_emb_t, tgt_mask=tgt_mask_t)
            logits_torch = torch_output_proj(tf_out).numpy()

        np.testing.assert_allclose(logits_np, logits_torch, atol=1e-5)

    def test_train_eval_toggles_dropout(self) -> None:
        """Verify train() and eval() toggle all dropout layers."""
        model = Transformer(
            src_vocab_size=20,
            tgt_vocab_size=20,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
            dropout_rate=0.1,
        )
        assert all(d.training for d in model.dropouts)
        model.eval()
        assert all(not d.training for d in model.dropouts)
        model.train()
        assert all(d.training for d in model.dropouts)

    def test_src_padding_mask_affects_output(self) -> None:
        """Verify that src_padding_mask changes the output vs unmasked."""
        model = Transformer(
            src_vocab_size=20,
            tgt_vocab_size=20,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
            dropout_rate=0.0,
        )
        src = np.array([[1, 5, 10, 3]], dtype=np.intp)
        tgt = np.array([[1, 3]], dtype=np.intp)

        logits_no_mask = model.forward(src, tgt)

        # mask out last two source positions via additive -inf mask
        # shape (1, 1, 1, 4) broadcastable to (batch, n_heads, seq_q, seq_k)
        src_mask = np.zeros((1, 1, 1, 4))
        src_mask[0, 0, 0, 2] = -np.inf
        src_mask[0, 0, 0, 3] = -np.inf

        logits_masked = model.forward(src, tgt, src_padding_mask=src_mask)

        assert not np.allclose(logits_no_mask, logits_masked)

    def test_tgt_padding_mask_affects_output(self) -> None:
        """Verify that tgt_padding_mask changes cross-attention output."""
        model = Transformer(
            src_vocab_size=20,
            tgt_vocab_size=20,
            d_model=16,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=32,
            max_len=50,
            rng=np.random.default_rng(0),
            dropout_rate=0.0,
        )
        src = np.array([[1, 5, 10, 3]], dtype=np.intp)
        tgt = np.array([[1, 3, 5]], dtype=np.intp)

        logits_no_mask = model.forward(src, tgt)

        # mask out last source position in cross-attention
        # shape (1, 1, 1, 4) broadcastable to (batch, n_heads, seq_q, seq_k)
        tgt_mask = np.zeros((1, 1, 1, 4))
        tgt_mask[0, 0, 0, 3] = -np.inf

        logits_masked = model.forward(src, tgt, tgt_padding_mask=tgt_mask)

        assert not np.allclose(logits_no_mask, logits_masked)
