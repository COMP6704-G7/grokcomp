#!/usr/bin/env python3
"""Sparse Parity runner for COMP6704 EGD experimental plan (Exp2)."""
from __future__ import annotations

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.utils import check_random_state

SETTINGS = {
    2: dict(n=400, hidden_dim=50, lr=1e-2, wd=1e-3, N=2000, batch_size=32, epochs=10100),
    3: dict(n=100, hidden_dim=100, lr=0.0416239, wd=1e-2, N=2000, batch_size=32, epochs=2000),
    4: dict(n=50, hidden_dim=100, lr=0.0230841, wd=1e-2, N=2000, batch_size=32, epochs=6000),
}


def make_parity_dataset(n, k, N, random_state=None):
    rng = check_random_state(random_state)
    X = rng.choice([-1, 1], size=(N, n)).astype(np.float32)
    y = X[:, :k].prod(axis=1).astype(np.float32)
    return jnp.asarray(X), jnp.asarray(y)


def init_params(n, hidden_size, random_state=None):
    rng = check_random_state(random_state)
    W1 = rng.randn(n, hidden_size) * sqrt(2 / n)
    b1 = rng.randn(hidden_size) * sqrt(2 / n)
    W2 = rng.randn(hidden_size) * sqrt(2 / hidden_size)
    return (
        jnp.asarray(W1, dtype=jnp.float32),
        jnp.asarray(b1, dtype=jnp.float32),
        jnp.asarray(W2, dtype=jnp.float32),
    )


def forward(params, X):
    W1, b1, W2 = params
    return jnp.maximum(0, X @ W1 + b1) @ W2


def make_loss(wd):
    def loss_fn(params, X, y):
        logits = forward(params, jnp.atleast_2d(X))
        y = jnp.atleast_1d(y)
        loss = jnp.mean((logits - y) ** 2)
        W1, _, W2 = params
        return loss + 0.5 * wd * (jnp.sum(W1**2) + jnp.sum(W2**2))
    return loss_fn


@jax.jit
def accuracy(params, X, y):
    return jnp.mean((y * forward(params, X)) > 0)


def randomized_svd_jax(G, rank, n_iter=2, key=jax.random.PRNGKey(0)):
    _, ncol = G.shape
    Q = jax.random.normal(key, (ncol, rank), dtype=G.dtype)
    for _ in range(n_iter):
        Q = jnp.linalg.qr(G @ Q, mode="reduced")[0]
        Q = jnp.linalg.qr(G.T @ Q, mode="reduced")[0]
    B = G @ Q
    U_hat, S, _ = jnp.linalg.svd(B, full_matrices=False)
    return U_hat, S


def make_updaters(loss_fn, lr, rank):
    @jax.jit
    def vanilla_update(params, X, y):
        grads = jax.grad(loss_fn)(params, X, y)
        return tuple(p - lr * g for p, g in zip(params, grads))

    @jax.jit
    def col_norm_update(params, X, y):
        gW1, gb1, gW2 = jax.grad(loss_fn)(params, X, y)
        scales = jnp.maximum(jnp.linalg.norm(gW1, axis=0), 0.01)
        W1, b1, W2 = params
        return (W1 - lr * (gW1 / scales[None, :]), b1 - lr * gb1, W2 - lr * gW2)

    @jax.jit
    def egd_update(params, X, y):
        # Equalize significant singular values; ignore near-null directions
        # (absolute 1e-6 floor over-amplifies noise on square W1 and destabilizes parity).
        grads = list(jax.grad(loss_fn)(params, X, y))
        G1 = grads[0]
        U, s, _ = jnp.linalg.svd(G1, full_matrices=False)
        thr = 1e-5 * (s[0] + 1e-12)
        s_inv = jnp.where(s > thr, 1.0 / s, 0.0)
        grads[0] = ((U * s_inv[None, :]) @ U.T) @ G1
        return tuple(p - lr * g for p, g in zip(params, grads))

    @jax.jit
    def egd_rsvd_update(params, X, y):
        grads = list(jax.grad(loss_fn)(params, X, y))
        G1 = grads[0]
        U, s = randomized_svd_jax(G1, rank)
        thr = 1e-5 * (s[0] + 1e-12)
        s_inv = jnp.where(s > thr, 1.0 / s, 0.0)
        grads[0] = ((U * s_inv[None, :]) @ U.T) @ G1
        return tuple(p - lr * g for p, g in zip(params, grads))

    return {
        "vanilla": vanilla_update,
        "col_norm": col_norm_update,
        "egd": egd_update,
        "egd_rsvd": egd_rsvd_update,
    }


def first_stable_threshold(values, epochs, thr=0.95, patience=3):
    run = 0
    for e, v in zip(epochs, values):
        if v >= thr:
            run += 1
            if run >= patience:
                return int(e)
        else:
            run = 0
    return None


def train(args):
    cfg = SETTINGS[args.k].copy()
    for key in ("n", "hidden_dim", "lr", "wd", "N", "batch_size", "epochs"):
        cli = getattr(args, key, None)
        if cli is not None and key in {"lr", "wd", "epochs", "batch_size", "hidden_dim", "N"}:
            # only override if user passed non-default sentinel via explicit flags
            pass
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"parity_k{args.k}_{args.method}"
        + (f"_r{args.rank}" if args.method == "egd_rsvd" else "")
        + f"_seed{args.seed}"
    )
    result_path = out_dir / f"{tag}.json"
    if result_path.exists() and not args.overwrite:
        print(f"[skip] exists: {result_path}")
        return json.loads(result_path.read_text())

    rng = check_random_state(args.seed)
    X_tr, y_tr = make_parity_dataset(cfg["n"], args.k, cfg["N"], random_state=rng)
    X_te, y_te = make_parity_dataset(cfg["n"], args.k, 10000, random_state=rng)

    loss_fn = make_loss(cfg["wd"])
    updaters = make_updaters(loss_fn, cfg["lr"], args.rank)
    update_fn = updaters[args.method]
    params = init_params(cfg["n"], cfg["hidden_dim"], random_state=args.seed)
    num_batches = max(1, X_tr.shape[0] // cfg["batch_size"])

    # warmup
    params = update_fn(params, X_tr[: cfg["batch_size"]], y_tr[: cfg["batch_size"]])
    params[0].block_until_ready()
    params = init_params(cfg["n"], cfg["hidden_dim"], random_state=args.seed)

    train_accs, test_accs, eval_epochs, epoch_times = [], [], [], []
    t_train_95 = t_test_95 = wall_to_test_95 = None
    t0 = time.perf_counter()

    for epoch in range(cfg["epochs"]):
        ep0 = time.perf_counter()
        perm = rng.permutation(X_tr.shape[0])
        X_shuf, y_shuf = X_tr[perm], y_tr[perm]
        for i in range(num_batches):
            s = i * cfg["batch_size"]
            e = s + cfg["batch_size"]
            if args.keep_method or t_test_95 is None:
                params = update_fn(params, X_shuf[s:e], y_shuf[s:e])
            else:
                params = updaters["vanilla"](params, X_shuf[s:e], y_shuf[s:e])
        params[0].block_until_ready()
        epoch_times.append(time.perf_counter() - ep0)

        if epoch % args.eval_every == 0 or epoch == cfg["epochs"] - 1 or (
            args.dense_until_95 and t_test_95 is None
        ):
            ta = float(accuracy(params, X_tr, y_tr))
            te = float(accuracy(params, X_te, y_te))
            train_accs.append(ta)
            test_accs.append(te)
            eval_epochs.append(epoch)
            if t_train_95 is None:
                t_train_95 = first_stable_threshold(
                    train_accs, eval_epochs, args.threshold, args.patience
                )
            if t_test_95 is None:
                cand = first_stable_threshold(
                    test_accs, eval_epochs, args.threshold, args.patience
                )
                if cand is not None:
                    t_test_95 = cand
                    wall_to_test_95 = time.perf_counter() - t0
            if epoch % max(1, args.eval_every) == 0:
                print(
                    f"{tag}: epoch={epoch:05d} train={ta:.4f} test={te:.4f}",
                    flush=True,
                )

        if args.stop_at_95 and t_test_95 is not None:
            break

    total_wall = time.perf_counter() - t0
    delay = (
        (t_test_95 - t_train_95)
        if (t_train_95 is not None and t_test_95 is not None)
        else None
    )
    result = {
        "task": "sparse_parity",
        "k": args.k,
        "n": cfg["n"],
        "method": args.method,
        "rank": args.rank if args.method == "egd_rsvd" else None,
        "seed": args.seed,
        "hyperparams": cfg,
        "metrics": {
            "epochs_to_train_95": t_train_95,
            "epochs_to_test_95": t_test_95,
            "grokking_delay": delay,
            "reached_test_95": t_test_95 is not None,
            "time_per_epoch_mean_s": float(np.mean(epoch_times)) if epoch_times else None,
            "wall_clock_to_test_95_s": wall_to_test_95,
            "total_wall_clock_s": total_wall,
            "epochs_run": len(epoch_times),
            "final_train_acc": train_accs[-1] if train_accs else None,
            "final_test_acc": test_accs[-1] if test_accs else None,
        },
        "curves": {
            "eval_epochs": eval_epochs,
            "train_acc": train_accs,
            "test_acc": test_accs,
            "epoch_times_s": epoch_times,
        },
        "device": str(jax.devices()[0]),
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {result_path}")
    return result


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--method", choices=["vanilla", "col_norm", "egd", "egd_rsvd"], default="egd")
    p.add_argument("--rank", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--dense_until_95", action="store_true", default=True)
    p.add_argument("--stop_at_95", action="store_true")
    p.add_argument("--keep_method", action="store_true")
    p.add_argument("--out_dir", type=str, default="experiments/results")
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
