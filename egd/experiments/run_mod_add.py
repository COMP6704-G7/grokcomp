#!/usr/bin/env python3
"""Modular Addition runner for COMP6704 EGD experimental plan.

Supports Exp1 (effectiveness / generalization across p) and Exp3 (efficiency).
Methods: vanilla (SGD), col_norm, egd (full SVD), egd_rsvd (randomized SVD).
"""
from __future__ import annotations

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
def make_disjoint_mod_datasets(key, p, train_fraction=0.5, shuffle_seed=42):
    total_examples = p * p
    all_a = jnp.repeat(jnp.arange(p), p)
    all_b = jnp.tile(jnp.arange(p), p)
    y = (all_a + all_b) % p

    perm_key = random.PRNGKey(shuffle_seed) if shuffle_seed is not None else key
    perm = random.permutation(perm_key, total_examples)

    train_size = int(total_examples * train_fraction)
    train_idx, test_idx = perm[:train_size], perm[train_size:]

    def encode(a, b, y):
        N = a.shape[0]
        X = jnp.zeros((N, 2 * p), dtype=jnp.float32)
        X = X.at[jnp.arange(N), a].set(1.0)
        X = X.at[jnp.arange(N), p + b].set(1.0)
        return X, jax.nn.one_hot(y, p)

    a_tr, b_tr, y_tr = all_a[train_idx], all_b[train_idx], y[train_idx]
    a_te, b_te, y_te = all_a[test_idx], all_b[test_idx], y[test_idx]
    return encode(a_tr, b_tr, y_tr), encode(a_te, b_te, y_te)


def init_params(key, n, hidden_size, out_dim):
    k1, _, k3 = random.split(key, 3)
    W1 = random.normal(k1, (n, hidden_size)) * sqrt(2 / n)
    b1 = jnp.zeros((hidden_size,))
    W2 = random.normal(k3, (hidden_size, out_dim)) * sqrt(2 / hidden_size)
    return (W1, b1, W2)


def forward(params, X):
    W1, b1, W2 = params
    return jnp.maximum(0, X @ W1 + b1) @ W2


def make_loss(wd):
    def cross_entropy_loss(params, X, y):
        logits = forward(params, X)
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
        loss = -jnp.mean(jnp.sum(y * log_probs, axis=1))
        W1, _, W2 = params
        return loss + 0.5 * wd * (jnp.sum(W1**2) + jnp.sum(W2**2))
    return cross_entropy_loss


def make_accuracy():
    @jax.jit
    def accuracy(params, X, y):
        preds = jnp.argmax(forward(params, X), axis=1)
        labels = jnp.argmax(y, axis=1)
        return jnp.mean(preds == labels)
    return accuracy


def randomized_svd_jax(G, rank, n_iter=2, key=jax.random.PRNGKey(0)):
    _, n = G.shape
    Q = jax.random.normal(key, (n, rank), dtype=G.dtype)
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
        col_scales = jnp.maximum(jnp.linalg.norm(gW1, axis=0), 0.01)
        dW1 = gW1 / col_scales[None, :]
        W1, b1, W2 = params
        return (W1 - lr * dW1, b1 - lr * gb1, W2 - lr * gW2)

    @jax.jit
    def egd_update(params, X, y):
        grads = list(jax.grad(loss_fn)(params, X, y))
        G1 = grads[0]
        U, s, _ = jnp.linalg.svd(G1, full_matrices=False)
        s = jnp.maximum(s, 1e-6)
        aux = (U * (1.0 / s)[None, :]) @ U.T
        grads[0] = aux @ G1
        return tuple(p - lr * g for p, g in zip(params, grads))

    @jax.jit
    def egd_rsvd_update(params, X, y):
        grads = list(jax.grad(loss_fn)(params, X, y))
        G1 = grads[0]
        U, s = randomized_svd_jax(G1, rank)
        s = jnp.maximum(s, 1e-6)
        aux = (U * (1.0 / s)[None, :]) @ U.T
        grads[0] = aux @ G1
        return tuple(p - lr * g for p, g in zip(params, grads))

    return {
        "vanilla": vanilla_update,
        "col_norm": col_norm_update,
        "egd": egd_update,
        "egd_rsvd": egd_rsvd_update,
    }


def first_stable_threshold(values, epochs, thr=0.95, patience=3):
    """First epoch where `patience` consecutive evals are all >= thr."""
    if patience <= 1:
        for e, v in zip(epochs, values):
            if v >= thr:
                return int(e)
        return None
    run = 0
    for e, v in zip(epochs, values):
        if v >= thr:
            run += 1
            if run >= patience:
                return int(e)
        else:
            run = 0
    return None


def peak_gpu_mem_mb():
    try:
        devices = jax.local_devices()
        # JAX does not expose peak mem portably; use nvidia-smi for device 0 of visible set
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            text=True,
        )
        vals = [float(x) for x in out.strip().splitlines() if x.strip()]
        return max(vals) if vals else None
    except Exception:
        return None


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"modadd_p{args.p}_{args.method}"
        + (f"_r{args.rank}" if args.method == "egd_rsvd" else "")
        + f"_seed{args.seed}"
    )
    result_path = out_dir / f"{tag}.json"
    if result_path.exists() and not args.overwrite:
        print(f"[skip] exists: {result_path}")
        return json.loads(result_path.read_text())

    (X_tr, y_tr), (X_te, y_te) = make_disjoint_mod_datasets(
        random.PRNGKey(0), args.p, train_fraction=args.train_fraction, shuffle_seed=42
    )
    n = X_tr.shape[1]
    loss_fn = make_loss(args.wd)
    accuracy = make_accuracy()
    updaters = make_updaters(loss_fn, args.lr, args.rank)
    if args.method not in updaters:
        raise ValueError(f"Unknown method {args.method}. Choose from {list(updaters)}")
    update_fn = updaters[args.method]

    params = init_params(random.PRNGKey(args.seed), n, args.hidden_dim, args.p)
    num_batches = max(1, X_tr.shape[0] // args.batch_size)

    # Warmup / compile
    xb0 = X_tr[: args.batch_size]
    yb0 = y_tr[: args.batch_size]
    params = update_fn(params, xb0, yb0)
    _ = accuracy(params, X_tr[:64], y_tr[:64]).block_until_ready()
    jax.effects_barrier() if hasattr(jax, "effects_barrier") else None

    train_accs, test_accs, eval_epochs, epoch_times = [], [], [], []
    t_train_95 = t_test_95 = None
    wall_to_test_95 = None
    mem_samples = []

    # Reset after warmup so curves start clean — re-init params
    params = init_params(random.PRNGKey(args.seed), n, args.hidden_dim, args.p)
    # one more compile-only step without counting
    params = update_fn(params, xb0, yb0)
    params = init_params(random.PRNGKey(args.seed), n, args.hidden_dim, args.p)

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        perm = random.permutation(random.PRNGKey(args.seed * 100000 + epoch), X_tr.shape[0])
        X_shuf, y_shuf = X_tr[perm], y_tr[perm]

        for i in range(num_batches):
            s = i * args.batch_size
            e = min(s + args.batch_size, X_tr.shape[0])
            # After test 95%, fall back to cheap vanilla (paper repo pattern) unless disabled
            if args.keep_method or t_test_95 is None:
                params = update_fn(params, X_shuf[s:e], y_shuf[s:e])
            else:
                params = updaters["vanilla"](params, X_shuf[s:e], y_shuf[s:e])

        # sync for accurate timing
        params[0].block_until_ready()
        epoch_dt = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_dt)

        # eval schedule denser early
        if epoch < 1000:
            eval_freq = args.eval_freq_early
        elif epoch < 5000:
            eval_freq = args.eval_freq_mid
        else:
            eval_freq = args.eval_freq_late

        do_eval = (epoch % eval_freq == 0) or (epoch == args.epochs - 1) or (
            t_test_95 is None and epoch_times and epoch % max(1, eval_freq // 2) == 0
        )
        # Always eval every epoch until both hit 95% if early_dense
        if args.dense_until_95 and t_test_95 is None:
            do_eval = True

        if do_eval:
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

            if epoch % max(1, eval_freq) == 0 or te >= args.threshold:
                print(
                    f"{tag}: epoch={epoch:05d} train={ta:.4f} test={te:.4f} "
                    f"t/ep={epoch_dt*1000:.1f}ms",
                    flush=True,
                )

        if epoch % 50 == 0:
            m = peak_gpu_mem_mb()
            if m is not None:
                mem_samples.append(m)

        if args.stop_at_95 and t_test_95 is not None:
            # finish current eval then stop
            break

    total_wall = time.perf_counter() - t0
    delay = None
    if t_train_95 is not None and t_test_95 is not None:
        delay = t_test_95 - t_train_95

    result = {
        "task": "modular_addition",
        "p": args.p,
        "method": args.method,
        "rank": args.rank if args.method == "egd_rsvd" else None,
        "seed": args.seed,
        "hyperparams": {
            "lr": args.lr,
            "wd": args.wd,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "train_fraction": args.train_fraction,
            "epochs_budget": args.epochs,
            "threshold": args.threshold,
            "patience": args.patience,
        },
        "metrics": {
            "epochs_to_train_95": t_train_95,
            "epochs_to_test_95": t_test_95,
            "grokking_delay": delay,
            "reached_test_95": t_test_95 is not None,
            "time_per_epoch_mean_s": float(np.mean(epoch_times)) if epoch_times else None,
            "time_per_epoch_std_s": float(np.std(epoch_times)) if epoch_times else None,
            "wall_clock_to_test_95_s": wall_to_test_95,
            "total_wall_clock_s": total_wall,
            "epochs_run": len(epoch_times),
            "peak_gpu_mem_mb": max(mem_samples) if mem_samples else None,
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
    print(
        f"  test@95={t_test_95} delay={delay} "
        f"wall95={wall_to_test_95} t/ep={result['metrics']['time_per_epoch_mean_s']:.4f}s"
    )
    return result


def build_parser():
    p = argparse.ArgumentParser(description="EGD Modular Addition experiment")
    p.add_argument("--p", type=int, default=97, choices=[79, 97, 127])
    p.add_argument(
        "--method",
        type=str,
        default="egd",
        choices=["vanilla", "col_norm", "egd", "egd_rsvd"],
    )
    p.add_argument("--rank", type=int, default=64, help="RSVD rank")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15000)
    p.add_argument("--lr", type=float, default=7e-1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--train_fraction", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--patience", type=int, default=3, help="Consecutive evals >= thr")
    p.add_argument("--eval_freq_early", type=int, default=50)
    p.add_argument("--eval_freq_mid", type=int, default=100)
    p.add_argument("--eval_freq_late", type=int, default=200)
    p.add_argument("--dense_until_95", action="store_true", default=True)
    p.add_argument("--no_dense_until_95", action="store_false", dest="dense_until_95")
    p.add_argument("--stop_at_95", action="store_true", help="Stop once test hits 95%")
    p.add_argument("--keep_method", action="store_true", help="Do not fall back to SGD after 95%")
    p.add_argument("--out_dir", type=str, default="experiments/results")
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
