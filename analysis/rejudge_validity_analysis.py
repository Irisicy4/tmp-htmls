"""Self-Evolve Bench judge reliability — canonical analysis.

Consumes the outputs of ``harness/rejudge_n_times.py`` and produces a 10x10
(or k x k) pairwise TVD-MI matrix on the binary verdict, plus a finest-bin
matrix on ``overall_score``, plus Cronbach's alpha and pairwise Cohen's
kappa for cross-reference. The deliverable is ``report.md`` + three figures.

Interpretation: TVD-MI(X;Y) = (1/2) * sum |P(X,Y) - P(X)P(Y)| equals the
total variation distance between the joint and the product of marginals.
By the variational characterisation of TV, (1/2) * TVD-MI is the advantage
over chance of an optimal binary classifier distinguishing same-task pairs
from independently-paired responses. So TVD-MI = 0.4 implies optimal
evaluator accuracy = 0.5 + 0.2 = 0.7.

Usage:
    # Real data produced by `harness/rejudge_n_times.py`:
    python analysis/rejudge_validity_analysis.py \\
        --rejudge-dir results/judge_variance \\
        --source cocoa-deprivacy-100 \\
        --out-dir analysis/output

    # Smoke test (no API calls; synthesises a small runs.jsonl):
    python analysis/rejudge_validity_analysis.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rejudge-dir", type=Path, default=Path("results/judge_variance"),
                   help="Directory containing <source>_runs.jsonl (from rejudge_n_times.py).")
    p.add_argument("--source", default="cocoa-deprivacy-100",
                   help="Source tag used as the filename prefix for runs.jsonl.")
    p.add_argument("--out-dir", type=Path, default=Path("analysis/output"),
                   help="Where to write figures, matrices, and report.md.")
    p.add_argument("--smoke", action="store_true",
                   help="Generate a tiny synthetic runs.jsonl and exit after analysis. "
                        "Useful for verifying the pipeline without running judges.")
    p.add_argument("--smoke-n-tasks", type=int, default=30)
    p.add_argument("--smoke-n-calls", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ── Smoke fixture ────────────────────────────────────────────────────────────

def write_smoke_fixture(rejudge_dir: Path, source: str, n_tasks: int,
                        n_calls: int, seed: int) -> Path:
    """Synthesize a small ``<source>_runs.jsonl`` with the schema produced by
    ``harness/rejudge_n_times.py``. Latent task quality drives both verdict
    and overall_score with judge-specific noise, so the resulting matrix has
    real structure (welfare > 0) — enough to exercise every code path."""
    rejudge_dir.mkdir(parents=True, exist_ok=True)
    runs_path = rejudge_dir / f"{source}_runs.jsonl"
    rng = np.random.default_rng(seed)
    judges = ["gpt-4o", "claude-sonnet-4-20250514"]
    dim_names = ["constraint_satisfaction", "result_specificity",
                 "comparison_quality", "task_completeness"]
    weights = {d: 0.25 for d in dim_names}
    threshold = 3.0
    with open(runs_path, "w") as f:
        for i in range(n_tasks):
            # latent quality in [1, 5]
            q = rng.uniform(1.0, 5.0)
            task = f"task-{i+1:03d}-synth"
            for jm in judges:
                # judge-specific bias
                bias = -0.1 if jm == "gpt-4o" else 0.1
                for c in range(1, n_calls + 1):
                    dims = {d: int(np.clip(round(q + bias + rng.normal(0, 0.6)),
                                            1, 5)) for d in dim_names}
                    overall = sum(weights[d] * dims[d] for d in dim_names)
                    passed = overall >= threshold
                    rec = {
                        "source": source,
                        "task": task,
                        "judge_model": jm,
                        "call_index": c,
                        "raw_response": f"<Answer>{json.dumps(dims)}</Answer>",
                        "dimension_scores": dims,
                        "dimension_weights": weights,
                        "overall_score": round(overall, 4),
                        "overall_source": "recomputed",
                        "passed": bool(passed),
                        "pass_threshold": threshold,
                        "input_hash": f"smoke-{i}",
                        "rubric_flavor": "A",
                    }
                    f.write(json.dumps(rec) + "\n")
    return runs_path


# ── Load response matrices ───────────────────────────────────────────────────

def load_runs(runs_path: Path):
    """Return (tasks, annotators, R_binary, S_overall) where annotators is the
    sorted list of (judge_model, call_index) tuples discovered in the file."""
    verdicts: dict = {}
    overall: dict = {}
    seen_ann: set = set()
    seen_tasks: set = set()
    with open(runs_path) as f:
        for line in f:
            r = json.loads(line)
            ann = (r["judge_model"], r["call_index"])
            seen_ann.add(ann)
            seen_tasks.add(r["task"])
            verdicts[(r["task"], *ann)] = r.get("passed")
            overall[(r["task"], *ann)] = r.get("overall_score")
    tasks = sorted(seen_tasks)
    annotators = sorted(seen_ann)  # (judge_model, call_index) tuples
    N, K = len(tasks), len(annotators)
    R = np.zeros((N, K), dtype=int)
    S = np.full((N, K), np.nan, dtype=float)
    for i, t in enumerate(tasks):
        for j, (jm, c) in enumerate(annotators):
            v = verdicts.get((t, jm, c))
            R[i, j] = 1 if v else 0
            s = overall.get((t, jm, c))
            if s is not None:
                S[i, j] = float(s)
    return tasks, annotators, R, S


def annotator_labels(annotators):
    """Short labels for figures: first letter of model + call_index. If
    multiple models share an initial we append a digit."""
    initials = defaultdict(int)
    labels = []
    for jm, c in annotators:
        letter = jm[0].upper()
        initials[letter] += 1
    # Use simple G/C/etc. mapping
    short = []
    by_model = {jm: i for i, jm in enumerate(sorted({a[0] for a in annotators}))}
    letter_for = {}
    for idx, jm in enumerate(sorted({a[0] for a in annotators})):
        letter_for[jm] = chr(ord("A") + idx) if jm[0].upper() in "ABCDE" else jm[0].upper()
    # Simpler: give each model a single uppercase letter in order of appearance.
    models_in_order = []
    for jm, _ in annotators:
        if jm not in models_in_order:
            models_in_order.append(jm)
    letter_for = {jm: chr(ord("A") + i) for i, jm in enumerate(models_in_order)}
    # Override A/B with G/C if it's the canonical GPT/Claude pair
    if set(models_in_order) >= {"gpt-4o", "claude-sonnet-4-20250514"}:
        letter_for = {"gpt-4o": "G", "claude-sonnet-4-20250514": "C"}
        for m in models_in_order:
            if m not in letter_for:
                letter_for[m] = m[0].upper()
    return [f"{letter_for[jm]}{c}" for (jm, c) in annotators], letter_for, models_in_order

# ── Statistics ───────────────────────────────────────────────────────────────

def tvd_mi(x, y):
    px = np.array([np.mean(x == v) for v in [0, 1]])
    py = np.array([np.mean(y == v) for v in [0, 1]])
    pxy = np.zeros((2, 2))
    for v1 in [0, 1]:
        for v2 in [0, 1]:
            pxy[v1, v2] = np.mean((x == v1) & (y == v2))
    return 0.5 * np.abs(pxy - np.outer(px, py)).sum()

def tvd_mi_categorical(x, y, alphabet):
    A = len(alphabet)
    idx = {a: i for i, a in enumerate(alphabet)}
    n = len(x)
    px = np.zeros(A); py = np.zeros(A)
    pxy = np.zeros((A, A))
    for v in x: px[idx[v]] += 1
    for v in y: py[idx[v]] += 1
    for a, b in zip(x, y): pxy[idx[a], idx[b]] += 1
    px /= n; py /= n; pxy /= n
    return 0.5 * np.abs(pxy - np.outer(px, py)).sum()

def offdiag_mean(B):
    n = B.shape[0]
    return (B.sum() - np.diag(B).sum()) / (n * n - n)

def cronbach_alpha(X):
    Kc = X.shape[1]
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    return (Kc / (Kc - 1)) * (1 - item_vars.sum() / total_var)

def cohens_kappa(x, y):
    po = np.mean(x == y)
    px1, py1 = x.mean(), y.mean()
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

# ── Block summaries (model-aware) ────────────────────────────────────────────

def block_summaries(M, annotators):
    """For each unique model, mean of the off-diagonal within-block of M
    (TVD-MI between calls of the same model). For each unordered pair of
    distinct models, mean of the corresponding between-block.

    Returns: dict keyed by ('within', m) or ('between', m_a, m_b),
             model order, and a per-model index list."""
    models = []
    for jm, _ in annotators:
        if jm not in models:
            models.append(jm)
    idx_by_model = {m: [j for j, (jm, _) in enumerate(annotators) if jm == m]
                    for m in models}
    out = {}
    for ma in models:
        ia = idx_by_model[ma]
        if len(ia) >= 2:
            sub = M[np.ix_(ia, ia)]
            out[("within", ma)] = (sub.sum() - np.diag(sub).sum()) / (len(ia) ** 2 - len(ia))
    for i, ma in enumerate(models):
        for mb in models[i+1:]:
            ia = idx_by_model[ma]
            ib = idx_by_model[mb]
            out[("between", ma, mb)] = M[np.ix_(ia, ib)].mean()
    return out, models, idx_by_model

# ── (figures/report/main defined below; see make_figures, write_report, main) ─

# ── Figures ──────────────────────────────────────────────────────────────────

def make_figures(M, hist, per_ann_reliab, welfare, short_labels,
                 idx_by_model, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = M.shape[0]

    # Figure 1: pairwise TVD-MI heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, vmin=0, vmax=max(0.5, float(M.max())), cmap="viridis", aspect="equal")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(short_labels); ax.set_yticklabels(short_labels)
    ax.set_title("Pairwise evaluation welfare (TVD-MI)")
    # Draw block dividers between models in matrix order
    boundary = 0
    for m, idxs in idx_by_model.items():
        boundary += len(idxs)
        if boundary < K:
            ax.axhline(boundary - 0.5, color="white", linewidth=2)
            ax.axvline(boundary - 0.5, color="white", linewidth=2)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < float(M.max()) * 0.5 else "black",
                    fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Evaluation welfare (TVD-MI)")
    plt.tight_layout()
    fig.savefig(out_dir / "figure_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Figure 2: per-annotator reliability
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = ["#4477AA", "#CC6677", "#117733", "#DDCC77", "#882255", "#88CCEE"]
    col_model = [0] * K
    for m_idx, (m, idxs) in enumerate(idx_by_model.items()):
        for j in idxs:
            col_model[j] = m_idx
    colours = [palette[col_model[j] % len(palette)] for j in range(K)]
    bars = ax.bar(short_labels, per_ann_reliab, color=colours, edgecolor="black")
    ax.axhline(welfare, color="gray", linestyle="--", linewidth=1,
               label=f"Mean welfare = {welfare:.3f}")
    ax.set_ylabel("Mean evaluation welfare (TVD-MI) vs other annotators")
    ax.set_xlabel("Annotator")
    ax.set_title("Per-annotator reliability")
    ax.set_ylim(0, max(float(per_ann_reliab.max()) * 1.15, 0.05))
    ax.legend(loc="lower right")
    for b, v in zip(bars, per_ann_reliab):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "figure_reliability.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Figure 3: pass-vote distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    n_max = len(hist) - 1
    edge = max(1, n_max // 5)
    bar_colors = []
    for k in range(n_max + 1):
        if k == 0 or k == n_max:
            bar_colors.append("#4477AA")
        elif k <= edge or k >= n_max - edge:
            bar_colors.append("#88CCEE")
        else:
            bar_colors.append("#CC6677")
    bars = ax.bar(range(n_max + 1), hist, color=bar_colors, edgecolor="black")
    ax.set_xlabel(f"Number of pass votes (out of {n_max} judge calls)")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Pass-vote distribution")
    ax.set_xticks(range(n_max + 1))
    for b, v in zip(bars, hist):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, str(int(v)),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(float(hist.max()) * 1.12, 1))
    plt.tight_layout()
    fig.savefig(out_dir / "figure_passdist.png", dpi=140, bbox_inches="tight")
    plt.close()



# ── Report ───────────────────────────────────────────────────────────────────

def write_report(out_dir, *, source, N, K, models, welfare, welfare_fine,
                 hist, contested, alpha_full, mean_kappa,
                 blocks_bin, blocks_fine, blocks_kappa, alpha_by_model,
                 n_unique_overall):
    n_max = int(len(hist) - 1)
    edge = max(1, n_max // 5)
    out = []
    A = out.append
    A(f"# Self-Evolve Bench Judge Reliability — {source}\n\n")
    A(f"**Dataset:** {source}, {N} tasks, {len(models)} judge model(s) "
      f"({', '.join(models)}), {K} annotator columns total. Inputs held "
      f"constant per task; only the judge call varies.\n\n")
    A("## TL;DR\n\n")
    A(f"Across the full {K}x{K} panel of judge calls, same-trace pairs are "
      f"distinguishable from independently-paired judgements "
      f"**{(0.5 + welfare / 2) * 100:.1f}%** of the time "
      f"(welfare = {welfare:.3f}; chance = 50%). Cronbach's a ~ {alpha_full:.2f}, "
      f"mean Cohen's k ~ {mean_kappa:.2f}. **{contested} of {N} tasks are "
      f"contested** ({edge+1}-{n_max-edge-1} of {n_max} calls pass).\n\n")
    A("## Pass-vote distribution\n\n")
    A("![Pass-vote distribution](figure_passdist.png)\n\n")
    A(f"Of {N} tasks, {int(hist[0] + hist[-1])} are unanimous "
      f"({int(hist[0])} unanimous fail, {int(hist[-1])} unanimous pass), "
      f"and **{contested} are contested**.\n\n")
    A("## Reliability summary\n\n")
    A("| Metric | Value |\n|---|---:|\n")
    A(f"| Evaluation welfare, binary (TVD-MI) | {welfare:.3f} |\n")
    A(f"| Evaluation welfare, finest histogram ({n_unique_overall} bins) | {welfare_fine:.3f} |\n")
    A(f"| Distinguishability (1/2 * TVD-MI + 1/2) | {0.5 + welfare / 2:.3f} |\n")
    A(f"| Mean Cohen's k | {mean_kappa:.3f} |\n")
    A(f"| Cronbach's a (full) | {alpha_full:.3f} |\n\n")
    if len(models) >= 1:
        A("### Per-block breakdown\n\n")
        A("| Block | TVD-MI binary | TVD-MI fine | Cohen's k | Cronbach's a |\n")
        A("|---|---:|---:|---:|---:|\n")
        for m in models:
            key = ("within", m)
            b = blocks_bin.get(key, float("nan"))
            bf = blocks_fine.get(key, float("nan"))
            bk = blocks_kappa.get(key, float("nan"))
            ba = alpha_by_model.get(m, float("nan"))
            A(f"| Within {m} | {b:.3f} | {bf:.3f} | {bk:.3f} | {ba:.3f} |\n")
        for i, ma in enumerate(models):
            for mb in models[i+1:]:
                key = ("between", ma, mb)
                b = blocks_bin.get(key, float("nan"))
                bf = blocks_fine.get(key, float("nan"))
                bk = blocks_kappa.get(key, float("nan"))
                A(f"| {ma} <-> {mb} | {b:.3f} | {bf:.3f} | {bk:.3f} | - |\n")
        A("\n")
    A("## Figures\n\n")
    A("![Pairwise evaluation welfare](figure_heatmap.png)\n\n")
    A("![Per-annotator reliability](figure_reliability.png)\n\n")
    A("## Actionable insights\n\n")
    A(f"1. **Adversarial spot-check on the {int(hist[0] + hist[-1])} unanimous tasks.**\n")
    A(f"2. **Triage human review of the {contested} contested tasks.**\n")
    (out_dir / "report.md").write_text("".join(out))


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        runs_path = write_smoke_fixture(args.rejudge_dir, args.source,
                                        args.smoke_n_tasks, args.smoke_n_calls,
                                        args.seed)
        print(f"[smoke] wrote synthetic fixture -> {runs_path}")
    else:
        runs_path = args.rejudge_dir / f"{args.source}_runs.jsonl"
        if not runs_path.exists():
            sys.exit(f"runs file not found: {runs_path}\n"
                     f"Generate it with harness/rejudge_n_times.py, or pass --smoke.")

    tasks, annotators, R, S = load_runs(runs_path)
    N, K = len(tasks), len(annotators)
    if N == 0 or K == 0:
        sys.exit(f"No data in {runs_path}")
    short_labels, _, _ = annotator_labels(annotators)
    print(f"Loaded {N} tasks x {K} annotators from {runs_path}")

    # Pairwise binary TVD-MI
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            M[i, j] = tvd_mi(R[:, i], R[:, j])
    welfare = offdiag_mean(M) if K > 1 else 0.0
    per_ann_reliab = np.array(
        [(M[i].sum() - M[i, i]) / (K - 1) if K > 1 else 0.0 for i in range(K)]
    )

    # Pairwise Cohen's kappa, block summaries, alpha
    K_kappa = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            K_kappa[i, j] = cohens_kappa(R[:, i], R[:, j])
    blocks_bin, models, idx_by_model = block_summaries(M, annotators)
    blocks_kappa, _, _ = block_summaries(K_kappa, annotators)
    mean_kappa = offdiag_mean(K_kappa) if K > 1 else 0.0
    alpha_full = cronbach_alpha(R) if K > 1 else float("nan")
    alpha_by_model = {}
    for m, idxs in idx_by_model.items():
        if len(idxs) >= 2:
            alpha_by_model[m] = cronbach_alpha(R[:, idxs])

    # Pass-vote distribution
    pass_counts = R.sum(axis=1)
    hist = np.bincount(pass_counts, minlength=K + 1)
    edge = max(1, K // 5)
    contested = int(((pass_counts > edge) & (pass_counts < K - edge)).sum())

    # Finest-histogram TVD-MI on overall_score
    S_round = np.where(np.isnan(S), np.nan, np.round(S, 4))
    alphabet = sorted({float(v) for v in S_round.flatten() if not np.isnan(v)})
    M_fine = np.zeros((K, K))
    if len(alphabet) > 1:
        for i in range(K):
            for j in range(K):
                xs = [v for v in S_round[:, i] if not np.isnan(v)]
                ys = [v for v in S_round[:, j] if not np.isnan(v)]
                # align on indices where both present
                pairs = [(float(a), float(b))
                         for a, b in zip(S_round[:, i], S_round[:, j])
                         if not (np.isnan(a) or np.isnan(b))]
                if not pairs:
                    M_fine[i, j] = 0.0
                    continue
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                M_fine[i, j] = tvd_mi_categorical(xs, ys, alphabet)
    blocks_fine, _, _ = block_summaries(M_fine, annotators)
    welfare_fine = offdiag_mean(M_fine) if K > 1 else 0.0

    # Save matrices
    np.save(out_dir / "mi_matrix.npy", M)
    np.save(out_dir / "mi_matrix_fine.npy", M_fine)

    # Figures
    make_figures(M, hist, per_ann_reliab, welfare, short_labels,
                 idx_by_model, out_dir)

    # Report
    write_report(out_dir, source=args.source, N=N, K=K, models=models,
                 welfare=welfare, welfare_fine=welfare_fine, hist=hist,
                 contested=contested, alpha_full=alpha_full,
                 mean_kappa=mean_kappa, blocks_bin=blocks_bin,
                 blocks_fine=blocks_fine, blocks_kappa=blocks_kappa,
                 alpha_by_model=alpha_by_model,
                 n_unique_overall=len(alphabet))

    print(f"  welfare (binary)             = {welfare:.4f}")
    print(f"  welfare (finest, {len(alphabet)} bins) = {welfare_fine:.4f}")
    print(f"  mean Cohen's kappa           = {mean_kappa:.4f}")
    print(f"  Cronbach's alpha (full)      = {alpha_full:.4f}")
    print(f"  unanimous: {int(hist[0] + hist[-1])} / {N}, contested: {contested} / {N}")
    print(f"  wrote: {out_dir}/report.md, figure_*.png, mi_matrix*.npy")


if __name__ == "__main__":
    main()
