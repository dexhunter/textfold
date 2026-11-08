#!/usr/bin/env python3
"""Frozen CPU baselines for TextFold intake.

Each target times one function on a fixed in-memory workload. Setup (building
strings, tokens, or HTML) is outside the clock. The clocked section disables
the cyclic GC so samples are the function's own work.

Run from the repository root:

    python3 osspr-intake/harness/bench.py <target>

Prints one JSON object: median_ms, noise_pct (pstdev / median * 100), samples_ms.
"""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WARMUP = 3
SAMPLES = 11


def _english_paragraphs(n=400):
    out = []
    for i in range(n):
        out.append(
            f"Claim {i}. The estimator tracks a latent state x{i} whose variance is sigma squared {1 + i / 10:.2f}. "
            f"We compare it with the baseline of Smith et al. (Fig. {1 + (i % 8)}). "
            f"Held-out gain is {2.2 + (i % 5):.2f} percent, significant at p<0.01. "
            "The paragraph stays long so sentence splitting and token alignment both have work to do."
        )
    return out


def _cjk_paragraph(n_chars, salt):
    sentence = (
        "本文提出了一种新的方法，我们在多个基准上进行了实验，结果表明这个方法是有效的。"
        f"第{salt}组吸附量下降，结果支持结论。"
    )
    return (sentence * ((n_chars // len(sentence)) + 2))[:n_chars]


def _formulas(n=160):
    out = []
    for i in range(n):
        out.append(
            rf"E_{{{i}}} = \sum_{{k=1}}^{{n}} \frac{{x_{{k}}^{{{(i % 5) + 1}}}}}{{1 + \sigma_{{{i % 7}}}^2}}"
            rf" + \mathrm{{softmax}}(W_{{{i}}} h)"
        )
    return out


def _time(fn):
    """Time fn with the cyclic GC off during the call and a collection between samples."""
    samples = []
    for i in range(WARMUP + SAMPLES):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter_ns()
            fn()
            elapsed = (time.perf_counter_ns() - t0) / 1e6
        finally:
            gc.enable()
        if i >= WARMUP:
            samples.append(elapsed)
    med = statistics.median(samples)
    noise = (statistics.pstdev(samples) / med * 100) if med else 0.0
    return {
        "median_ms": round(med, 3),
        "noise_pct": round(noise, 2),
        "samples_ms": [round(x, 3) for x in samples],
    }


def load_targets():
    from textfold import ladder, langs, semantic, tokens

    paragraphs = _english_paragraphs()
    book = "\n\n".join(paragraphs)
    book_toks = tokens.tokenize_text(book)
    # budgets_of calls _cjk_share once per unit; a 400-paragraph import does the same
    unit_texts = ["".join(tk.get("t") or "" for tk in tokens.tokenize_text(p)) for p in paragraphs]

    cjk_normed = []
    for i in range(4):
        toks = tokens.tokenize_text(_cjk_paragraph(2500, i))
        norms = [tokens.norm(tk, {}) for tk in toks]
        cjk_normed.append((norms[:60], norms))

    norm_toks = book_toks

    formulas = _formulas()

    # language samples at detect's own 40_000-character cap, mixed scripts
    zh = _cjk_paragraph(40000, 9)
    en = (book + "\n")[:40000]
    de = ("Der Schätzer folgt einem latenten Zustand und der Gewinn ist signifikant. " * 800)[:40000]
    fr = ("L'estimateur suit un état latent et le gain reste significatif sur le jeu. " * 800)[:40000]
    detect_texts = [en, zh, de, fr] * 6  # 24 documents, each already at the cap

    def cjk_share():
        acc = 0.0
        for text in unit_texts:
            acc += ladder._cjk_share(text)
        return acc

    def tokenize_text():
        return len(tokens.tokenize_text(book))

    def lcs_pairs():
        n = 0
        for brief, full in cjk_normed:
            n += len(tokens.lcs_pairs(brief, full))
        return n

    def norm():
        atoms = {}
        n = 0
        for tk in norm_toks:
            if tokens.norm(tk, atoms):
                n += 1
        return n

    def tex_math():
        n = 0
        for latex in formulas:
            if semantic.tex_math(latex).get("alttext"):
                n += 1
        return n

    def detect():
        n = 0
        for text in detect_texts:
            if langs.detect(text):
                n += 1
        return n

    return {
        "cjk_share": ("textfold/ladder.py", "_cjk_share", cjk_share),
        "tokenize_text": ("textfold/tokens.py", "tokenize_text", tokenize_text),
        "lcs_pairs": ("textfold/tokens.py", "lcs_pairs", lcs_pairs),
        "norm": ("textfold/tokens.py", "norm", norm),
        "tex_math": ("textfold/semantic.py", "tex_math", tex_math),
        "detect": ("textfold/langs.py", "detect", detect),
    }


def main(argv):
    targets = load_targets()
    if len(argv) != 2 or argv[1] not in targets:
        print(f"usage: python3 osspr-intake/harness/bench.py <{'|'.join(targets)}>", file=sys.stderr)
        return 2
    name = argv[1]
    file, function, fn = targets[name]
    out = _time(fn)
    out.update(target=name, file=file, function=function, samples=SAMPLES, warmup=WARMUP)
    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
