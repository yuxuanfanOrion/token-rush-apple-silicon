# Token Rush on Apple Silicon

The same engine on a Mac, through MLX: `tokenrush/mlx/`. Same model, same
published int4 checkpoint, same drafts (the MTP head and DFlash2), same serving
layer. Measured on one machine, a MacBook Pro with an **M5 Pro and 64 GB**,
macOS 26.4, mlx 0.32.3, 2026-10-07. Development numbers from one laptop, not a
Phase 4 style sitting: the machine heats under sustained load and repeated runs
of the same configuration drift by 10-15% (see "Heat" below).

```
uv sync --extra mac-bench                     # mlx, CPU torch (the shared serving code imports it), mlx-lm for the benches
uv run python -m tokenrush.serve              # picks the MLX backend on macOS; 64k context by default
uv run --extra bench pytest tests/mac -q      # kernels, engine and session on a small random model
```

## The wall

`bench/mac/bandwidth.py`: a streaming reduction over a 4 GB buffer reads at
**291 GB/s**, 1/5.8 of the 5090's 1701. The text path streams 13.65 GB per token
(the embedding table excluded), so **the raw-decode ceiling is 21.3 tok/s**.

MLX's own int4 matmul at M = 1 already streams at 95-97% of that at every
layer shape, in-graph over more than 1 GB of distinct weights per shape. As on
the 5090, dense weight streaming is solved before we start.

## No conversion

Our packing (element 2j in the low nibble of byte j) read as little-endian
uint32 is MLX's affine 4-bit layout, and both dequantize as `q * scale + mn`.
The published checkpoint loads with a `view(uint32)` per matrix: no
requantization, the GPTQ codes and every quality number in
`docs/quantization.md` carried over unchanged (`tokenrush/mlx/weights.py`).

## Correctness

The reference is mlx-lm's Qwen3.5 implementation fed the same int4 bytes, built
in memory (`bench/mac/mlx_lm_ref.py`): a second implementation that shares no
kernel with ours (GDN, attention, norms, rope all differ).
`bench/mac/diff_mlx_lm.py`, three prompts, our engine decoding step by step and
mlx-lm scoring the sequence teacher-forced:

| prompt | top-1 agreement | KL mean | KL max |
|---|---|---|---|
| prose | 0.979 (one near-tie in 48) | 4.0e-4 | 2.6e-3 |
| code | 1.000 | 2.2e-4 | 2.6e-3 |
| math | 1.000 | 2.7e-4 | 3.8e-3 |

Two orders of magnitude below the quantization's own KL to bf16 (0.0232).
Greedy output matches mlx-lm's for the first 114 of 128 tokens on the decode
benchmark prompt. `tests/mac` checks every kernel against a plain-op reference,
chunked prefill against token-by-token decode, the speculative step against raw
greedy at every acceptance length, and a snapshot resume against a cold prefill
(each with a negative control that must fail).

## Raw decode

`bench/mac/decode.py`, 256 greedy tokens, same bytes for both:

| | tok/s | % of the wall |
|---|---|---|
| ours, one eval per step | 17.5 | 82.1% |
| **ours, pipelined** (step t+1 built while t runs; the token never visits the host) | **18.5** | **86.7%** |
| mlx-lm 0.32 | 17.8 | 83.4% |

The same picture as the 5090: raw decode is close to the wall in every engine,
and the margin over the best rival is a few points. The headroom is speculation.

## The verify step is not free here

On the 5090 a K = 7 verify costs 1.13x a decode step (`docs/progress.md` step 28),
which is what made deep speculation pay. On this Mac, MLX's int4 matmul reads the
weights once per row for 2 <= M < 16 (`bench/mac/bandwidth.py`, `qmv_rows.py`):

| rows M | 2 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|
| cost vs M = 1 (gate_up) | 1.04x | 1.36x | 2.62x | 3.39x | 1.72x |

M = 16 is cheaper than M = 8 because it switches to a tiled kernel on the M5
GPU's neural accelerators, but only for wide outputs: at N = 5120 (out_proj,
down_proj) M = 16 costs 5x, too few tiles and no split-K.

What we tried, all in `tokenrush/mlx/kernels.py`, measured in-graph:

- **scalar M-row GEMV** (`qmv_rows`): each weight dequantized once, applied to all
  rows. Ties MLX up to M = 4, ~3x at M = 8. Not a code problem: 8 FMA per weight
  at 291 GB/s is ~9 TFLOPS of fp32 scalar work.
- **simdgroup_float8x8** (`qmm8`), dequantizing straight into the lane's fragment
  registers: flat in M but ~1.7-2.0x even at M = 1. The same flops on the same ALUs.
- **Metal 4 `matmul2d` on the neural accelerators** (`qmm_nax`), W supplied
  transposed so each lane's four codes are half a word, bf16 dequant in registers,
  split-K for narrow outputs: **1.45-1.85x at M = 8** against MLX's 2.2-3.6x, but
  1.2-1.5x at M = 1 (the dequantization is the fixed cost).

So the engine routes by row count: M <= 4 to MLX, 5..8 to `qmm_nax`, on M5 and
later GPUs only (`applegpu_g17` and up; earlier chips keep MLX's kernels, untested). Interleaved
measurement of the whole verify step (`bench/mac/step_profile.py`): M = 8 goes
from 215 ms to 113 ms; M <= 5 is unchanged. Closing the remaining gap at M = 4..8
means taking the dequantization off the critical path, the open item below.

## Speculative decoding

Same drafts as the CUDA engine, same acceptance rules (greedy: argmax; sampled:
one draw per position, accepted while it equals the draft). Each step's draft
chain and verify are one lazy MLX graph, evaluated once; the only value read
back is n accepted. The GDN layers use a **deferred commit**: a verify step
writes no recurrent state, the caller keeps the rows' (k, v, g, beta), and the
next call replays the accepted n + 1 of them before its own rows. The CUDA
engine instead writes K + 1 state snapshots per step (48 x 8 x 3.1 MB), cheap at
1.7 TB/s and not here.

`bench/mac/spec_bench.py`, Phase 0 prompts (`bench/mtp_accept.py`, thinking
off), 256 greedy tokens, two passes in one sitting (ranges are the two passes):

| | essay | code | math |
|---|---|---|---|
| raw | 16.4-17.2 | 16.7-17.4 | 16.7-17.3 |
| MTP chain, K = 3 | 28.4-33.2 (2.61/step) | 31.8-38.2 (2.98) | 35.1-42.8 (3.32) |
| **DFlash2** (K = 7) | 25.3-28.2 (3.08) | **39.9-42.2** (4.41) | **46.9-49.9** (5.22) |

The better draft per family is worth 1.7-1.9x over raw on essays (MTP), 2.4x on code
and 2.7-2.9x on math (DFlash2), against the CUDA engine's 2.3x / 3.5x / 3.7x
(228 / 357 / 378 over 100.9); the difference is the verify cost above. DFlash2's acceptance per step matches what
it does on the 5090, which is also evidence for the port. The MTP chain's best
depth is K = 3 here (K = 4: the M = 5 step jumps), not the CUDA chain's 3-4 with
dynamic depth; dynamic depth loses on this machine.

Greedy speculative output equals raw greedy until the first bf16 near-tie (45,
41, 140 tokens on the three prompts): the verify step's M-row matmul rounds
differently from the M = 1 one. The CUDA engine's `consistent` mode, which makes
them bit-identical, has no Mac counterpart yet.

## Serving

`python -m tokenrush.serve` on macOS loads `tokenrush/mlx/serve.py` and the MLX
`Session` (`tokenrush/mlx/session.py`, same interface as `tokenrush/session.py`),
so the OpenAI and Anthropic APIs, tool calls and the output parser are the CUDA
server's code. Snapshots keep references to the immutable recurrent states and
pending rows and copy only the conv rings. Checked end to end: an OpenAI chat
request for code ran at **61 tok/s** on DFlash2 (6.5 tokens per step), an
Anthropic request in Chinese on the MTP chain at 25 tok/s greedy (2.1 per step),
and a second turn reused the first turn's prompt.

## Heat

This is a laptop. Over a long benchmark session the same configuration got
slower: the verify step at M = 4 took 74 ms in one run and 115 ms in a later one,
while `pmset -g therm` reported no thermal warning. Compare configurations interleaved in one run,
as `step_profile.py` does, never across runs.

## Not done

- **mx.compile.** The CUDA graph's job (no per-op host work) is mostly done by
  MLX's lazy graph plus pipelining (one eval per step; 1 tok/s between the eager
  and pipelined loops). Positions are Python ints in the attention slices and
  rope offsets, so compiling the step needs them on device first.
- **The verify kernel at M = 4..8.** `qmm_nax` pays 20-50% at M = 1 for
  dequantization; a version that overlaps it with the loads, or dequantizes into
  threadgroup memory shared by more output rows, is the next lever on the
  speculative numbers.
- **Long context.** Only short prompts were measured. KV is bf16 and contiguous;
  no fp8 cache, no needle test.
- **The quality row on Mac.** KL against bf16 needs the bf16 checkpoint (55.6 GB);
  the engine-vs-mlx-lm KL above is the correctness evidence, and the quantization
  numbers carry over because the bytes are the same.
- **Other Apple GPUs.** Everything was measured on one M5 Pro. M1-M4 run the same
  code with the `matmul2d` kernel switched off; their walls and verify costs are
  unmeasured.
