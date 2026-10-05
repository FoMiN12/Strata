# Unsloth Q8_0 (experimental, manual install)

**Experimental, no setup entry yet.** This is the manual path from a fresh clone to a running server. It needs the
engine from branch `f-ple-q8` (until it merges: it teaches the engine to read Unsloth's Q8_0 PLE n-gram table;
without it the engine exits at load with "per_layer_token_embd.weight is Q8_0, not IQ4_NL, Q5_0 or FP8 (I8)").
Validated 2026-10-05 on 2x RTX 3090 + 154 GB RAM + Ryzen 7 9700X, Linux, CUDA 13.4: 38.9 tok/s decode at a 25K
reply, 705 tok/s reading a 43K prompt, full 262144 context, fp16 KV, 138 GB RAM in use.

## What the file is

`unsloth/Qwen3.8-Flash-Next-GGUF`, folder `Q8_0`, revision `ff34bcd`: six shards, 188.2 GB. Same arch and tensor
names as the other packs. Routed experts all **Q8_0** (119.53 GiB together); dense projections and the PLE key
Q8_0; the 54.4 GB PLE n-gram table Q8_0 (the engine reads it in place from shard 3 - that is the `f-ple-q8`
branch). Expert blobs are read from the GGUFs in place (native path, like UD-Q4_K_XL); nothing is re-quantized.

## Requirements

- **Disk**: ~190 GB for the six shards + ~2.5 GB for the pack and the draft (`packs/q8_0` 1.4 GB, `mtp/rt` 0.8 GB).
- **GPU**: NVIDIA. One card of 24 GB works; two work faster. Not run on AMD (HIP) - treat it as NVIDIA-only.
- **RAM**: see [Choosing the mode](#choosing-the-mode) - from ~65 GB (small budget, SSD-backed) to ~140 GB (full
  two-card split at 262K context).
- **CUDA toolkit** 12 or 13 (built with 13.4), a C++20 compiler, Ninja, and Python 3.10+ for the tools.
- **Network**: the draft head is fetched from Hugging Face (~4.9 GB of range reads, step 4).

## Choosing the mode

Two ways to run; pick by RAM and card count. The engine **refuses `--resident-budget-gib` together with a layer
split** - a split always loads everything.

| Mode | Needs (RAM free) | What happens | Speed |
| --- | --- | --- | --- |
| **Two GPUs, split** | ~140 GB at 262K ctx, ~130 GB at 32K | all 119.5 GiB of experts into RAM at start, layers split across the cards, no file reads while running | measured here: 38.9 tok/s decode, 705 tok/s prompt |
| **One GPU, budget** | budget + ~15 GB | `--resident-budget-gib N` pins N GiB of experts (the most-used, by profile rank) in RAM; the GPU cache holds ~3000 more; the rest are read from disk when routed | slower the smaller N is; the log prints the cache hit rate |

Rules of thumb:

- **2 cards and ~140 GB free** - take the split. It is the fastest mode and never touches the disk mid-reply.
  (Same flow measured on UD-Q4_K_XL, #498: 31 tok/s one card vs 64-78 tok/s split.)
- **~128-140 GB free** - split works with a smaller context (32K needs ~130 GB total), or one card with
  `--resident-budget-gib 100`.
- **Less RAM** - one card, budget = free RAM minus ~15 GB (dense, draft, KV, OS). Every GiB under 119.5 means
  some experts come from disk: decode stays usable if the budget covers the hot experts (the profile ranks them;
  the log's "decode expert cache hit rate" line tells you - watch it after a real conversation).
- **Changing RAM usage later**: single-card mode is the knob - edit `--resident-budget-gib`. In split mode there
  is no knob; the only levers are `--max-context` (above 64K the KV cache moves to RAM: fp16 KV is ~26 KB per
  token, ~7 GB at 262K) and dropping to one card with a budget.

## 1. Download the model

From `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/ff34bcd/Q8_0` get the six shards into one
folder (keep the names - the engine finds shards 2-6 from shard 1's name). Check them before packing:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` | 10,946,624 | `2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2` |
| `Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf` | 682,434,912 | `494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e` |
| `Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf` | 54,400,261,312 | `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20` |
| `Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf` | 49,446,841,216 | `bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a` |
| `Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf` | 49,668,930,400 | `232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97` |
| `Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf` | 34,015,618,784 | `e3bfa34c424fdb65b9a79ace3acb917ce6cbb7be6ae325bd5d26c24e39cf6755` |

    sha256sum Qwen3.8-Flash-Next-Q8_0-0000*.gguf

Do NOT download `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` - it cannot be used (step 4 builds the draft instead).

## 2. Clone and build the engine

    git clone https://github.com/FoMiN12/Strata.git && cd Strata
    git checkout f-ple-q8
    python3 -m venv .venv
    .venv/bin/pip install numpy pyyaml jinja2 regex requests tqdm cmake ninja
    .venv/bin/python -c "import setup; setup.get_llama_cpp()"   # pinned llama.cpp -> third_party/llama.cpp

Build (sm_86 is the 3090; use your card's compute capability - `nvidia-smi --query-gpu=compute_cap --format=csv`):

    cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build --target strata -j

`build/strata --help` should print; `build/strata-device` lists the cards. If a tool complains about `gguf-py`,
export `STRATA_GGUF_PY=third_party/llama.cpp/gguf-py`.

**CPU: AVX-512 is not needed.** The build always compiles both the AVX-512 and the AVX2 CPU expert kernels (the
AVX-512 flags go on those two source files only, never on the whole binary), and the engine picks at run time by
cpuid - a CPU without AVX-512 simply runs the AVX2 kernels (slower CPU-side experts; the GPU path is unaffected).
To test the fallback on a new CPU: `STRATA_FORCE_AVX2=1` (or `STRATA_NO_IQ512=1` for just the i-quant kernels).
Only if the CPU lacks even AVX2 (the default floor) there is an experimental older-CPU build: add
`-DSTRATA_ISA_FLOOR=avx` (AVX1) or `-DSTRATA_ISA_FLOOR=none` (SSE4.2) to the cmake line - it changes the ggml
dispatch floor, is not what to use for a Zen/Intel chip without AVX-512, and cannot combine with
`-DSTRATA_PORTABLE=ON`.

## 3. Pack the model

One pass over all six shards (reads 188 GB; an hour-class job on an HDD, faster on NVMe):

    .venv/bin/python tools/iq_pack.py \
      --gguf /path/to/models/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
      --out packs/q8_0 --compat-bf16

Expect `index: 1079 tensors, 303 served natively` and a `conversions.json` with max |err| ~0.0144 (the BF16
rounding of the F32 hyper-connections - same as UD-Q4_K_XL). Layers 17 and 35 straddle shards; the pack's
`native_experts.txt` v4 records that per role - no action needed.

## 4. Build the draft head (serve mode needs it)

`serve` requires `--mtp DIR` on top of `--spec` >= 2. The draft is the base model's, the same one every other
pack uses; `mtp_fetch` range-reads its 31 tensors (~4.9 GB) from the BF16 checkpoint - resumable, rerun on a
network hiccup:

    .venv/bin/python tools/mtp_fetch.py fetch --out mtp
    .venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
    .venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
    cp data/draft_vocab.bin mtp/rt/draft_vocab.bin

## 5. Configure

Create `strata-q8_0.json` at the repo root. **Two GPUs** (the measured mode; needs ~140 GB free at 262K):

    {
      "exe": "build/strata",
      "args": [
        "--pack", "packs/q8_0",
        "--native", "/path/to/models/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
        "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
        "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
        "--mtp", "mtp/rt", "--kv", "fp16", "--max-context", "262144"
      ],
      "gpu": [0, 1],
      "layer_split": "auto",
      "cwd": ".",
      "tokenizer": "packs/q8_0/tokenizer",
      "model_name": "qwen3.8-flash-next-q8_0",
      "log": "strata-q8_0.log",
      "host": "127.0.0.1",
      "port": 8001
    }

**One GPU**: delete the `"gpu"` and `"layer_split"` lines and add to `args`

    "--resident-budget-gib", "100"

with the number set by the table above (budget + ~15 GB must fit in free RAM). The rest of the args:

- `--native SHARD1` - experts are read from the GGUFs in place; the path must stay valid.
- `--spec 4 --spec-min-p 0.5` - MTP speculation depth; a native pack refuses to start without `--spec` >= 2.
- `--kv fp16` - the engine's highest-quality KV (it is also the default; `int8` / `q4_0` / `k8v4` trade quality
  for RAM/VRAM). Above 64K context the KV lives in RAM: ~7 GB at 262K.
- `--max-context` - 262144 is the model's trained length. Lower it (225000 / 200000 / 131072) if RAM runs short;
  every token costs ~26 KB of KV.

## 6. Launch

    .venv/bin/python -m serve.server --engine strata --config strata-q8_0.json --port 8001

The first start reads the shards once and can stall the PC for a minute or two - normal. The log
(`strata-q8_0.log`) should show, in order:

    layer split auto: ... K=24 ... the caches hold X of 24576 profiled pairs
    PLE on, table 320001536 rows of .../Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf
    strata mtp: draft layer loaded, ... MiB of VRAM
    strata serve: layer split: layers 0-23 (CUDA0), 24-47 (CUDA1), one hand-off per window
    serve: ready

Then the browser app opens; or check by hand:

    curl -s http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
      "model": "qwen3.8-flash-next-q8_0", "messages": [{"role": "user", "content": "Write a haiku about 48 layers."}],
      "temperature": 0, "max_tokens": 64}'

Keep `free -g` open during the first run: in split mode it must not drop under ~8 GB. If pinning fails (an
`mlock` / `cannot pin` line), restart with `ulimit -l unlimited`.

## Troubleshooting

| Message | Meaning / fix |
| --- | --- |
| `is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK` | missing `--spec 4` (and `--prefill auto`) in the config |
| `strata serve: needs --spec T, --mtp DIR and --prefill CHUNK` | step 4 not done - build `mtp/rt` |
| `per_layer_token_embd.weight is Q8_0, not IQ4_NL, Q5_0 or FP8 (I8)` | engine without the Q8_0 table path - `git checkout f-ple-q8` and rebuild |
| engine exits when given both a budget and `"gpu": [0, 1]` | by design: remove `--resident-budget-gib` for a split |
| RAM thrash / OOM at start | lower `--max-context`, or switch to one GPU with a smaller budget |
| slow decode, low `decode expert cache hit rate` | raise the budget (single card) or check the disk tier is an SSD |

## Optional: the smaller Q5_0 PLE table

The shipped Q8_0 table (54.4 GB) is read through the OS page cache; on a RAM-tight PC it competes with the expert
arena. `tools/ple_q5_pack.py` converts it to Q5_0 (35.2 GB, ~3% block error - the same band as the FP8 tables the
other packs ship):

    .venv/bin/python tools/ple_q5_pack.py --src /path/to/models/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf \
      --out /path/to/models/ple-q5_0-of-00001.gguf

and add `"--ple-gguf", "/path/to/models/ple-q5_0-of-00001.gguf"` to the config args. Takes ~10 minutes; the
converter self-checks against the source blocks. Skip it if RAM is comfortable - the shipped table is exact.

## UD-Q6_K_XL instead?

Unsloth's `UD-Q6_K_XL` folder (169.2 GB) keeps the same Q8_0 PLE table (shards 2-3 are byte-identical to Q8_0's -
hardlink them to save 55 GB) with Q6_K gate/up experts: ~101.7 GiB of experts instead of 119.5, so the split
needs ~15 GiB less RAM and caches ~18% more experts per card. The CPU pool runs Q6_K gate/up through ggml's
generic dots (no fast multi-token kernel), so measure both before choosing. Everything in this guide applies
unchanged - same branch, same pack command, same config with the paths swapped.
