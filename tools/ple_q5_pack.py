#!/usr/bin/env python3
"""tools/ple_q5_pack.py - the PLE n-gram table as Q5_0, from a Q8_0 GGUF (OrcaRouter's table type, #296).

    python tools/ple_q5_pack.py --src <model>-00003-of-0000N.gguf --out <ple-q5_0.gguf> [--start N]

Reads `per_layer_token_embd.weight` (ne [160, 320,001,536]) wherever it lives among the shards given and writes a
one-tensor Q5_0 GGUF the engine's PleTable reads natively (src/kernels/ngram.cpp: Q5_0 rows of 5 x 22 B). Q5_0's
block (2 B fp16 scale, 16 B low nibbles, 4 B high bits, v = d * (q16 - 16), q16 in [0, 31]) is bit-identical to
ggml's; strata::dequantize_q5_0 (include/strata/artifact/dequant.hpp) is the consumer checked against it.

Why Q5_0 and not FP8/I8: the shipped table is a ~5.6K-level quantization (5,605 unique values per 167M samples,
measured on unsloth's BF16 folder 2026-10-05), FINER than the E4M3 grid; Q5_0 (~2-3% block error, worst ~6% at
small values against an E4M3-derived source) preserves it best of the three types the engine accepts, at 35.2 GB
against FP8's 51.2. Q8_0 source (unsloth's Q8_0/UD-Q6 shard 3) and BF16 carry the same values; Q8_0 is smaller to
read and was verified equal to BF16 within Q8_0 error (check-rows output).

--start N resumes an interrupted run into the .part file (whole BATCH-row batches are complete on disk).
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

GGML_Q5_0 = 7
TENSOR_NAME = "per_layer_token_embd.weight"
NE0, NE1, ROWS_PER_BLOCK = 160, 320001536, 5
RB = ROWS_PER_BLOCK * 22                      # 110 B per row
SRC_RB = 170                                  # one Q8_0 row of the table: 160 values = 5 blocks of (32 B + 2 B scale)
BATCH = 8192


def find_source(srcs: list[pathlib.Path]) -> tuple[GGUFFile, object]:
    for s in srcs:
        g = GGUFFile(s)
        for t in g.tensors:
            if t.name == TENSOR_NAME:
                if t.type_name != "Q8_0":
                    sys.exit(f"{s.name}: {TENSOR_NAME} is {t.type_name}, expected Q8_0")
                return g, t
    sys.exit("per_layer_token_embd.weight is not in the shard(s) given")


def q8_dequant(rb34: np.ndarray) -> np.ndarray:
    """(BATCH, 5, 34) Q8_0 blocks -> float32 (BATCH, 160). ggml's block_q8_0: f16 d FIRST, then 32 int8."""
    d = rb34[:, :, 2:34].view(np.int8).astype(np.float32) * rb34[:, :, 0:2].view(np.float16).astype(np.float32)
    return d.reshape(rb34.shape[0], 160)


def q5_quant(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 (BATCH, 160) -> (f16 (BATCH, 5) scales, u8 q16 (BATCH, 160)), ggml's Q5_0 rounding."""
    blk = v.reshape(-1, 5, 32)
    amax = np.maximum(np.max(np.abs(blk), axis=2), 1e-30)
    d16 = (amax / 15.0).astype(np.float16)
    qq = np.round(blk / d16[:, :, None].astype(np.float32)) + 16.0
    np.clip(qq, 0, 31, out=qq)
    return d16, qq.astype(np.uint8)


def pack_blocks(d16: np.ndarray, q16: np.ndarray) -> np.ndarray:
    """(rows, 5) scales + (rows, 160) q16 -> the flat Q5_0 block bytes.

    Layout per ggml-common.h block_q5_0 and strata::dequantize_q5_0: f16 d (0:2), qh (2:6, bit j = 5th
    bit of value j), qs (6:22, low nibble of value j, high nibble of value j+16)."""
    nb = d16.size // 1
    d1, q1 = d16.reshape(-1, 1), q16.reshape(-1, 32)
    lo = q1[:, :16] | (q1[:, 16:] << 4)                        # qs: 16 bytes
    bh_lo = np.packbits((q1[:, :16] >> 4).astype(np.uint8), axis=1, bitorder="little")  # qh bytes 0-1
    bh_hi = np.packbits((q1[:, 16:] >> 4).astype(np.uint8), axis=1, bitorder="little")  # qh bytes 2-3
    blk = np.empty((nb, 22), np.uint8)
    blk[:, 0:2] = d1.view(np.uint16)
    blk[:, 2:4] = bh_lo
    blk[:, 4:6] = bh_hi
    blk[:, 6:22] = lo
    return blk.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, nargs="+", help="the shard(s) holding the PLE table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="row to resume at (from the .part's size)")
    ap.add_argument("--check-rows", type=int, default=0, help="max rows to convert (0: all); for the round-trip check")
    a = ap.parse_args()

    g, t = find_source([pathlib.Path(s) for s in a.src])
    if tuple(t.shape[:2]) != (NE0, NE1):
        sys.exit(f"{g.path.name}: unexpected PLE shape {tuple(t.shape)}")
    base = g.data_start + t.offset
    src_rows = (pathlib.Path(g.path).stat().st_size - base) // SRC_RB
    n_rows = min(src_rows, NE1)
    out = pathlib.Path(a.out)
    part = out.with_name(out.name + ".part")
    if a.start and (not part.exists() or part.stat().st_size != a.start * RB):
        sys.exit(f"--start {a.start} does not match {part.name}'s size "
                 f"{part.stat().st_size if part.exists() else 'missing'}")
    mode = "r+b" if a.start else "wb"

    src = open(g.path, "rb")
    src.seek(base + a.start * SRC_RB)
    dst = open(part, mode)
    v = np.empty((BATCH, 160), np.float32)
    t0, maxrel, maxabs = time.time(), 0.0, 0.0
    done = a.start
    limit = NE1 if not a.check_rows else min(NE1, a.start + a.check_rows)
    while done < limit:
        raw = np.frombuffer(src.read(BATCH * SRC_RB), dtype=np.uint8)
        if raw.size < BATCH * SRC_RB:
            break
        v[:] = q8_dequant(raw.reshape(BATCH, 5, 34))
        d16, q16 = q5_quant(v)
        deq = (d16.astype(np.float32)[:, :, None] * (q16.reshape(BATCH, 5, 32).astype(np.float32) - 16.0)).reshape(BATCH, 160)
        err = np.abs(deq - v)
        maxabs = max(maxabs, float(err.max()))
        maxrel = max(maxrel, float((err / np.maximum(np.abs(v), 1e-9)).max()))
        dst.write(pack_blocks(d16, q16).tobytes())
        done += BATCH
        if (done // BATCH) % 512 == 0:                          # ~10M rows
            el = time.time() - t0
            print(f"rows {done}/{NE1} ({el:.0f}s, {done / max(el, 1) / 1e6:.1f}M rows/s)", flush=True)
    dst.truncate(n_rows * RB)
    dst.close()

    hdr = struct.pack("<I", 0x46554747) + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 0)
    hdr += struct.pack("<Q", len(TENSOR_NAME)) + TENSOR_NAME.encode()
    hdr += struct.pack("<I", 2) + struct.pack("<Q", NE0) + struct.pack("<Q", NE1)
    hdr += struct.pack("<I", GGML_Q5_0) + struct.pack("<Q", 0)
    hdr += b"\0" * ((-len(hdr)) % 32)                           # data starts aligned (ggml default 32)
    with open(out, "wb") as f:
        f.write(hdr)
        with open(part, "rb") as p:
            while True:
                b = p.read(1 << 24)
                if not b:
                    break
                f.write(b)
    part.unlink(missing_ok=True)
    el = time.time() - t0
    print(f"[check] max |err| {maxabs:.6f} ({maxabs / 0.0491 * 100:.2f}% of the table's max value), "
          f"max rel {maxrel * 100:.2f}%")
    print(f"packed Q5_0 PLE table: {n_rows * RB / 2**30:.1f} GiB in {el:.0f}s -> {out}")


if __name__ == "__main__":
    main()
