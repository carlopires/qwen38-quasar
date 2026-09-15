# Execution Plan — qwen38-quasar

**Status:** draft / plan of record for milestone 1 (all-W4A4 QUASAR baseline)
**Created:** 2026-09-15
**Supersedes:** nothing. Derived from the implementation handoff
`/tmp/efficientthink-quasar-ninfer-implementation-handoff.md` (2026-09-15).

This document is the *planning* artifact: what we build, in what order, what each
stage costs, and the exact conditions under which we are allowed to spend more
money. It is not a substitute for the handoff, which remains the implementation
specification.

---

## 1. Objective

Produce, from the BF16 EfficientThink Qwen3.8-27B checkpoint, a faithful
all-W4A4 QUASAR NVFP4 model that:

1. is trained by quantization-aware distillation from EfficientThink as its own
   frozen teacher (never distilled back toward vanilla Qwen);
2. exports as a standard Hugging Face `compressed-tensors` checkpoint requiring
   no custom inference code;
3. loads in vLLM;
4. converts to a `.ninfer` artifact via `carlopires/ninfer-rtx5090-mobile`;
5. serves through NInfer's OpenAI-compatible API and is usable from MCC.

```text
EfficientThink BF16 (teacher + student init)
  → QUASAR QAD/QAT
  → compressed-tensors NVFP4 W4A4
  → vLLM validation
  → .ninfer artifact
  → ninfer-serve
  → MCC
```

**Explicit non-goal for milestone 1:** mixed precision. All 496 transformer
Linear modules are W4A4. Mixed precision is a follow-up experiment (§51 of the
handoff) and gets its own recipe name.

---

## 2. Repository boundary

This repository (`carlopires/qwen38-quasar`) owns:

- QUASAR reconstruction / QAT code
- EfficientThink teacher/student setup
- Open-PerfectBlend data preparation
- QAD training and export
- model-side fidelity evaluation
- provenance and reproducibility records

`carlopires/ninfer-rtx5090-mobile` continues to own: `.ninfer` artifact format,
Qwen3.8 runtime registration, CUDA/NVFP4 kernels, `.ninfer` conversion, native
W4A4 execution, MTP/DFlash2, serving, and runtime parity tests.

**Do not copy the NInfer runtime into this repository.** NInfer-side work happens
on branch `feat/efficientthink-quasar` in the NInfer checkout, never on `master`.

---

## 3. Verified environment facts

Established 2026-09-15 on the development workstation. These are measurements,
not assumptions, and they retire three of the four structural unknowns.

### 3.1 Toolchain and local runtime

| Item | Value |
|---|---|
| `gh` auth | logged in as `carlopires`, scopes `repo, workflow, read:org, gist` |
| git identity | `Carlo Pires <carlopires@gmail.com>` |
| `uv` | 0.11.8 |
| Python | **3.14.4** (uv-managed, pinned in `.python-version`) |
| torch | **2.14.0+cu130**, CUDA 13.0 runtime |
| transformers | 5.17.0 |
| compressed-tensors | **0.18.0** |
| accelerate / datasets / safetensors / numpy | 1.15.0 / 5.0.1 / 0.8.0 / 2.5.3 |
| local GPU | RTX 5090 Laptop, **sm_120** (native in torch arch list), 25.15 GB |
| host RAM | 183 GB total (~37 GB currently available) |

The Python decision required by handoff §8 is therefore **resolved by
measurement**: the full ML stack (`torch`, `transformers`,
`compressed-tensors`, `accelerate`) resolves, installs, and imports on Python
3.14.4, and torch publishes `cp314` wheels with `sm_120` support. We do not
downgrade to 3.13/3.12.

`uv.lock` is committed and pinned.

### 3.2 Structural verification — 496/496 confirmed

Read from `config.json` + `model.safetensors.index.json` at the pinned
EfficientThink revision `882e1af3b8f5844f3b3163b132e343c06e91908f`. No weights
downloaded.

| Component | Layers | Linears/layer | Subtotal |
|---|---|---|---|
| `linear_attention` (Gated DeltaNet) | 48 | `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | 240 |
| `full_attention` | 16 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 64 |
| `mlp` | 64 | `gate_proj`, `up_proj`, `down_proj` | 192 |
| | | **total** | **496** |

This matches handoff §21 exactly. Two traps identified that the patcher must
respect:

- `attn_output_gate: true`, yet only **4** attention Linears exist → the output
  gate is fused into `q_proj`. The patcher must not look for a separate gate
  Linear, and must not be confused by `q_norm` / `k_norm` (which are 1-D, not
  Linear).
- GDN layers carry `conv1d.weight` (a `Conv1d`), `A_log`, `dt_bias` → excluded
  naturally by class/target matching but *not* by naive substring matching.

Text config of note: 64 layers, `full_attention_interval: 4`,
`hidden_size` 5120, `intermediate_size` 17408, `vocab_size` 248320,
`tie_word_embeddings: false`, `max_position_embeddings` 262144, plus a vision
stack and an MTP head.

### 3.3 Schema oracle captured

From `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` @ `cfd1460322b9d8367a4ab13564a2182d50852d60`
(public, unauthenticated fetch, `config.json` only — **no weights copied**):

```text
quant_method         : compressed-tensors
quantization_status  : compressed
format               : nvfp4-pack-quantized
producer             : {"name": "qatfactory", "version": "0.1.0"}
targets              : ["Linear"]

weights:
  num_bits 4, type float, strategy tensor_group, group_size 16,
  symmetric true, scale_dtype torch.float8_e4m3fn,
  dynamic false, observer memoryless_minmax

input_activations:
  num_bits 4, type float, strategy tensor_group, group_size 16,
  symmetric true, scale_dtype torch.float8_e4m3fn,
  dynamic "local", observer static_minmax

ignore (6): lm_head
            re:.*visual.*
            re:.*mtp.*
            re:.*embed_vision.*
            re:.*embed_audio.*
            re:.*vision_embedder.*
```

Two consequences worth recording:

- `lm_head` is typically a `Linear`, so it **must** be explicitly ignored.
  `embed_tokens` is an `nn.Embedding` and is excluded implicitly by
  `targets: ["Linear"]`. Encoding the ignore set as naive substrings would get
  this backwards.
- Handoff §27's predicted schema matches the real checkpoint exactly, including
  the asymmetric `dynamic` flags (`false` for weights, `"local"` for
  activations). Export config tests can assert against this captured JSON.

---

## 4. Compute decision: Nebius H200, not GCP

### 4.1 Why not GCP

GCP is viable but 2–3× more expensive for identical silicon, and requires an
8×H100-class quota request, which is a schedule risk independent of price.

### 4.2 Verified Nebius pricing (fetched 2026-09-15)

Per GPU-hour, from the Nebius pricing page:

| Instance | Preemptible | On-demand |
|---|---|---|
| NVIDIA HGX H200 | **$2.45** | **$4.50** |
| NVIDIA HGX H100 | $2.15 | $3.85 |
| NVIDIA HGX B200 | $3.95 | $7.15 |
| NVIDIA HGX B300 | $4.30 | $7.85 |

Storage: object storage $0.0147/GiB/month, egress $0.0150/GiB.
A 200 GiB bucket plus 60 GiB egress ≈ **$3.80/month**.

### 4.3 Node rate comparison, 8-GPU ($/hour)

| | Nebius on-demand | Nebius preemptible | GCP list | GCP DWS Flex-start | GCP spot |
|---|---|---|---|---|---|
| 8× H100 | **30.80** | **17.20** | 88.49 | 38.32 | 52.96 |
| 8× H200 | **36.00** | **19.60** | 84.81 | 42.40 | 50.87 |
| 8× B200 | **57.20** | **31.60** | n/a | 64.44 | 39.63 |

Nebius H200 on-demand is **2.36× cheaper than GCP list**, 1.18× cheaper than
GCP's cheapest DWS Flex-start, and 1.41× cheaper than GCP spot. Nebius
preemptible H200 ($19.60) is **2.6× cheaper than GCP spot** ($50.87) for
identical hardware.

### 4.4 Selected target

```text
provider    : Nebius AI Cloud
region      : us-central1   (fallbacks: eu-north1, eu-west1)
platform    : gpu-h200-sxm   (Intel Xeon Platinum 8468, Intel Sapphire Rapids)
preset      : 8gpu-128vcpu-1600gb
GPUs        : 8 × H200 SXM, 141 GB HBM3e each = 1128 GB total
interconnect: 4th-gen NVLink 900 GB/s GPU-GPU, 400 Gbps InfiniBand
              (8× ConnectX-7), BlueField-3 200 Gbps
```

**First choice confirmed by the vendor's own docs.** All platforms with GPUs
support preemptible VMs on Nebius, so the cheap tier is available for this
preset.

### 4.5 Why H200 over H100

| | total HBM | vs. bf16 27B weights (54 GB) |
|---|---|---|
| 8× H100 | 640 GB | 11.9× |
| **8× H200** | **1128 GB** | **20.9×** |
| 8× B200 | 1440 GB | 26.7× |

A full-parameter QAD run needs roughly: student bf16 params 54 GB + gradients
54 GB + AdamW fp32 states ~216 GB + frozen bf16 teacher 54 GB ≈ **380 GB**,
before activations. On 8×H100 that leaves ~260 GB for activations and we would
be reaching for CPU offload, aggressive recomputation, or approximate KL on day
one. On 8×H200 the first correct run fits without any of those crutches.

For a from-scratch implementation, **every offload shortcut forced by memory
pressure is a place a bug can hide** while we are still establishing the fault
boundary. H200 costs 17% more than H100 on-demand and buys 76% more VRAM. That
is the right trade.

---

## 5. Staged spend ladder

The governing principle: **no stage is authorized until the previous stage's
exit criteria are met and recorded.** Cost is a gate, not an afterthought.

```text
RUNG 0 ── $0 ──────── implementation + unit tests proven, locally
   │
   ▼ exit: all unit tests pass, smoke QAD runs, export round-trips
RUNG 1 ── $0–50 ───── optional: datacenter-silicon validation
   │
   ▼ exit: vLLM loads an NVFP4 checkpoint, teacher-data pilot produced
RUNG 2 ── $72–108 ─── 8×H200 bring-up + 50 real steps
   │
   ▼ exit: s/step measured, peak HBM recorded, ckpt save/resume works
RUNG 3 ── decision ─── extrapolate → authorize full run
   │
   ▼ exit: full QAD complete, export validated
RUNG 4 ── $2–20 ───── NInfer conversion + serving + MCC smoke
```

### Rung 0 — local, $0

Everything that can be developed and proven on the workstation. Costs nothing
but time, and it is a prerequisite for every later rung.

Work items, in handoff §53 order:

1. **Bootstrap** — repo, `uv.lock`, `.python-version`, layout, LICENSE. ✅ *(this turn)*
2. **Pin provenance** — `scripts/bootstrap_sources.sh` resolving immutable HF
   revisions and NInfer git SHA into `provenance.lock.json`; download
   EfficientThink BF16 and the QUASAR schema oracle.
3. **QUASAR primitives** — E2M1 codebook, saliency-weighted least-squares group
   scale, candidate search over 0.30→1.00 step 0.05, STE, dynamic-local A4
   activation fake quantization.
4. **Tensor selection** — patcher targeting exactly 496, with the count assertion
   from §3.2; non-target modules frozen.
5. **Export** — compressed-tensors NVFP4 writer matching the §3.3 schema.
6. **Verification** — simulated-vs-materialized fidelity comparison.
7. **Smoke QAD** — small Qwen-family model, 10–100 steps, proving: patching
   works, latent weights receive gradients, non-target modules stay frozen,
   AdamW `exp_avg_sq` populates, candidate search runs, selection changes
   sensibly, KL stays finite, checkpoint save/resume works, export works.
8. **Probe harness** — `configs/probe-cost.toml` plus the `metrics.jsonl` schema
   so rung 2's measurement is a scripted artifact rather than a manual reading.

**Exit criteria:** `pytest` green, `ruff check` + `ruff format --check` clean,
smoke QAD completed, export round-trips, probe harness dry-run against the local
GPU.

### Rung 1 — optional, $0–50

Only worth spending if rung 0 leaves doubt about datacenter-silicon behavior.

| Purpose | Hardware | Duration | Cost |
|---|---|---|---|
| Teacher-data pilot (subset) | 1× H200 `1gpu-16vcpu-200gb` @ $4.50/h | ~6–10 h | $27–45 |
| §28 vLLM NVFP4 loadability on non-laptop silicon | same | ~1–2 h | $5–9 |

**Try vLLM locally first** — a 27B W4A4 model is ~15–17 GB and fits the 25 GB
5090. Rung 1 exists only as a fallback if sm_120 vLLM cannot load NVFP4.

**Exit criteria:** either vLLM loads and generates correctly (locally or on
rented silicon), or the failure is diagnosed and attributable to a specific
component rather than to "the model".

### Rung 2 — 8×H200 bring-up + probe, $72–108

The most valuable three hours in the entire project. It converts `seconds/step`
from an unknown into a measurement.

Scope:

1. Provision `gpu-h200-sxm` / `8gpu-128vcpu-1600gb` in us-central1.
2. Fetch EfficientThink BF16 + Open-PerfectBlend teacher sequences.
3. Bring up FSDP2 with `torchrun`; full-shard student params, shard optimizer
   states, activation checkpointing, frozen teacher sharded/offloaded.
4. Run **50 real optimizer steps** and record the probe metrics.
5. Verify checkpoint write + deterministic resume.

**The probe must run at the real target configuration**, or the extrapolation is
worthless:

```text
sequence_length 4096   global_batch_size 32
full QUASAR candidate search ENABLED
no CPU offload   no reduced precision   full model
```

Required probe outputs:

```text
seconds/step                    (primary)
seconds/step with search OFF    (isolates QUASAR-specific overhead)
peak HBM per rank               (proves no offload engaged)
MFU, tokens/s
KL finiteness, top-1 agreement on a fixed batch
exp_avg_sq non-degeneracy
candidate-factor histogram
FP8 scale saturation statistics
checkpoint write time + resume time   (amortized into every long run)
data-loader throughput
```

If offload activates, the measured step time is **not representative** and we
fix that before extrapolating. A probe that silently falls back to offload is
worse than no probe.

**Exit criteria:** all probe outputs recorded in `runs/<run-id>/metrics.jsonl`;
`summary.json` written; peak HBM per rank consistent with a non-offloaded run;
s/step stable over the last ~20 steps (not still warming up).

### Rung 3 — authorization gate

Authorization is a **rule applied to a measurement**, not a judgment call made
at spend time. Full run = 2446 steps, at $36/h on-demand or $19.60/h preemptible.

| measured s/step | full-run hours | on-demand | preemptible | verdict |
|---|---|---|---|---|
| 10 | 6.8 | $245 | $133 | mint |
| **12** | **8.2** | **$293** | **$160** | expected |
| 14 | 9.5 | $342 | $186 | fine |
| 15 | 10.2 | $367 | $200 | fine |
| 20 | 13.6 | $489 | $266 | approve with cap |
| 25 | 17.0 | $612 | $333 | investigate first |
| ≥30 | ≥20.4 | ≥$734 | ≥$400 | **abort** |

A step time ≥30 s implies roughly 10% MFU, which means something is wrong —
offload thrashing, bad sharding, or activation checkpointing not applied. Do not
buy the remaining steps to find out.

Budget the full run at the extrapolation **plus 25–30% overhead** for restarts,
preemption, and evaluation passes, and authorize that as a hard cap with
automatic stop.

### Rung 4 — NInfer + MCC, $2–20

Conversion and serving are CPU/GPU-light. This rung is where the NInfer-side
branch work lands (§29–38 of the handoff).

---

## 6. Budget envelope

| Tier | Contents | Cost |
|---|---|---|
| **Lean** | rung 0 only, then one clean QAD run on-demand | $250–450 |
| **Recommended** | rung 0 + rung 2 probe + full run + retries | **$450–900** |
| **Comfortable** | adds §51 mixed-precision ablation, some preemptible | $700–1400 |
| **Worst realistic** | two full attempts on-demand, aborts, B200 fallback | $1500–2200 |

Plus ~$5–20 object storage and egress.

**Proposed ceiling: $900**, with an automatic hard stop if the rung 2 probe
measures s/step ≥ 30.

For scale: the identical realistic plan on GCP `a3-ultragpu-8g` at DWS
Flex-start would cost **$1000–2100** — roughly 2.4× more.

**Abort-early property:** the ladder is designed so that the worst case of a
bad probe is **~$200 total**, not a failed $1000 run. That is the entire point
of staging it this way.

---

## 7. Non-negotiable invariants

Carried forward from the handoff. Violating any of these invalidates the run.

1. **Source is BF16, not GGUF.** Use the `BF16/` subdir of the EfficientThink
   repo. The GGUF sibling is reference-only.
2. **Teacher = student initialization = EfficientThink BF16.** Never
   `teacher = vanilla Qwen3.8`. The objective is to preserve EfficientThink
   behavior under NVFP4, not to reconstruct vanilla behavior.
3. **Baseline data is Open-PerfectBlend only.** No MCC histories. MCC/domain
   data is a second experiment with its own config name.
4. **Exact forward KL**, response-token positions only, prompt kept as context,
   padding masked. Do not silently substitute top-k or approximate KL to fit
   memory. If exact KL will not fit the hardware, say so.
5. **The FP8-rounded group scale participates in candidate scoring.** Do not
   select a candidate using the ideal floating-point scale and round afterwards.
6. **Saliency is AdamW `exp_avg_sq`**, not its square root. The first-step
   bootstrap (zero/unavailable `exp_avg_sq` → ones) is a *local implementation
   choice* and must be labelled as such in config and provenance.
7. **Non-quantized text tensors come from EfficientThink**, not from vanilla
   Qwen. Equality must be *proven* before any substitution, for embedding,
   lm_head, norms, GDN control tensors, MTP, and the vision stack.
8. **No false completion claims.** If the real 27B run has not happened, say so.
   Never substitute PTQ and call it QAT.
9. **Every unspecified value is an explicit, labelled local choice** — in config,
   in provenance, tagged `local_choice` vs `public_reference`.
10. **Speculation off for all first comparisons.** MTP and DFlash2 are measured
    separately, after target-model parity is established.

---

## 8. Risk register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | us-central1 capacity for `8gpu-128vcpu-1600gb` unavailable at schedule time | delays rung 2 | fall back to eu-north1 / eu-west1 (same preset listed); verify before scheduling |
| R2 | No Nebius account/CLI/credentials on the workstation | blocks rung 2 | owner action; `nebius` CLI absent, no config found |
| R3 | sm_120 vLLM cannot load NVFP4 locally | blocks §28 gate | rung 1 rents H200 for a 1–2 h answer; costs ~$9 |
| R4 | QUASAR candidate search adds more wall-clock than predicted (memory-bound, not FLOP-bound) | raises full-run cost | probe measures search-on vs search-off explicitly |
| R5 | 27B FSDP2 lands at low MFU (20% rather than 30–35%) | up to +50% cost | decision table catches it at rung 3; abort threshold at 30 s/step |
| R6 | Export/vLLM fidelity mismatch (simulated ≠ materialized) | invalidates training | §28 mandatory structural + fidelity comparison before NInfer conversion |
| R7 | NInfer GDN `in_proj_a`/`in_proj_b` bypass native W4A4 semantics | long-run quality drift | confirmed present at `recipe_nvfp4_quasar.py` L138/L141, L468–510; fix before trusting long MCC results |
| R8 | QUASAR has no public reference implementation | reimplementation risk | independent implementation from the paper is permitted and expected; pin the arXiv revision; re-check for official code before finalizing |
| R9 | Preemption during the long run | lost progress | deterministic resume is mandatory from rung 0; probe proves it works |
| R10 | Quality regression that only appears at full schedule length | wasted spend | 600-step tripwire (below), plus §40 B-vs-C PTQ comparison |

### Quality tripwire (not a verdict)

At ~600 steps, compare held-out KL and teacher top-1 agreement against a
plain-STE PTQ control. Cost ≈ 2 h ≈ $72.

If QUASAR is *behind* PTQ on reconstruction error at 600 steps, stop and debug
rather than buy the remaining 1846 steps.

**Caveat:** QUASAR may need the full schedule to demonstrate its gain, so this
is a sanity tripwire, not a quality verdict. The authoritative quality answer is
the §40 experiment matrix, in particular **B vs C** (existing EfficientThink PTQ
W4A4 vs QUASAR QAT W4A4).

---

## 9. Fault boundary

Strict ordering for interpreting failures (handoff §49):

```text
1. EfficientThink BF16 teacher
        ↓
2. simulated QUASAR student
        ↓
3. exported compressed-tensors NVFP4
        ↓
4. vLLM native NVFP4
        ↓
5. NInfer .ninfer conversion
        ↓
6. NInfer target runtime
        ↓
7. MCC agent loop
```

```text
2 != 3   → exporter / materialization bug
3 ok, 4 fails → compressed-tensors / vLLM compatibility
4 ok, 5/6 diverge → NInfer converter / runtime
1–6 ok, MCC fails → agent behavior, context, or tool loop
```

**Never diagnose training, export, runtime, and MCC simultaneously.**

---

## 10. Provenance to pin

`provenance.lock.json` must record immutable revisions for:

| Source | Repo | Revision |
|---|---|---|
| EfficientThink BF16 | `nerkyor/Qwen3.8-27B-EfficientThink-…-SFT-SimPO-DFlash2` | `882e1af3b8f5844f3b3163b132e343c06e91908f` |
| QUASAR schema oracle | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | `cfd1460322b9d8367a4ab13564a2182d50852d60` |
| QAD dataset | `mlabonne/open-perfectblend` | `af60f3c18201652a83a93f46fcfee1b646ba3df7` |
| NInfer | `carlopires/ninfer-rtx5090-mobile` | baseline `895de784034d5a28ee6a65d92dec3774b69edffd` |
| QUASAR paper | arXiv 2608.13966 | pin retrieval date + version |

No formal experiment runs against unpinned moving revisions.

Note: the local NInfer checkout is `~/code/ninfer` with the upstream remote
named **`carlo`**, not `origin` (plus `mirko`, `upstream`). The handoff's
`gh repo clone` / `git checkout master` sequence does not apply verbatim and
must be adapted rather than creating a second clone.

---

## 11. Open questions / owner actions

| # | Question | Owner | Blocks |
|---|---|---|---|
| Q1 | Nebius account + API credentials provisioned | owner | rung 2 |
| Q2 | `8gpu-128vcpu-1600gb` availability confirmed in us-central1 | owner | rung 2 |
| Q3 | Budget ceiling ratified ($900 proposed) | owner | rung 3 |
| Q4 | On-demand for probe, preemptible for full run — confirm | owner | rung 2/3 |
| Q5 | Is there a remote GPU host among `testa`/`testb`/`testx`/`scratch` that changes rung 1? | owner | rung 1 (optional) |
| Q6 | Confirm no official QUASAR code has appeared before finalizing the implementation | impl | rung 0 exit |

---

## 12. Immediate next actions (rung 0 continuation)

1. `scripts/bootstrap_sources.sh` + `provenance.lock.json` — pin and download.
2. `src/qwen38_quasar/quantization/e2m1.py` + `tests/test_e2m1.py`.
3. `quantization/quasar.py` + `tests/test_nvfp4_reconstruction.py`,
   `tests/test_quasar_search.py`.
4. `quantization/ste.py` + `tests/test_ste.py`.
5. `quantization/activation.py` + `tests/test_activation_quant.py`.
6. `models/patch_linear.py` + `tests/test_tensor_selection.py` (assert 496).
7. `export/compressed_tensors.py` + `tests/test_export_config.py`
   (assert against the §3.3 oracle).
8. Smoke QAD config + `scripts/train_smoke.sh`.
9. Probe harness: `configs/probe-cost.toml`, `metrics.jsonl` schema.

---

## 13. Definition of done — milestone 1

Gates A–I are enumerated in §48 of the handoff and are adopted verbatim. Gate E
is the one that governs honesty about compute:

```text
[ ] full 27B QAD run completed
```

or, if compute is unavailable:

```text
[ ] small-model smoke run completed
[ ] full 27B launch command documented
[ ] implementation is runnable
[ ] no false claim that 27B QAD completed
```

This plan's rung 0/1/2 work is designed so that the first branch is reachable
deliberately and affordably, and the second branch is a fully defensible
outcome if the budget is not approved.
