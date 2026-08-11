# Building on a 2019 MacBook Pro / AMD Radeon Pro 5500M (4 GB)

Self-contained recipe for running LocalAI's OpenAI-compatible API with the model resident in a **discrete AMD GPU's VRAM** on an Intel Mac, via ggml's **Vulkan backend over MoltenVK**.

**No source changes to LocalAI are required.** This file is the whole payload of the branch: LocalAI already supports `BUILD_TYPE=vulkan` and external gRPC backends. What it does not do is target this hardware — its macOS build path is Apple-Silicon-and-Metal oriented, and on a discrete AMD GPU Metal is the broken backend, not the fast one. The steps below compose the pieces that upstream leaves unconnected, and document the two flags that silently produce the wrong result if you omit them.

Target: **MacBook Pro 16" 2019 · AMD Radeon Pro 5500M (4 GB, RDNA1) · Intel UHD 630 (present, excluded) · macOS 14.x.**

## Why Vulkan, not Metal

ggml's Metal backend re-reads model weights over PCIe on every token when the GPU is discrete rather than unified-memory — roughly 0.8 tok/s. Vulkan/MoltenVK keeps the weights resident. Every build step below disables Metal explicitly.

## Requires MoltenVK >= 1.4.2

Up to 1.4.1, MoltenVK reported `subgroupSize 64` on AMD Macs while Metal runs 32-wide SIMD groups, so every subgroup reduction in every shader spanned the wrong lane count — the root cause behind the matmul corruption and the NaN in the hybrid-model shaders. `brew list --versions molten-vk` must report >= 1.4.2. The llama.cpp fixes gate their workarounds on `driverVersion`, so older drivers still produce correct output, but the flash-attention and quantized-KV behaviour documented below assumes 1.4.2.

## Dependencies (Homebrew)

```sh
brew install go node cmake grpc protobuf abseil \
  vulkan-headers vulkan-loader molten-vk glslang shaderc spirv-headers libomp
```

`go` >= 1.26 and Node/npm are for the LocalAI core; `grpc`/`protobuf`/`abseil` are what the C++ backend links against (LocalAI can also build gRPC from source via `BUILD_GRPC_FOR_BACKEND_LLAMA=1` — much slower, only needed if Homebrew's protobuf disagrees with the checked-in `backend.proto` codegen; Homebrew gRPC 1.83 / protobuf 35.1 worked here).

## 1. Build the LocalAI core

```sh
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/LocalAI.git
cd LocalAI
make build          # protogen + React UI + go build -> ./local-ai
./local-ai --version
```

## 2. Compose the backend's llama.cpp source

LocalAI's llama backend is a **separate gRPC server process** built from a llama.cpp checkout, with LocalAI's `grpc-server.cpp` grafted into `llama.cpp/tools/`. Two things have to be true of that checkout at once:

1. it must carry the **Vulkan fixes** for this GPU — subgroup matmul, the NaN-producing `SSM_SCAN`/`GATED_DELTA_NET` shaders, the q8_1 activation quantizer, the q8_0 vector-matmul guard, plus the two performance changes (emulated integer dot, and flash attention actually running on the GPU) — and
2. it must be the **commit LocalAI pins** — `grpc-server.cpp` tracks llama.cpp's internal server API, which moves within days.

So don't build from the llama.cpp fork's branch tip. Check out the pinned commit and cherry-pick the fix commits onto it:

```sh
PIN="$(sed -n 's/^LLAMA_VERSION?=//p' backend/cpp/llama-cpp/Makefile)"

git clone https://github.com/maximosipov/llama.cpp.git backend/cpp/llama-cpp/llama.cpp
cd backend/cpp/llama-cpp/llama.cpp
# Fetch the pinned commit explicitly by SHA. It is a plain master commit, not a tag,
# and it is not necessarily an ancestor of the fork's branch, so a fork-only clone
# will not have it.
git fetch https://github.com/ggml-org/llama.cpp.git "$PIN"
git checkout -b build "$PIN"
# The fix commits from the fork's macbook-pro-2019-radeon-5500m-4gb branch, oldest
# first. Explicit SHAs: upstream llama.cpp history has its own unrelated "vulkan: fix ..."
# commits, so a message grep is not safe here.
git cherry-pick 7847fca13 595936bcd a91f29c5f \
                acb45b872 8cbd19056 8e8d7fb26 16ea5d8e7 e629266a1 2707498b9
git submodule update --init --recursive --depth 1
cd ../../../..
```

| Commit | What it does |
|---|---|
| `7847fca13` | matmul subgroup correctness on MoltenVK (RDNA1 + older Intel iGPUs) |
| `595936bcd` | NaN `SSM_SCAN` / `GATED_DELTA_NET` — hybrid/state-space models (Granite-4.0-H, Qwen3.5) |
| `a91f29c5f` | diagnostic env knobs (opt-in, no default change) |
| `acb45b872` | disable subgroup clustered ops — fixes the q8_1 activation quantizer |
| `8cbd19056` | enable emulated integer dot for matmul: **+68% prefill** |
| `8e8d7fb26` | flash-attention subgroup diagnostic knob (superseded by the next one) |
| `16ea5d8e7` | **run flash attention on the GPU** — it was silently falling back to CPU |
| `e629266a1` | guard the broken q8_0 `mul_mat_vecq` path |
| `2707498b9` | trust MoltenVK subgroups from 1.4.2 onwards |

Verified: all nine cherry-pick cleanly onto `1cbfd1988`, and `prepare.sh` still grafts on top. If a future `LLAMA_VERSION` bump makes them stop applying, the changes are narrow (`ggml-vulkan.cpp` plus three `.comp` shaders) — diff them off the branch and re-apply.

Then graft in LocalAI's gRPC server. **Run this once**: `prepare.sh` applies LocalAI's own patches to the llama.cpp tree and is not idempotent.

```sh
cd backend/cpp/llama-cpp
mkdir -p llama.cpp/tools/grpc-server
bash prepare.sh
cd ../../..
```

## 3. Build the Vulkan backend

```sh
P="$(brew --prefix)"
export BUILD_TYPE=vulkan
export CMAKE_ARGS="-DGGML_METAL=OFF \
  -DVulkan_INCLUDE_DIR=$P/opt/vulkan-headers/include \
  -DVulkan_LIBRARY=$P/opt/vulkan-loader/lib/libvulkan.dylib \
  -DVulkan_GLSLC_EXECUTABLE=$P/opt/shaderc/bin/glslc \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE=$P/opt/glslang/bin/glslangValidator \
  -DOpenMP_ROOT=$P/opt/libomp \
  -DCMAKE_CXX_FLAGS=-I$P/opt/spirv-headers/include \
  -DCMAKE_BUILD_TYPE=Release"

make -C backend/cpp/llama-cpp grpc-server
```

Two non-obvious details, both of which produce a build that *looks* fine and behaves wrongly:

- **`CMAKE_ARGS` must be exported, not passed on the make command line.** `backend/cpp/llama-cpp/Makefile` does `CMAKE_ARGS?=` and then several `CMAKE_ARGS+=`. A command-line variable assignment overrides those appends, silently dropping `-DGGML_VULKAN=1` itself. An environment variable seeds `?=` and keeps the appends.
- **`-DGGML_METAL=OFF` is required.** In that Makefile, `BUILD_TYPE=vulkan` and the `Darwin` case are arms of a single if/else chain — so on macOS, choosing vulkan means the Darwin arm (the one that would have turned Metal off) never runs. ggml auto-enables Metal on macOS, and Metal then wins device 0 at runtime.

Gate:

```sh
otool -L backend/cpp/llama-cpp/grpc-server | grep -i vulkan     # expect libvulkan
otool -L backend/cpp/llama-cpp/grpc-server | grep -i Metal      # expect no output
```

## 4. Register the backend and serve

LocalAI launches an external backend as a child process, so the MoltenVK environment has to be on the child. Wrap the binary:

```sh
cat > backend/cpp/llama-cpp/run-vulkan.sh <<'EOF'
#!/usr/bin/env bash
P="$(brew --prefix)"
export VK_ICD_FILENAMES="$P/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$P/opt/molten-vk/lib:$P/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
# Pin the discrete GPU. Load-bearing on a Mac with an integrated GPU as well: the Intel
# UHD 630 advertises 32 GiB of shared system memory against the Radeon's real 4 GiB, so
# any "pick the biggest GPU" heuristic picks it -- and its matmul is numerically wrong
# under MoltenVK.
export GGML_VK_VISIBLE_DEVICES=0
exec "$(dirname "$0")/grpc-server" "$@"
EOF
chmod +x backend/cpp/llama-cpp/run-vulkan.sh
```

Model YAML. **The GGUF must live inside `--models-path`** (LocalAI rejects absolute paths outside it as invalid) — a symlink is fine:

```sh
mkdir -p models
ln -s /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf models/

cat > models/qwen3-4b.yaml <<'EOF'
name: qwen3-4b
backend: llama-cpp
parameters:
  model: Qwen3-4B-Instruct-2507-Q4_K_M.gguf
context_size: 4096
f16: true
gpu_layers: 99
# Flash attention runs on the GPU with commit 16ea5d8e7 and is the faster option for
# decode (~+5%). Earlier revisions of this file said "false, ~2.3x slower" -- that
# measurement was real but the cause was not a slow kernel: FLASH_ATTN_EXT was
# unsupported on Vulkan for AMD Macs, so every attention op ran on the CPU.
flash_attention: "true"
EOF

./local-ai run --address 127.0.0.1:8080 --models-path ./models \
  --external-grpc-backends "llama-cpp:$PWD/backend/cpp/llama-cpp/run-vulkan.sh"
```

## Verify

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-4b",
  "messages": [{"role":"user","content":"How much is 20+34? Answer with just the number."}]
}'
```

Expect a coherent `54`. Garbled tokens mean the wrong device was selected — recheck `GGML_VK_VISIBLE_DEVICES`.

LocalAI's own hardware probe reports `GPU vendor=""` and `Total available VRAM 0` on macOS, so **its logs will not tell you whether the model reached the GPU**, and it does not forward the backend's ggml banner. Ask IOKit instead:

```sh
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

A 4B Q4_K_M model resident on the GPU shows ~3.1–3.5 GiB in use on the AMD accelerator node (measured: 3.29 GB with `flash_attention: "true"`, ctx 4096, f16 KV). If it stays near idle while generation runs, the model is on the CPU — check that `gpu_layers` survived into the load (LocalAI's macOS hardware auto-tuning can zero it; `LOCALAI_DISABLE_HARDWARE_DEFAULTS=true` turns that off).

Sustained serving throughput on this stack, 128-token completions against a warm server: **~28–31 tok/s** for Qwen3-4B-Instruct-2507 Q4_K_M. That is the most reproducible measurement on this machine — much tighter than `llama-bench`, whose numbers swing ~2.4x with GPU clock state alone.

A thinking model (Qwen3, etc.) returns its chain of thought in `choices[].message.reasoning` and leaves `content` empty until it closes the think block — an empty `content` with `finish_reason: "length"` is a token budget problem, not a correctness one.

## 5. Text-to-image on the same GPU

LocalAI's `stablediffusion-ggml` backend wraps stable-diffusion.cpp, so image generation reuses this Vulkan stack. **No source changes are needed here either** — the flag this hardware cannot do without is reachable from the model config.

The build repeats the trap from step 3, one directory over: in `backend/go/stablediffusion-ggml/Makefile`, `BUILD_TYPE=vulkan` and the `OS=Darwin` case are arms of one if/else chain, so selecting vulkan on macOS skips the Darwin arm that would have set `GGML_METAL=OFF`. Pass it explicitly, through the environment:

```sh
export BUILD_TYPE=vulkan
export CMAKE_ARGS="-DGGML_METAL=OFF \
  -DVulkan_INCLUDE_DIR=$(brew --prefix)/opt/vulkan-headers/include \
  -DVulkan_LIBRARY=$(brew --prefix)/opt/vulkan-loader/lib/libvulkan.dylib \
  -DVulkan_GLSLC_EXECUTABLE=$(brew --prefix)/opt/shaderc/bin/glslc \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE=$(brew --prefix)/opt/glslang/bin/glslangValidator \
  -DOpenMP_ROOT=$(brew --prefix)/opt/libomp \
  -DCMAKE_CXX_FLAGS=-I$(brew --prefix)/opt/spirv-headers/include \
  -DCMAKE_BUILD_TYPE=Release"
make -C backend/go/stablediffusion-ggml JOBS="$(sysctl -n hw.ncpu)" stablediffusion-ggml
```

Gate: `otool -L backend/go/stablediffusion-ggml/libgosd-fallback.so` lists libvulkan and **no** `Metal.framework`.

The backend is a child process like the llama.cpp one, so it needs its own wrapper carrying the MoltenVK environment. It resolves the library it dlopens through `SD_LIBRARY`:

```sh
cat > backend/go/stablediffusion-ggml/run-vulkan.sh <<'WRAP'
#!/usr/bin/env bash
P="$(brew --prefix)"
D="$(cd "$(dirname "$0")" && pwd)"
export VK_ICD_FILENAMES="$P/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$D/lib:$P/opt/molten-vk/lib:$P/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export GGML_VK_VISIBLE_DEVICES=0
export SD_LIBRARY="$D/libgosd-fallback.so"
exec "$D/stablediffusion-ggml" "$@"
WRAP
chmod +x backend/go/stablediffusion-ggml/run-vulkan.sh
```

`diffusion_conv_direct` is **mandatory on RDNA1/MoltenVK**. The default GPU convolution path (im2col + matmul) is numerically broken on this card and produces full-frame colourful noise; conv-direct is both correct and ~3× faster. It is a per-model option:

```sh
cat > models/sd-turbo.yaml <<'EOF2'
name: sd-turbo
backend: stablediffusion-ggml
parameters:
  model: sd_turbo.safetensors
step: 4
cfg_scale: 1
options:
- "sampler:euler"
- "diffusion_conv_direct:true"
- "wtype:q8_0"
EOF2
```

Register both backends and serve:

```sh
./local-ai run --address 127.0.0.1:8080 --models-path ./models \
  --external-grpc-backends "llama-cpp:$PWD/backend/cpp/llama-cpp/run-vulkan.sh,stablediffusion-ggml:$PWD/backend/go/stablediffusion-ggml/run-vulkan.sh"

curl -s http://127.0.0.1:8080/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "sd-turbo",
  "prompt": "a red apple on a wooden table, studio lighting",
  "size": "512x512",
  "n": 1
}'
```

Open the returned URL. A recognisable apple means the whole path is correct; full-frame colourful noise means `diffusion_conv_direct` did not reach the backend. Measured on this box: SD-Turbo q8_0 at 4 steps, 512×512, ~2 GB resident and ~3.0 GB peak (peak is at VAE decode, not sampling), first call ~50 s including model load and on-the-fly quantization. SD 1.5 fp16 at 20 steps also works; SDXL-Turbo needs `"keep_vae_on_cpu:true"` and sits at 3.8 GB, which is the edge of a 4 GB card.

## 6. Recommended models for this card

| Role | Model | Why |
|---|---|---|
| Text, default | **Granite-4.0-H-Micro** Q4_K_M | Best tool calling measured here including negative cases (24/24 on a local set, 0.863 BFCL AST). Mamba-2 hybrid, so context growth costs almost no VRAM — the property that matters for a loop that accumulates history. ~2.1 GB headroom, served at **32 tok/s**. |
| Text, alternative | Qwen3-4B-Instruct-2507 Q4_K_M | Marginally better single-turn function-calling score and faster on the fixed stack (40 tg128 in `llama-bench`); dense, so the KV cache grows fastest in the set — bound your contexts or quantize the cache. |
| Image, default | **SD-Turbo** `wtype:q8_0` | 2 GB, 1–4 steps, comfortable headroom. |
| Image, ecosystem | SD 1.5 fp16 | 2 GB, 20 steps, the largest LoRA/ControlNet library. |

Granite requires the `SSM_SCAN`/`GATED_DELTA_NET` shader fix from step 2 — without it, hybrid models emit token salad on the GPU while staying coherent on CPU.

An earlier revision credited Granite as "the only model that finishes a 12–19K-token prompt". That was an artifact of a 400 s client timeout in the bench scripts, not a property of the models: re-measured on the fixed stack, an 8192-token prefill completes in 4–6 minutes on Qwen3-4B-2507 and neither loses the device. Pick Granite for the recurrent state and the tool calling, not for being the only survivor.

## 7. Packaging a DMG

LocalAI has no upstream desktop app, so the bundle is a launcher app: it exports the MoltenVK environment and the device pin, writes the backend wrapper scripts, starts the server, opens the WebUI, and stops the server when its dialog is dismissed.

The part that takes the work is making the bundle self-contained. `grpc-server` links **107 Homebrew dylibs** (grpc, protobuf, abseil, re2, c-ares, openssl@3, libomp, vulkan-loader). The packaging script walks that closure with `otool`, copies each library into `Contents/Frameworks/`, rewrites every install name to `@rpath`, and adds the matching rpaths; the ICD json inside the bundle points at `../../Frameworks/libMoltenVK.dylib` so the app can be moved anywhere. The acceptance test is that no `/usr/local/opt` path survives anywhere in the bundle.

Weights are deliberately **not** bundled — they are gigabytes. The launcher seeds the model YAMLs into `~/Library/Application Support/LocalAI/models` on first run and symlinks weights from a development checkout when one is present.

Result on this machine: a 367 MB `.app`, a 143 MB compressed DMG, ad-hoc signed (no Apple identity here, so it is not notarized — clear the quarantine bit on first launch with `xattr -dr com.apple.quarantine "/Applications/LocalAI M.app"`).
