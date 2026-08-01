# Building on a 2019 MacBook Pro / AMD Radeon Pro 5500M (4 GB)

Self-contained recipe for running LocalAI's OpenAI-compatible API with the model resident
in a **discrete AMD GPU's VRAM** on an Intel Mac, via ggml's **Vulkan backend over
MoltenVK**.

**No source changes to LocalAI are required.** This file is the whole payload of the
branch: LocalAI already supports `BUILD_TYPE=vulkan` and external gRPC backends. What it
does not do is target this hardware — its macOS build path is Apple-Silicon-and-Metal
oriented, and on a discrete AMD GPU Metal is the broken backend, not the fast one. The
steps below compose the pieces that upstream leaves unconnected, and document the two
flags that silently produce the wrong result if you omit them.

Target: **MacBook Pro 16" 2019 · AMD Radeon Pro 5500M (4 GB, RDNA1) · Intel UHD 630
(present, excluded) · macOS 14.x.**

## Why Vulkan, not Metal

ggml's Metal backend re-reads model weights over PCIe on every token when the GPU is
discrete rather than unified-memory — roughly 0.8 tok/s. Vulkan/MoltenVK keeps the weights
resident. Every build step below disables Metal explicitly.

## Dependencies (Homebrew)

```sh
brew install go node cmake grpc protobuf abseil \
  vulkan-headers vulkan-loader molten-vk glslang shaderc spirv-headers libomp
```

`go` >= 1.26 and Node/npm are for the LocalAI core; `grpc`/`protobuf`/`abseil` are what
the C++ backend links against (LocalAI can also build gRPC from source via
`BUILD_GRPC_FOR_BACKEND_LLAMA=1` — much slower, only needed if Homebrew's protobuf
disagrees with the checked-in `backend.proto` codegen; Homebrew gRPC 1.83 / protobuf 35.1
worked here).

## 1. Build the LocalAI core

```sh
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/LocalAI.git
cd LocalAI
make build          # protogen + React UI + go build -> ./local-ai
./local-ai --version
```

## 2. Compose the backend's llama.cpp source

LocalAI's llama backend is a **separate gRPC server process** built from a llama.cpp
checkout, with LocalAI's `grpc-server.cpp` grafted into `llama.cpp/tools/`. Two things
have to be true of that checkout at once:

1. it must carry the **Vulkan correctness fixes** for this GPU (subgroup matmul, and the
   NaN-producing `SSM_SCAN`/`GATED_DELTA_NET` shaders), and
2. it must be the **commit LocalAI pins** — `grpc-server.cpp` tracks llama.cpp's internal
   server API, which moves within days.

So don't build from the llama.cpp fork's branch tip. Check out the pinned commit and
cherry-pick the three fix commits onto it:

```sh
PIN="$(sed -n 's/^LLAMA_VERSION?=//p' backend/cpp/llama-cpp/Makefile)"

git clone https://github.com/maximosipov/llama.cpp.git backend/cpp/llama-cpp/llama.cpp
cd backend/cpp/llama-cpp/llama.cpp
git fetch https://github.com/ggml-org/llama.cpp.git master --tags
git checkout -b build "$PIN"
# The 3 fix commits from the fork's macbook-pro-2019-radeon-5500m-4gb branch, oldest
# first. Explicit SHAs: upstream llama.cpp history has its own unrelated "vulkan: fix ..."
# commits, so a message grep is not safe here.
git cherry-pick 7847fca13 595936bcd a91f29c5f
git submodule update --init --recursive --depth 1
cd ../../../..
```

If a future `LLAMA_VERSION` bump makes those stop applying, the fixes are narrow
(`ggml-vulkan.cpp` plus two `.comp` shaders) — diff them off the branch and re-apply.

Then graft in LocalAI's gRPC server. **Run this once**: `prepare.sh` applies LocalAI's own
patches to the llama.cpp tree and is not idempotent.

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

Two non-obvious details, both of which produce a build that *looks* fine and behaves
wrongly:

- **`CMAKE_ARGS` must be exported, not passed on the make command line.**
  `backend/cpp/llama-cpp/Makefile` does `CMAKE_ARGS?=` and then several `CMAKE_ARGS+=`.
  A command-line variable assignment overrides those appends, silently dropping
  `-DGGML_VULKAN=1` itself. An environment variable seeds `?=` and keeps the appends.
- **`-DGGML_METAL=OFF` is required.** In that Makefile, `BUILD_TYPE=vulkan` and the
  `Darwin` case are arms of a single if/else chain — so on macOS, choosing vulkan means
  the Darwin arm (the one that would have turned Metal off) never runs. ggml auto-enables
  Metal on macOS, and Metal then wins device 0 at runtime.

Gate:

```sh
otool -L backend/cpp/llama-cpp/grpc-server | grep -i vulkan     # expect libvulkan
otool -L backend/cpp/llama-cpp/grpc-server | grep -i Metal      # expect no output
```

## 4. Register the backend and serve

LocalAI launches an external backend as a child process, so the MoltenVK environment has
to be on the child. Wrap the binary:

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

Model YAML. **The GGUF must live inside `--models-path`** (LocalAI rejects absolute paths
outside it as invalid) — a symlink is fine:

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
# Flash attention measures ~3.9x SLOWER on MoltenVK/RDNA1 (no cooperative-matrix
# kernels, so it takes a fallback path). Leave it off.
flash_attention: "false"
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

Expect a coherent `54`. Garbled tokens mean the wrong device was selected — recheck
`GGML_VK_VISIBLE_DEVICES`.

LocalAI's own hardware probe reports `GPU vendor=""` and `Total available VRAM 0` on
macOS, so **its logs will not tell you whether the model reached the GPU**, and it does
not forward the backend's ggml banner. Ask IOKit instead:

```sh
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

A 4B Q4_K_M model resident on the GPU shows ~3.4 GiB in use on the AMD accelerator node.
If it stays near idle while generation runs, the model is on the CPU — check that
`gpu_layers` survived into the load (LocalAI's macOS hardware auto-tuning can zero it;
`LOCALAI_DISABLE_HARDWARE_DEFAULTS=true` turns that off).

A thinking model (Qwen3, etc.) returns its chain of thought in `choices[].message.reasoning`
and leaves `content` empty until it closes the think block — an empty `content` with
`finish_reason: "length"` is a token budget problem, not a correctness one.
