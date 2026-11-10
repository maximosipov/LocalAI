#!/usr/bin/env bash
# Package LocalAI + the Vulkan/MoltenVK backends as an installable "LocalAI M.dmg",
# for an Intel Mac with a discrete AMD GPU (built for a 2019 MacBook Pro, Radeon Pro 5500M).
#
#   contrib/macos-radeon-app/build-app.sh
#
# Expects ./local-ai (make build) and the Vulkan llama-cpp backend already built, and
# optionally the Vulkan stablediffusion-ggml backend; see RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md.
#   LOCALAI_DIR   LocalAI tree holding those builds (default: this checkout)
#   MODELS_SEED   directory of model YAML files to seed on first run (default: none)
#   DIST          output directory (default: dist/macos-app)
#
# The bundle is a menu-bar app that owns the server: it starts it with the MoltenVK
# environment set and the Radeon pinned, and stops it on quit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BREW_PREFIX="$(brew --prefix)"

APP_NAME="${APP_NAME:-LocalAI M}"
LOCALAI_DIR="${LOCALAI_DIR:-$(cd "$HERE/../.." && pwd)}"
DIST="${DIST:-$LOCALAI_DIR/dist/macos-app}"
APP="$DIST/$APP_NAME.app"
RES="$APP/Contents/Resources"
FW="$APP/Contents/Frameworks"
LLAMA_BACKEND="$LOCALAI_DIR/backend/cpp/llama-cpp/grpc-server"
SD_DIR="$LOCALAI_DIR/backend/go/stablediffusion-ggml"

status() { echo >&2 ">>> $*"; }

[ -x "$LOCALAI_DIR/local-ai" ] || { echo "missing $LOCALAI_DIR/local-ai -- run make build" >&2; exit 1; }
[ -x "$LLAMA_BACKEND" ]        || { echo "missing $LLAMA_BACKEND -- build the Vulkan llama-cpp backend first" >&2; exit 1; }

MOLTENVK_LIB="$BREW_PREFIX/opt/molten-vk/lib/libMoltenVK.dylib"
VULKAN_LOADER="$BREW_PREFIX/opt/vulkan-loader/lib/libvulkan.1.dylib"
for f in "$MOLTENVK_LIB" "$VULKAN_LOADER"; do
  [ -f "$f" ] || { echo "missing $f (brew install molten-vk vulkan-loader)" >&2; exit 1; }
done

# --- 1. Skeleton -------------------------------------------------------------------
status "Assembling $APP_NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/backends/llama-cpp" "$RES/backends/stablediffusion-ggml" "$RES/models-seed" "$FW" "$RES/vulkan"

cp -a "$LOCALAI_DIR/local-ai" "$RES/local-ai"
cp -a "$LLAMA_BACKEND"        "$RES/backends/llama-cpp/grpc-server"

SD_PRESENT=0
if [ -x "$SD_DIR/stablediffusion-ggml" ]; then
  cp -a "$SD_DIR/stablediffusion-ggml" "$RES/backends/stablediffusion-ggml/"
  for cand in "$SD_DIR"/libgosd-fallback.dylib "$SD_DIR"/libgosd-fallback.so; do
    [ -e "$cand" ] && cp -a "$cand" "$RES/backends/stablediffusion-ggml/"
  done
  SD_PRESENT=1
else
  status "WARNING: image backend not built -- packaging text only"
fi

if [ -n "${MODELS_SEED:-}" ]; then
  cp -a "$MODELS_SEED"/*.yaml "$RES/models-seed/"
fi

# --- 2. Recursive dylib bundling ---------------------------------------------------
# grpc-server pulls ~107 Homebrew dylibs (grpc, protobuf, abseil, re2, c-ares,
# openssl@3, libomp, vulkan-loader). Copy the transitive closure into Frameworks/ and
# rewrite every install name to @rpath so the bundle stands alone. This is what
# dylibbundler would do; doing it here keeps the toolchain to what's already required.
status "Bundling Homebrew dylibs (transitive closure)"
declare -a QUEUE=("$RES/backends/llama-cpp/grpc-server")
[ "$SD_PRESENT" = 1 ] && for f in "$RES/backends/stablediffusion-ggml"/*; do QUEUE+=("$f"); done
cp -f "$VULKAN_LOADER" "$FW/" && cp -f "$MOLTENVK_LIB" "$FW/"
chmod u+w "$FW"/*.dylib
QUEUE+=("$FW/libvulkan.1.dylib" "$FW/libMoltenVK.dylib")

# macOS ships bash 3.2, which has no associative arrays -- track seen names in a
# delimited string instead.
SEEN=" "
NBUNDLED=0
while [ ${#QUEUE[@]} -gt 0 ]; do
  cur="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
  [ -f "$cur" ] || continue
  while read -r dep; do
    case "$dep" in "$BREW_PREFIX"/*) ;; *) continue ;; esac
    base="$(basename "$dep")"
    case "$SEEN" in
      *" $base "*) ;;
      *)
        SEEN="$SEEN$base "
        if cp -f "$dep" "$FW/$base" 2>/dev/null; then
          chmod u+w "$FW/$base"
          install_name_tool -id "@rpath/$base" "$FW/$base" 2>/dev/null || true
          QUEUE+=("$FW/$base")
          NBUNDLED=$((NBUNDLED + 1))
        fi
        ;;
    esac
    install_name_tool -change "$dep" "@rpath/$base" "$cur" 2>/dev/null || true
  done < <(otool -L "$cur" | tail -n +2 | awk '{print $1}')
done
status "  bundled $NBUNDLED dylibs"

# rpaths: executables live in Resources/backends/<name>/, libs in Frameworks/
install_name_tool -add_rpath "@executable_path/../../../../Frameworks" "$RES/backends/llama-cpp/grpc-server" 2>/dev/null || true
if [ "$SD_PRESENT" = 1 ]; then
  install_name_tool -add_rpath "@executable_path/../../../../Frameworks" "$RES/backends/stablediffusion-ggml/stablediffusion-ggml" 2>/dev/null || true
  for lib in "$RES/backends/stablediffusion-ggml"/libgosd-fallback.*; do
    [ -e "$lib" ] && install_name_tool -add_rpath "@loader_path/../../../Frameworks" "$lib" 2>/dev/null || true
  done
fi
for lib in "$FW"/*.dylib; do install_name_tool -add_rpath "@loader_path" "$lib" 2>/dev/null || true; done

# ICD json pointing at the bundled MoltenVK, relative to its own directory.
cat > "$RES/vulkan/MoltenVK_icd.json" <<'JSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "../../Frameworks/libMoltenVK.dylib",
        "api_version": "1.2.0",
        "is_portability_driver": true
    }
}
JSON

# --- 3. Menu-bar app ----------------------------------------------------------------
# Preferred: a real NSStatusItem app (no Dock icon) that owns the server process.
# Falls back to the shell launcher below if swiftc is unavailable.
MENUBAR_SRC="$HERE/LocalAIMenuBar.swift"
USE_MENUBAR=0
if command -v swiftc >/dev/null && [ -f "$MENUBAR_SRC" ]; then
  status "Compiling menu-bar app"
  if swiftc -O -target x86_64-apple-macosx14.0 \
       -framework AppKit -o "$APP/Contents/MacOS/LocalAI" "$MENUBAR_SRC" 2>&1 | sed 's/^/    /'; then
    USE_MENUBAR=1
  else
    status "  swiftc failed -- falling back to the shell launcher"
  fi
fi

if [ "$USE_MENUBAR" = 0 ]; then
cat > "$APP/Contents/MacOS/LocalAI" <<'LAUNCHER'
#!/usr/bin/env bash
# LocalAI M -- starts the server with the Vulkan/MoltenVK environment, opens the WebUI,
# and shuts down when the dialog is dismissed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/../Resources"
FW="$HERE/../Frameworks"
ADDR="${LOCALAI_ADDR:-127.0.0.1:8085}"
SUPPORT="$HOME/Library/Application Support/LocalAI"
MODELS="$SUPPORT/models"
mkdir -p "$MODELS" "$SUPPORT/logs"

# Seed model configs on first run. Weights are NOT bundled (they are gigabytes); if the
# development checkout is present, symlink them, otherwise the YAMLs document what to add.
# Seed only while the models folder holds no config, so deleted models stay deleted.
if ! ls "$MODELS"/*.yaml >/dev/null 2>&1; then
  for y in "$RES"/models-seed/*.yaml; do
    [ -e "$y" ] && cp "$y" "$MODELS/"
  done
fi
for d in "${LOCALAI_DEV_MODELS:-$HOME/Projects/macbook-pro-2019-local-ai/scratchpad/models-localai}"; do
  [ -d "$d" ] || continue
  for w in "$d"/*.gguf "$d"/*.safetensors; do
    [ -e "$w" ] && [ ! -e "$MODELS/$(basename "$w")" ] && ln -sfn "$(readlink "$w" || echo "$w")" "$MODELS/" || true
  done
done

export VK_ICD_FILENAMES="$RES/vulkan/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$FW:${DYLD_LIBRARY_PATH:-}"
# Pin the discrete Radeon. Never the integrated GPU.
export GGML_VK_VISIBLE_DEVICES=0
export LOCALAI_DISABLE_HARDWARE_DEFAULTS=true

LLAMA_WRAP="$SUPPORT/run-llama-vulkan.sh"
cat > "$LLAMA_WRAP" <<EOF
#!/usr/bin/env bash
export VK_ICD_FILENAMES="$RES/vulkan/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$FW:\${DYLD_LIBRARY_PATH:-}"
export GGML_VK_VISIBLE_DEVICES=0
exec "$RES/backends/llama-cpp/grpc-server" "\$@"
EOF
chmod +x "$LLAMA_WRAP"
EXTERNAL="llama-cpp:$LLAMA_WRAP"

if [ -x "$RES/backends/stablediffusion-ggml/stablediffusion-ggml" ]; then
  SD_WRAP="$SUPPORT/run-sd-vulkan.sh"
  SD_LIB="$RES/backends/stablediffusion-ggml/libgosd-fallback.dylib"
  [ -e "$SD_LIB" ] || SD_LIB="$RES/backends/stablediffusion-ggml/libgosd-fallback.so"
  cat > "$SD_WRAP" <<EOF
#!/usr/bin/env bash
export VK_ICD_FILENAMES="$RES/vulkan/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$FW:\${DYLD_LIBRARY_PATH:-}"
export GGML_VK_VISIBLE_DEVICES=0
export SD_LIBRARY="$SD_LIB"
exec "$RES/backends/stablediffusion-ggml/stablediffusion-ggml" "\$@"
EOF
  chmod +x "$SD_WRAP"
  EXTERNAL="$EXTERNAL,stablediffusion-ggml:$SD_WRAP"
fi

LOG="$SUPPORT/logs/local-ai.log"
"$RES/local-ai" run --address "$ADDR" --models-path "$MODELS" --addons-enabled=false \
  --external-grpc-backends "$EXTERNAL" >>"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://$ADDR/readyz" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || break
  sleep 1
done
open "http://$ADDR/"

osascript -e "display dialog \"LocalAI is running at http://$ADDR

Models: $MODELS
Log: $LOG

Click Stop to shut the server down.\" buttons {\"Stop\"} default button 1 with title \"LocalAI M\"" >/dev/null 2>&1 || true
LAUNCHER
fi
chmod +x "$APP/Contents/MacOS/LocalAI"

VERSION="$(cd "$LOCALAI_DIR" && git describe --tags --always 2>/dev/null || echo dev)"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>ai.localai.m.radeon</string>
  <key>CFBundleExecutable</key><string>LocalAI</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><$( [ "$USE_MENUBAR" = 1 ] && echo true || echo false )/>
</dict></plist>
PLIST

# --- 4. Acceptance test: nothing may still point at Homebrew -------------------------
status "Checking for residual Homebrew paths"
LEAKS=$(find "$RES" "$FW" -type f \( -name '*.dylib' -o -name '*.so' -o -name 'grpc-server' -o -name 'stablediffusion-ggml' -o -name 'local-ai' \) \
  -exec otool -L {} \; 2>/dev/null | grep -F "$BREW_PREFIX/opt" || true)
if [ -n "$LEAKS" ]; then
  echo "FAIL: bundle still references Homebrew:" >&2; echo "$LEAKS" | sort -u | head -20 >&2; exit 1
fi
echo "  clean -- no $BREW_PREFIX/opt references"

# --- 5. Ad-hoc signing + DMG ---------------------------------------------------------
status "Ad-hoc signing (no Apple identity on this machine -- not notarized)"
codesign -f -s - --deep "$APP" 2>/dev/null || codesign -f -s - "$APP"

status "Creating DMG"
DMG="$DIST/${APP_NAME// /-}.dmg"
rm -f "$DMG"
STAGE="$DIST/dmg-stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -a "$APP" "$STAGE/"
ln -sf /Applications "$STAGE/Applications"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"

echo
echo "Built: $DMG  ($(du -h "$DMG" | cut -f1))"
echo "First launch: xattr -dr com.apple.quarantine \"/Applications/$APP_NAME.app\" (ad-hoc signed, not notarized)"
