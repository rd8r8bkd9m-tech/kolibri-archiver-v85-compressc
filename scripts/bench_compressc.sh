#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/bin/kolibri_archiver"
INPUT_FILE="${1:-$ROOT_DIR/src/compress.c}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file not found: $INPUT_FILE" >&2
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  echo "Build first: make all" >&2
  exit 1
fi

input_size=$(stat -f%z "$INPUT_FILE")

run_timed() {
  local out_var="$1"
  shift
  local start end elapsed
  start=$(python3 -c 'import time; print(time.perf_counter())')
  if "$@"; then
    end=$(python3 -c 'import time; print(time.perf_counter())')
    elapsed=$(python3 - <<PY
s=float("$start")
e=float("$end")
print(f"{(e-s)*1000:.2f}")
PY
)
    printf -v "$out_var" "%s" "$elapsed"
    return 0
  fi
  printf -v "$out_var" "0"
  return 1
}

ratio() {
  python3 - <<PY
orig=$1
comp=$2
print(f"{orig/comp:.3f}" if comp>0 else "0")
PY
}

# Kolibri
k_out="$TMP_DIR/out.kolibri"
k_dec="$TMP_DIR/out.dec"
run_timed k_comp_ms "$BIN" compress "$INPUT_FILE" "$k_out" >/dev/null 2>&1 || true
run_timed k_decomp_ms "$BIN" decompress "$k_out" "$k_dec" >/dev/null 2>&1 || true
k_size=$(stat -f%z "$k_out" 2>/dev/null || echo 0)
k_ratio=$(ratio "$input_size" "$k_size")

results="$TMP_DIR/results.csv"
: > "$results"
echo "Kolibri V85 (compress.c);$k_size;$k_ratio;$k_comp_ms;$k_decomp_ms" >> "$results"

bench_ext() {
  local name="$1"
  local comp_cmd="$2"
  local dec_cmd="$3"
  local out="$4"
  local dec="$5"

  run_timed comp_ms bash -lc "$comp_cmd" >/dev/null 2>&1 || true
  run_timed decomp_ms bash -lc "$dec_cmd" >/dev/null 2>&1 || true
  local size="0"
  [[ -f "$out" ]] && size=$(stat -f%z "$out")
  local r
  r=$(ratio "$input_size" "$size")
  echo "$name;$size;$r;$comp_ms;$decomp_ms" >> "$results"
}

bench_ext "gzip -9" \
  "gzip -9 -c '$INPUT_FILE' > '$TMP_DIR/out.gz'" \
  "gzip -d -c '$TMP_DIR/out.gz' > '$TMP_DIR/out.gz.dec'" \
  "$TMP_DIR/out.gz" "$TMP_DIR/out.gz.dec"

bench_ext "bzip2 -9" \
  "bzip2 -9 -c '$INPUT_FILE' > '$TMP_DIR/out.bz2'" \
  "bzip2 -d -c '$TMP_DIR/out.bz2' > '$TMP_DIR/out.bz2.dec'" \
  "$TMP_DIR/out.bz2" "$TMP_DIR/out.bz2.dec"

bench_ext "xz -9e" \
  "xz -9e -c '$INPUT_FILE' > '$TMP_DIR/out.xz'" \
  "xz -d -c '$TMP_DIR/out.xz' > '$TMP_DIR/out.xz.dec'" \
  "$TMP_DIR/out.xz" "$TMP_DIR/out.xz.dec"

bench_ext "zstd -19" \
  "zstd -19 -f -c '$INPUT_FILE' > '$TMP_DIR/out.zst'" \
  "zstd -d -f -c '$TMP_DIR/out.zst' > '$TMP_DIR/out.zst.dec'" \
  "$TMP_DIR/out.zst" "$TMP_DIR/out.zst.dec"

bench_ext "lz4 -9" \
  "lz4 -9 -f -c '$INPUT_FILE' > '$TMP_DIR/out.lz4'" \
  "lz4 -d -f -c '$TMP_DIR/out.lz4' > '$TMP_DIR/out.lz4.dec'" \
  "$TMP_DIR/out.lz4" "$TMP_DIR/out.lz4.dec"

if command -v brotli >/dev/null 2>&1; then
  bench_ext "brotli -q11" \
    "brotli -q 11 -c '$INPUT_FILE' > '$TMP_DIR/out.br'" \
    "brotli -d -c '$TMP_DIR/out.br' > '$TMP_DIR/out.br.dec'" \
    "$TMP_DIR/out.br" "$TMP_DIR/out.br.dec"
fi

mkdir -p "$ROOT_DIR/docs"
{
  echo "Архиватор;Размер(bytes);Ratio;Сжатие(ms);Распаковка(ms)"
  cat "$results"
} | tee "$ROOT_DIR/docs/benchmark_compressc_latest.csv"

echo
echo "Saved: $ROOT_DIR/docs/benchmark_compressc_latest.csv"
