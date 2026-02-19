#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
SILESIA_URL = "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"


def download_if_missing(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as resp, target.open("wb") as out:
        shutil.copyfileobj(resp, out)


def extract_zip_if_needed(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def pick_silesia_files(root: Path) -> list[Path]:
    candidates = [
        "dickens",
        "mozilla",
        "osdb",
        "samba",
    ]
    all_files = [p for p in root.rglob("*") if p.is_file() and p.name not in {".extracted"}]
    by_lower = {p.name.lower(): p for p in all_files}
    picked = []
    for name in candidates:
        if name in by_lower:
            picked.append(by_lower[name])
    if len(picked) < 2:
        # fallback: top 4 largest files
        picked = sorted(all_files, key=lambda p: p.stat().st_size, reverse=True)[:4]
    return picked


def run_cmd(args: list[str], stdout_path: Path | None = None) -> None:
    if stdout_path is None:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        with stdout_path.open("wb") as out:
            subprocess.run(args, check=True, stdout=out, stderr=subprocess.DEVNULL)


def compressor_defs(kolibri_bin: Path):
    def k_comp(inp: Path, out: Path):
        run_cmd([str(kolibri_bin), "compress", str(inp), str(out)])

    def k_decomp(out: Path, dec: Path):
        run_cmd([str(kolibri_bin), "decompress", str(out), str(dec)])

    def gz_comp(inp: Path, out: Path):
        run_cmd(["gzip", "-9", "-c", str(inp)], stdout_path=out)

    def gz_decomp(out: Path, dec: Path):
        run_cmd(["gzip", "-d", "-c", str(out)], stdout_path=dec)

    def bz_comp(inp: Path, out: Path):
        run_cmd(["bzip2", "-9", "-c", str(inp)], stdout_path=out)

    def bz_decomp(out: Path, dec: Path):
        run_cmd(["bzip2", "-d", "-c", str(out)], stdout_path=dec)

    def xz_comp(inp: Path, out: Path):
        run_cmd(["xz", "-6", "-c", str(inp)], stdout_path=out)

    def xz_decomp(out: Path, dec: Path):
        run_cmd(["xz", "-d", "-c", str(out)], stdout_path=dec)

    def zstd_comp(inp: Path, out: Path):
        run_cmd(["zstd", "-19", "-f", "-c", str(inp)], stdout_path=out)

    def zstd_decomp(out: Path, dec: Path):
        run_cmd(["zstd", "-d", "-f", "-c", str(out)], stdout_path=dec)

    def lz4_comp(inp: Path, out: Path):
        run_cmd(["lz4", "-9", "-f", "-c", str(inp)], stdout_path=out)

    def lz4_decomp(out: Path, dec: Path):
        run_cmd(["lz4", "-d", "-f", "-c", str(out)], stdout_path=dec)

    return [
        ("Kolibri V85", k_comp, k_decomp),
        ("gzip -9", gz_comp, gz_decomp),
        ("bzip2 -9", bz_comp, bz_decomp),
        ("xz -6", xz_comp, xz_decomp),
        ("zstd -19", zstd_comp, zstd_decomp),
        ("lz4 -9", lz4_comp, lz4_decomp),
    ]


def timed(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def benchmark_one(inp: Path, comp_name: str, comp_fn, decomp_fn, repeats: int, tmp: Path):
    comp_times = []
    decomp_times = []
    ok = True
    out = tmp / f"{inp.name}.{comp_name.replace(' ', '_').replace('-', '_')}"
    dec = tmp / f"{inp.name}.{comp_name.replace(' ', '_').replace('-', '_')}.dec"

    for _ in range(repeats):
        if out.exists():
            out.unlink()
        if dec.exists():
            dec.unlink()
        try:
            c_ms = timed(comp_fn, inp, out)
            d_ms = timed(decomp_fn, out, dec)
            same = dec.read_bytes() == inp.read_bytes()
        except Exception:
            c_ms = 0.0
            d_ms = 0.0
            same = False
        comp_times.append(c_ms)
        decomp_times.append(d_ms)
        ok = ok and same

    csize = out.stat().st_size if out.exists() else 0
    return {
        "compressed_bytes": csize,
        "ratio": (inp.stat().st_size / csize) if csize > 0 else 0.0,
        "compress_ms_median": statistics.median(comp_times),
        "decompress_ms_median": statistics.median(decomp_times),
        "roundtrip_ok": ok,
    }


def main():
    ap = argparse.ArgumentParser(description="Standard corpus benchmark for Kolibri V85")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root)
    kolibri_bin = root / "bin" / "kolibri_archiver"
    if not kolibri_bin.exists():
        print("Build first: make all", file=sys.stderr)
        sys.exit(1)

    data_dir = root / "bench_data"
    raw = data_dir / "raw"
    corp = data_dir / "corpora"

    enwik8_zip = raw / "enwik8.zip"
    silesia_zip = raw / "silesia.zip"

    print("[1/4] Downloading corpora...")
    download_if_missing(ENWIK8_URL, enwik8_zip)
    download_if_missing(SILESIA_URL, silesia_zip)

    print("[2/4] Extracting corpora...")
    enwik8_dir = corp / "enwik8"
    silesia_dir = corp / "silesia"
    extract_zip_if_needed(enwik8_zip, enwik8_dir)
    extract_zip_if_needed(silesia_zip, silesia_dir)

    datasets: list[tuple[str, Path]] = []
    enwik8_file = next((p for p in enwik8_dir.rglob("*") if p.is_file() and p.name.lower() == "enwik8"), None)
    if enwik8_file:
        datasets.append(("enwik8", enwik8_file))

    for p in pick_silesia_files(silesia_dir):
        datasets.append((f"silesia/{p.name}", p))

    datasets.append(("code/compress.c", root / "src" / "compress.c"))

    print("[3/4] Benchmarking...")
    rows = []
    defs = compressor_defs(kolibri_bin)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for dname, dpath in datasets:
            isize = dpath.stat().st_size
            print(f"  dataset: {dname} ({isize} bytes)")
            for cname, cfn, dfn in defs:
                r = benchmark_one(dpath, cname, cfn, dfn, args.repeats, tmp)
                c_ms = r["compress_ms_median"]
                d_ms = r["decompress_ms_median"]
                c_mbps = (isize / 1048576.0) / (c_ms / 1000.0) if c_ms > 0 else 0.0
                d_mbps = (isize / 1048576.0) / (d_ms / 1000.0) if d_ms > 0 else 0.0
                rows.append({
                    "dataset": dname,
                    "input_bytes": isize,
                    "archiver": cname,
                    "compressed_bytes": r["compressed_bytes"],
                    "ratio": f"{r['ratio']:.3f}",
                    "compress_ms_median": f"{c_ms:.2f}",
                    "decompress_ms_median": f"{d_ms:.2f}",
                    "compress_mbps": f"{c_mbps:.2f}",
                    "decompress_mbps": f"{d_mbps:.2f}",
                    "roundtrip_ok": "yes" if r["roundtrip_ok"] else "no",
                })

    print("[4/4] Writing reports...")
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    csv_path = docs / "benchmark_world_standard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)

    # Summary for Kolibri ranks
    summary_path = docs / "benchmark_world_standard_summary.md"
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    lines = [
        "# Benchmark Summary (Standard Corpora)",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Repeats: {args.repeats} (median)",
        "",
        "| Dataset | Kolibri Ratio Rank | Kolibri Compress Speed Rank |",
        "|---|---:|---:|",
    ]

    ratio_ranks = []
    speed_ranks = []
    for dname, arr in by_dataset.items():
        ratio_sorted = sorted(arr, key=lambda x: float(x["ratio"]), reverse=True)
        speed_sorted = sorted(arr, key=lambda x: float(x["compress_ms_median"]))
        rr = next(i + 1 for i, x in enumerate(ratio_sorted) if x["archiver"] == "Kolibri V85")
        sr = next(i + 1 for i, x in enumerate(speed_sorted) if x["archiver"] == "Kolibri V85")
        total = len(arr)
        ratio_ranks.append((rr, total))
        speed_ranks.append((sr, total))
        lines.append(f"| {dname} | {rr}/{total} | {sr}/{total} |")

    avg_ratio = sum(r for r, _ in ratio_ranks) / len(ratio_ranks)
    avg_speed = sum(r for r, _ in speed_ranks) / len(speed_ranks)
    lines += [
        "",
        f"Average ratio rank: **{avg_ratio:.2f}**",
        f"Average speed rank: **{avg_speed:.2f}**",
        "",
        f"CSV: `{csv_path}`",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
