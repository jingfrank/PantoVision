import argparse
import csv
import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


_METRIC_RE = re.compile(r"^(?P<key>[A-Za-z0-9_]+):\s*(?P<val>[-+]?\d*\.?\d+)")


@dataclass
class AccuracyResult:
    path: str
    metrics: Dict[str, float]


def parse_quantiles_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def find_latest_results_file(results_dir: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(results_dir, "results_*.txt")))
    return candidates[-1] if candidates else None


def parse_results_txt(path: str) -> AccuracyResult:
    metrics: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _METRIC_RE.match(line.strip())
            if not m:
                continue
            key = m.group("key")
            val = float(m.group("val"))
            metrics[key] = val
    return AccuracyResult(path=path, metrics=metrics)


def run_benchmark(
    python_exe: str,
    benchmark_py: str,
    checkpoint: str,
    device: str,
    warmup_runs: int,
    test_runs: int,
    latency_bs: int,
    throughput_bs: int,
    dtype: str,
    out_json: str,
    tag: str,
    log_dir: Optional[str],
) -> Dict:
    cmd = [
        python_exe,
        benchmark_py,
        "--checkpoint",
        checkpoint,
        "--device",
        device,
        "--warmup-runs",
        str(warmup_runs),
        "--test-runs",
        str(test_runs),
        "--latency-batch-size",
        str(latency_bs),
        "--throughput-batch-size",
        str(throughput_bs),
        "--dtype",
        dtype,
        "--out-json",
        out_json,
        "--tag",
        tag,
    ]
    if log_dir:
        cmd += ["--log-dir", log_dir]

    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Sweep n_quantiles trade-offs (accuracy/latency/memory)")
    p.add_argument("--quantiles", type=str, default="1,2,3,4,5,6,7,8,9")
    p.add_argument(
        "--checkpoint-pattern",
        type=str,
        required=True,
        help="e.g. /root/shared-nvme/FastAD/checkpoints/MVTec_AD/wide_resnet50_2/cable/q{q}/fastad.pth",
    )
    p.add_argument(
        "--results-dir-pattern",
        type=str,
        default=None,
        help="Optional. e.g. /root/shared-nvme/FastAD/outputs/MVTec_AD/cable/base/q{q}",
    )
    p.add_argument(
        "--accuracy-key",
        type=str,
        default="fused_auroc",
        help="Metric key in results_*.txt (e.g. fused_auroc / patch_auroc)",
    )
    p.add_argument("--python", type=str, default="python", help="Python executable to run benchmark")
    p.add_argument(
        "--benchmark-py",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "benchmark.py"),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--warmup-runs", type=int, default=200)
    p.add_argument("--test-runs", type=int, default=200)
    p.add_argument("--latency-batch-size", type=int, default=1)
    p.add_argument("--throughput-batch-size", type=int, default=16)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--out-csv", type=str, default="tradeoff.csv")
    p.add_argument("--log-dir", type=str, default=None)
    args = p.parse_args()

    quantiles = parse_quantiles_list(args.quantiles)
    os.makedirs(args.out_dir, exist_ok=True)

    out_csv_path = args.out_csv
    if not os.path.isabs(out_csv_path):
        out_csv_path = os.path.join(args.out_dir, out_csv_path)

    rows: List[Dict] = []

    for q in quantiles:
        checkpoint = args.checkpoint_pattern.format(q=q)
        tag = f"n_quantiles={q}"

        bench_json = os.path.join(args.out_dir, f"benchmark_q{q}.json")
        bench = run_benchmark(
            python_exe=args.python,
            benchmark_py=args.benchmark_py,
            checkpoint=checkpoint,
            device=args.device,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
            latency_bs=args.latency_batch_size,
            throughput_bs=args.throughput_batch_size,
            dtype=args.dtype,
            out_json=bench_json,
            tag=tag,
            log_dir=args.log_dir,
        )

        accuracy_val = None
        results_file = None
        if args.results_dir_pattern:
            results_dir = args.results_dir_pattern.format(q=q)
            results_file = find_latest_results_file(results_dir)
            if results_file:
                parsed = parse_results_txt(results_file)
                accuracy_val = parsed.metrics.get(args.accuracy_key)

        mem_final = bench.get("mem_final") or {}
        latency = bench.get("latency") or {}
        throughput = bench.get("throughput") or {}

        row = {
            "n_quantiles": q,
            "accuracy_key": args.accuracy_key,
            "accuracy": accuracy_val,
            "results_file": results_file,
            "checkpoint": checkpoint,
            "lat_mean_ms": latency.get("mean_ms"),
            "lat_p50_ms": latency.get("p50_ms"),
            "lat_p95_ms": latency.get("p95_ms"),
            "fps_mean": latency.get("fps_mean"),
            "throughput_images_s": throughput.get("images_per_s"),
            "gpu_alloc_mb": mem_final.get("alloc_mb"),
            "gpu_reserved_mb": mem_final.get("reserved_mb"),
            "gpu_peak_alloc_mb": mem_final.get("max_alloc_mb"),
            "gpu_peak_reserved_mb": mem_final.get("max_reserved_mb"),
        }
        rows.append(row)

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv_path}")


if __name__ == "__main__":
    main()
