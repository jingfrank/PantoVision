import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
import tqdm

from fastad2 import FastAD
def setup_logging(log_dir):
    """Configure logging """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f"{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def _mem_stats_mb():
    return {
        "alloc_mb": float(torch.cuda.memory_allocated() / 1024**2),
        "reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
        "max_alloc_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2),
    }


def benchmark(
    model,
    input_size=(3, 224, 224),
    device="cuda",
    warmup_runs=200,
    test_runs=200,
    latency_batch_size=1,
    throughput_batch_size=16,
    dtype="fp16",
):
    model = model.to(device).eval()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    init_mem = _mem_stats_mb() if device.startswith("cuda") and torch.cuda.is_available() else None
    logging.info(f"Model initialized. GPU mem: {init_mem}" if init_mem else "Model initialized.")

    # Warmup
    dummy = torch.randn(latency_batch_size, *input_size, device=device)
    if dtype == "fp16":
        dummy = dummy.half()
    else:
        dummy = dummy.float()

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model._predict(dummy)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_mem = _mem_stats_mb() if device.startswith("cuda") and torch.cuda.is_available() else None

    # Latency
    lat_input = torch.randn(latency_batch_size, *input_size, device=device)
    lat_input = lat_input.half() if dtype == "fp16" else lat_input.float()

    latencies_ms = []
    with torch.no_grad():
        for _ in tqdm.tqdm(range(test_runs), desc=f"Latency (bs={latency_batch_size})"):
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = model._predict(lat_input)
            # ensure work is done
            if isinstance(out, torch.Tensor):
                _ = out.sum().item()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    latencies_ms = np.asarray(latencies_ms, dtype=np.float64)
    latency_summary = {
        "mean_ms": float(latencies_ms.mean()),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "fps_mean": float(1000.0 / latencies_ms.mean()),
        "batch_size": int(latency_batch_size),
        "runs": int(test_runs),
    }

    # Throughput
    thr_input = torch.randn(throughput_batch_size, *input_size, device=device)
    thr_input = thr_input.half() if dtype == "fp16" else thr_input.float()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    thr_start = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(test_runs), desc=f"Throughput (bs={throughput_batch_size})"):
            _ = model._predict(thr_input)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    thr_end = time.perf_counter()

    total_s = thr_end - thr_start
    total_images = test_runs * throughput_batch_size
    throughput_summary = {
        "images_per_s": float(total_images / total_s),
        "batch_size": int(throughput_batch_size),
        "runs": int(test_runs),
        "avg_batch_ms": float((total_s / test_runs) * 1000.0),
        "avg_image_ms": float((total_s / total_images) * 1000.0),
    }

    final_mem = _mem_stats_mb() if device.startswith("cuda") and torch.cuda.is_available() else None

    return {
        "device": device,
        "dtype": dtype,
        "input_size": list(input_size),
        "warmup_runs": int(warmup_runs),
        "latency": latency_summary,
        "throughput": throughput_summary,
        "mem_init": init_mem,
        "mem_after_warmup": warmup_mem,
        "mem_final": final_mem,
    }

def load_model(load_path,device):
    fastad = FastAD.load_model(load_path, device=device)
    fastad.eval()
    return fastad

def _parse_args():
    p = argparse.ArgumentParser(description="FastAD latency/memory benchmark")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to fastad.pth")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    p.add_argument("--input-size", type=str, default="3,224,224", help="C,H,W")
    p.add_argument("--warmup-runs", type=int, default=200)
    p.add_argument("--test-runs", type=int, default=200)
    p.add_argument("--latency-batch-size", type=int, default=1)
    p.add_argument("--throughput-batch-size", type=int, default=16)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None, help="Metadata tag (e.g. n_quantiles=3)")
    p.add_argument("--out-json", type=str, default=None, help="Write results JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.log_dir:
        logger = setup_logging(args.log_dir)
    else:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    c, h, w = (int(x) for x in args.input_size.split(","))
    input_size = (c, h, w)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        device = "cpu"

    model = load_model(args.checkpoint, device)
    results = benchmark(
        model,
        input_size=input_size,
        device=device,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs,
        latency_batch_size=args.latency_batch_size,
        throughput_batch_size=args.throughput_batch_size,
        dtype=args.dtype,
    )

    results["checkpoint"] = args.checkpoint
    results["tag"] = args.tag
    results["timestamp"] = datetime.now().isoformat(timespec="seconds")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)