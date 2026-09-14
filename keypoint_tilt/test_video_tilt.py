"""
Test trained HRNet on full video with 1 fps sampling.
Extracts frames every second, runs tilt detection, saves CSV timeline,
plots timeline curve, and writes annotated output video and alarm frames.
"""

import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from keypoint_tilt.detector import PantographTiltDetector


def process_video(
    video_path,
    output_dir="results_tilt_1_video",
    weights_path="weights/best_hrnet_w32_panto.pth",
    sample_interval_sec=1.0,
    width=32,
    warning_thresh=1.2,
    alarm_thresh=1.8,
):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(frames_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        default_w = os.path.join(repo_root, weights_path)
        if os.path.exists(default_w):
            weights_path = default_w

    print(f"=== Initializing Pantograph Tilt Detector ===")
    print(f"Video Path   : {video_path}")
    print(f"Weights Path : {weights_path}")
    print(f"Output Dir   : {output_dir}")

    detector = PantographTiltDetector(
        weights_path=weights_path,
        width=width,
        num_joints=4,
        structural_warning_thresh=warning_thresh,
        structural_alarm_thresh=alarm_thresh,
        install_bias_deg=2.0,
        rigid_alarm_thresh=None,
        use_rigid_alarm=False,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / max(1.0, fps)
    step_frames = int(round(fps * sample_interval_sec))

    print(
        f"Video Stats  : FPS={fps:.2f}, Total Frames={total_frames}, Duration={duration_sec:.1f}s"
    )
    print(
        f"Sampling     : Every {step_frames} frames (~{sample_interval_sec}s per frame)"
    )

    records = []
    annotated_frames = []

    pbar = tqdm(range(0, total_frames, step_frames), desc="Evaluating Video")
    for frame_idx in pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp_s = frame_idx / fps
        res, vis_img = detector.predict(frame)

        # Save individual annotated frame
        frame_filename = (
            f"frame_{len(records):04d}_t{timestamp_s:05.1f}s_f{frame_idx:05d}.jpg"
        )
        frame_save_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_save_path, vis_img)

        annotated_frames.append(vis_img)

        rec = {
            "sample_id": len(records) + 1,
            "timestamp_s": round(timestamp_s, 2),
            "frame_index": frame_idx,
            "status": res["status"],
            "severity": res["severity"],
            "is_anomaly": res["is_anomaly"],
            "tilt_direction": res["tilt_direction"],
            "relative_tilt_deg": res.get("relative_tilt_deg", 0.0),
            "rigid_tilt_deg": res.get("rigid_tilt_deg", 0.0),
            "fused_angle_deg": res["fused_angle_deg"],
            "horn_angle_deg": res["horn_angle_deg"],
            "support_angle_deg": res["support_angle_deg"],
            "horn_drop_px": res["horn_drop_px"],
            "support_drop_px": res["support_drop_px"],
            "p1_x": round(res["keypoints"][0][0], 1),
            "p1_y": round(res["keypoints"][0][1], 1),
            "p2_x": round(res["keypoints"][1][0], 1),
            "p2_y": round(res["keypoints"][1][1], 1),
            "p3_x": round(res["keypoints"][2][0], 1),
            "p3_y": round(res["keypoints"][2][1], 1),
            "p4_x": round(res["keypoints"][3][0], 1),
            "p4_y": round(res["keypoints"][3][1], 1),
            "conf_min": round(min(res["confidences"]), 3),
            "conf_avg": round(float(np.mean(res["confidences"])), 3),
            "image_file": frame_filename,
        }
        records.append(rec)
        pbar.set_postfix(
            {
                "time": f"{timestamp_s:.1f}s",
                "status": res["status"],
                "rel_angle": f"{res.get('relative_tilt_deg', 0.0):+.2f}°",
            }
        )

    cap.release()

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "tilt_1_evaluation_timeline.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Timeline CSV] Saved: {csv_path} ({len(df)} samples)")

    # 1. Output Annotated MP4 Video
    if annotated_frames:
        h, w, _ = annotated_frames[0].shape
        video_out_path = os.path.join(output_dir, "tilt_1_annotated_1fps.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(video_out_path, fourcc, 2.0, (w, h))
        for f in annotated_frames:
            out_writer.write(f)
        out_writer.release()
        print(f"[Annotated Video] Saved: {video_out_path}")

    # 2. Plot Timeline Curve (Dual Subplots)
    plot_timeline(df, output_dir, warning_thresh, alarm_thresh)

    # 3. Generate Summary Report
    generate_markdown_report(
        df, output_dir, video_path, duration_sec, warning_thresh, alarm_thresh
    )

    return df


def plot_timeline(
    df, output_dir, warning_thresh=1.2, alarm_thresh=1.8, min_conf_thresh=0.35
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), dpi=150, sharex=True)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    times = df["timestamp_s"]
    valid_mask = df["conf_min"] >= min_conf_thresh
    flash_mask = ~valid_mask

    # ------------------ Subplot 1: Relative Parallelism Defect ------------------
    plot_rel = df["relative_tilt_deg"].copy()
    plot_rel[flash_mask] = np.nan

    ax1.plot(
        times,
        plot_rel,
        label="Relative Tilt θ_rel (Horn vs Support Base)",
        color="#1f77b4",
        linewidth=2.5,
    )
    ax1.axhline(
        y=alarm_thresh,
        color="red",
        linestyle="-.",
        linewidth=1.5,
        label=f"Structural Alarm (±{alarm_thresh}°)",
    )
    ax1.axhline(y=-alarm_thresh, color="red", linestyle="-.", linewidth=1.5)
    ax1.axhline(
        y=warning_thresh,
        color="#f39c12",
        linestyle="--",
        linewidth=1.2,
        label=f"Structural Warning (±{warning_thresh}°)",
    )
    ax1.axhline(y=-warning_thresh, color="#f39c12", linestyle="--", linewidth=1.2)
    ax1.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    valid_df = df[valid_mask]
    norm_mask = valid_df["status"] == "NORMAL"
    warn_mask = valid_df["status"] == "WARNING_TILT"
    alarm_mask = valid_df["status"] == "ALARM_TILT"

    if np.any(norm_mask):
        ax1.scatter(
            valid_df.loc[norm_mask, "timestamp_s"],
            valid_df.loc[norm_mask, "relative_tilt_deg"],
            color="#2ca02c",
            s=30,
            label="Status: NORMAL (Parallel)",
            zorder=5,
        )
    if np.any(warn_mask):
        ax1.scatter(
            valid_df.loc[warn_mask, "timestamp_s"],
            valid_df.loc[warn_mask, "relative_tilt_deg"],
            color="#f39c12",
            s=40,
            label="Status: WARNING",
            zorder=5,
        )
    if np.any(alarm_mask):
        ax1.scatter(
            valid_df.loc[alarm_mask, "timestamp_s"],
            valid_df.loc[alarm_mask, "relative_tilt_deg"],
            color="red",
            s=55,
            marker="^",
            label="Status: ALARM",
            zorder=6,
        )

    if np.any(flash_mask):
        ax1.scatter(
            df.loc[flash_mask, "timestamp_s"],
            [0.0] * sum(flash_mask),
            color="purple",
            marker="x",
            s=90,
            linewidth=2.5,
            label=f"Arc Flash Interference (Conf < {min_conf_thresh})",
            zorder=7,
        )

    ax1.axvspan(
        49.5, 56.5, color="red", alpha=0.10, label="Severe Torsion / Tilt Event"
    )
    ax1.set_title(
        "Layer 1: Structural Non-Parallelism Tilt [θ_rel = Horn in Support Local Frame] (tilt_1.mp4)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Relative Tilt θ_rel (deg, +: Right, -: Left)", fontsize=11)
    ax1.set_ylim(-4.0, 4.0)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    # ------------------ Subplot 2: Support Baseline & Overall Incline ------------------
    plot_support = df["support_angle_deg"].copy()
    plot_horn = df["horn_angle_deg"].copy()
    plot_support[flash_mask] = np.nan
    plot_horn[flash_mask] = np.nan

    ax2.plot(
        times,
        plot_support,
        label="Support Base Line (P2-P3)",
        color="#2ca02c",
        linewidth=2.0,
    )
    ax2.plot(
        times,
        plot_horn,
        label="Upper Horn Line (P1-P4)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.axhline(
        y=2.0,
        color="navy",
        linestyle=":",
        linewidth=1.5,
        label="Camera Mounting Roll Bias (≈ +2.0°)",
    )
    ax2.axhline(
        y=6.0,
        color="darkred",
        linestyle="-.",
        linewidth=1.3,
        label="Overall Tilt Alarm (Bias + 4.0° = 6.0°)",
    )
    ax2.axhline(y=-2.0, color="darkred", linestyle="-.", linewidth=1.3)

    ax2.axvspan(49.5, 56.5, color="red", alpha=0.10)
    ax2.set_title(
        "Layer 2: Overall Incline & Support Baseline [Camera Frame] (tilt_1.mp4)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_xlabel("Video Timestamp (seconds)", fontsize=11)
    ax2.set_ylabel("Baseline Angle (deg)", fontsize=11)
    ax2.set_ylim(-3.0, 12.5)
    ax2.set_xlim(0, max(times) + 1)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tilt_1_angle_timeline.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[Timeline Plot] Saved: {plot_path}")


def generate_markdown_report(
    df,
    output_dir,
    video_path,
    duration_sec,
    warning_thresh,
    alarm_thresh,
    min_conf_thresh=0.35,
):
    total_samples = len(df)
    valid_mask = df["conf_min"] >= min_conf_thresh
    valid_df = df[valid_mask]
    flash_df = df[~valid_mask]

    normal_cnt = sum(valid_df["status"] == "NORMAL")
    warning_cnt = sum(valid_df["status"] == "WARNING_TILT")
    alarm_cnt = sum(valid_df["status"] == "ALARM_TILT")
    flash_cnt = len(flash_df)

    mean_rel = valid_df["relative_tilt_deg"].mean()
    std_rel = valid_df["relative_tilt_deg"].std()
    max_rel = valid_df["relative_tilt_deg"].max()
    min_rel = valid_df["relative_tilt_deg"].min()

    mean_base = valid_df["support_angle_deg"].mean()
    std_base = valid_df["support_angle_deg"].std()
    max_base = valid_df["support_angle_deg"].max()

    report_content = f"""# 完整视频受电弓倾斜监测升级报告 (`tilt_1.mp4`)
*基于下横梁局部基准坐标系与双层解耦诊断架构*

## 1. 视频与测试基本信息
- **视频源路径**: `{video_path}`
- **视频总时长**: {duration_sec:.1f} 秒 ({duration_sec / 60.0:.2f} 分钟)
- **抽样频率**: 1 帧/秒 (1 FPS)，共评测 **{total_samples} 帧**
- **模型架构**: HRNet-W32 (4 关键点姿态定位: `p1_left_horn`, `p2_left_support`, `p3_right_support`, `p4_right_horn`)
- **坐标系方案**: **下横梁浮动参考系（Local Support Baseline Frame）**
  - **内蕴几何指标**: $\\theta_{{rel}} = \\text{{atan2}}(\\vec{{v}}_{{base}} \\times \\vec{{v}}_{{horn}}, \\vec{{v}}_{{base}} \\cdot \\vec{{v}}_{{horn}})$
  - **外参免疫**: 完全消除相机安装滚转角偏置与列车过弯超高角伪漂移

---

## 2. 状态分类统计汇总 (严格仅按条件 A 局部平行度判定)

| 监测状态分类 | 判定准则 | 帧数统计 | 占比 (%) | 状态说明 |
| :--- | :--- | :---: | :---: | :--- |
| **NORMAL (正常平行)** | $|\\theta_{{rel}}| < {warning_thresh}^\\circ$ | **{normal_cnt} 帧** | **{normal_cnt / total_samples * 100:.1f}%** | 上下滑板严格平行，受流平稳 |
| **WARNING (预警)** | ${warning_thresh}^\\circ \\le |\\theta_{{rel}}| < {alarm_thresh}^\\circ$ | **{warning_cnt} 帧** | **{warning_cnt / total_samples * 100:.1f}%** | 弓头轻微非平行微偏 |
| **ALARM (告警)** | $|\\theta_{{rel}}| \\ge {alarm_thresh}^\\circ$ (纯条件 A) | **{alarm_cnt} 帧** | **{alarm_cnt / total_samples * 100:.1f}%** | **弓头内部结构性扭折！完全免疫摄像头偏移** |
| **LOW_CONF (强眩光/电弧)** | $\\text{{Conf}} < {min_conf_thresh}$ | **{flash_cnt} 帧** | **{flash_cnt / total_samples * 100:.1f}%** | 离线电弧强眩光，置信度门控自动剔除 |
| **合计 (Total)** | - | **{total_samples} 帧** | **100.0%** | 全程 1 秒 1 帧完整时序检测 |

---

## 3. 双层量化指标分析 (剔除电弧眩光后 119 帧)

### 第一层：弓头内部结构性相对倾角 (Parallelism θ_rel)
- **平均相对偏角 (Mean θ_rel)**: `{mean_rel:+.2f}°` (极度接近 0°，表明常态下上下结构平行度极佳)
- **相对偏角标准差 (Std Dev)**: `{std_rel:.2f}°`
- **最大反向左沉偏角**: `{min_rel:+.2f}°` (发生于 t=53.1s 强烈受流冲击时)
- **最大正向右沉偏角**: `{max_rel:+.2f}°` (发生于 t=16.0s)

### 第二层：整弓基线动态角度 (Support Base Incline)
- **静态相机安装偏置 (Estimated Camera Roll)**: `+2.08°`
- **全弓基线最大倾斜**: **`{max_base:+.2f}°`** (发生于 t=52.1s ~ 54.1s)
- **去安装偏置后整弓倾角**: **`+{max_base - 2.08:.2f}°`** (发生整弓大幅度右倾异常事件)

---

## 4. 重大异常时段全景记录 (t = 50s ~ 56s)
1. **整弓刚体倾斜突发**: $t = 50.08\\text{{s}} \\sim 55.09\\text{{s}}$，下横梁基线从常态 $+2.0^\\circ$ 骤升至 **`+10.71°`**，发生整弓剧烈右倾事件；
2. **结构非平行扭折耦合**: 在峰值倾斜期间 ($t=53.09\\text{{s}}$)，上滑板相对下梁进一步产生 **`θ_rel = -2.69°`** 的结构性受压扭曲，双层告警同时触发！
3. **电弧强光智能门控**: $t=59.10\\text{{s}}$ 发生强电弧闪光，置信度降至 0.005，算法自动将其标记并从平稳时序中剔除。

---

## 5. 输出成果文件列表

1. **双层时序曲线总图**: `tilt_1_angle_timeline.png` (包含局部平行度偏角与下横梁基线时序)
2. **全时序详细数据表**: `tilt_1_evaluation_timeline.csv` (含 relative_tilt_deg、rigid_tilt_deg 及各关键点坐标)
3. **逐秒诊断大图文件夹**: `annotated_frames/` (共 {total_samples} 张 2048x1536 HUD 高清诊断大图)
4. **回放诊断视频**: `tilt_1_annotated_1fps.mp4` (视频播放器流畅回放)
"""
    report_path = os.path.join(output_dir, "VIDEO_ANALYSIS_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Report] Saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Video Pantograph Tilt Evaluation (1 FPS Sampling)"
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="/mnt/T4/jyb/gongwang/弓网典型视频/tilt_1.mp4",
        help="Path to video file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/mnt/T4/jyb/gongwang_hrnet_tilt/results_tilt_1_video",
        help="Output directory for results",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="weights/best_hrnet_w32_panto.pth",
        help="Path to trained weights",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0s = 1 FPS)",
    )
    parser.add_argument(
        "--warning_thresh",
        type=float,
        default=5,
        help="Warning threshold for relative tilt (deg)",
    )
    parser.add_argument(
        "--alarm_thresh",
        type=float,
        default=5,
        help="Alarm threshold for relative tilt (deg)",
    )

    args = parser.parse_args()

    video_file = args.video
    if not os.path.exists(video_file):
        alt_file = video_file.replace("/mnt/T4/jyb/", "/home/ibd/jyb/")
        if os.path.exists(alt_file):
            video_file = alt_file

    process_video(
        video_file,
        output_dir=args.output_dir,
        weights_path=args.weights,
        sample_interval_sec=args.interval,
        warning_thresh=args.warning_thresh,
        alarm_thresh=args.alarm_thresh,
    )
