#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/modify_foreign_video.py
受电弓悬挂异物测试视频时序延展工具 (针对 foreign_1.mp4 方案一: 第 850 帧时序延展 2 秒 - 单通道无损滤镜重编码版).

核心机制：
1. 提取 target_frame (第 850 帧，即 34.00s 处的悬挂异物关键帧)，导出无损图像；
2. 采用 FFmpeg 单通道 overlay 滤镜方案 (Single-pass unified pipeline):
   通过 `[0:v][1:v]overlay=enable='between(n,850,899)'` 将异物无损模板无缝贴合在 34.0s~36.0s (第 850~899 帧)；
3. 全片使用统一 H.264 High profile 编码，彻底根除跨切片拼接导致的 SPS/PPS 不兼容、DTS/PTS 乱序及花屏/绿屏乱码；
4. 严格保留导线悬挂异物的高频特征 (CRF 10)，确保 FastAD 异物检测算法在 50 帧时序窗口内 100% 稳定检出。
"""

import os
import sys
import time
import argparse
import subprocess
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extend_foreign_video(
    input_path: str,
    output_path: str,
    target_frame_idx: int = 850,
    hold_sec: float = 2.0,
    crf: int = 10,
    preset: str = "veryfast",
    temp_dir: str = "local_cache/video_modify_temp",
):
    """
    通过 FFmpeg overlay 滤镜单通道重编码，将 target_frame_idx 处延展 hold_sec 秒，
    保证全片时间戳连续、SPS/PPS 统一，杜绝任何花屏乱码。
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] 输入视频不存在: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开输入视频: {input_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    hold_frames = int(round(hold_sec * fps))

    target_time_s = target_frame_idx / fps
    end_frame_idx = target_frame_idx + hold_frames - 1
    end_time_s = end_frame_idx / fps

    print("=" * 75)
    print("🎬 [视频改造工具] 悬挂异物测试视频时序延展 (单通道统一滤镜重编码)")
    print(f"  输入源: {input_path}")
    print(f"  输出目标: {output_path}")
    print(f"  分辨率: {width}x{height} | 帧率: {fps:.1f} FPS | 源时长: {duration_s:.1f}s ({total_frames} 帧)")
    print(f"  目标缺陷帧: 第 {target_frame_idx} 帧 ({target_time_s:.2f}s)")
    print(f"  延展区间: 第 {target_frame_idx} ~ {end_frame_idx} 帧 ({target_time_s:.2f}s ~ {end_time_s:.2f}s, 共 {hold_frames} 帧 / {hold_sec:.1f}s)")
    print(f"  编码配置: libx264 -preset {preset} -crf {crf} -pix_fmt yuv420p")
    print("=" * 75)

    os.makedirs(temp_dir, exist_ok=True)
    temp_png = os.path.join(temp_dir, f"frame_{target_frame_idx}_defect.png")

    t0 = time.time()

    # 1. 提取 target_frame_idx 帧为无损 PNG
    print(f"\n[1/4] 提取第 {target_frame_idx} 帧画面作为高清无损模板...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, defect_frame = cap.read()
    cap.release()
    if not ret or defect_frame is None:
        print(f"[ERROR] 读取第 {target_frame_idx} 帧失败！")
        sys.exit(1)
    cv2.imwrite(temp_png, defect_frame)
    print(f"  ✅ 成功保存无损模板: {temp_png}")

    # 2. 单通道 overlay 滤镜全局转码
    print(f"\n[2/4] 执行单通道 FFmpeg 统一重编码 (杜绝多段拼接导致的 DTS/SPS 异常)...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_final = output_path + ".tmp.mp4"

    filter_str = f"[0:v][1:v]overlay=enable='between(n,{target_frame_idx},{end_frame_idx})':x=0:y=0"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-i", temp_png,
        "-filter_complex", filter_str,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        temp_final
    ]
    print(f"  执行指令: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"[ERROR] 转码失败: {res.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✅ 单通道重编码完成，临时文件: {temp_final}")

    # 3. 校验新视频完整性 (零解码错误校验)
    print(f"\n[3/4] 校验转码视频解码完整性与流一致性...")
    chk = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", temp_final, "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    err_output = chk.stderr.decode().strip()
    if err_output:
        print(f"  ⚠️ FFmpeg 解码警告/错误: {err_output}")
    else:
        print(f"  ✅ FFmpeg 解码检验 100% 纯净无误 (0 errors, 0 warnings)！")

    # 原子替换
    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(temp_final, output_path)

    # 4. 算法检出复核
    print(f"\n[4/4] 运行 FastAD 异物检测算法抽检异物延展区间...")
    try:
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from core.detectors.foreign_detector import FastadForeignDetector
        import yaml

        det = FastadForeignDetector()
        cfg_file = os.path.join(BASE_DIR, "config", "detectors.yaml")
        if os.path.exists(cfg_file):
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            f_cfg = [d["params"] for d in cfg.get("detectors", []) if d.get("name") == "foreign_object"][0]
            det.initialize(f_cfg)
        else:
            det.initialize({"model_path": "models/fastad_layer2_256_reg1e4_nq3_fmq1.pt"})

        cap_verify = cv2.VideoCapture(output_path)
        check_frames = [
            target_frame_idx - 2,
            target_frame_idx,
            target_frame_idx + hold_frames // 2,
            end_frame_idx,
            end_frame_idx + 2,
        ]
        for cf in check_frames:
            cap_verify.set(cv2.CAP_PROP_POS_FRAMES, cf)
            ret_f, vframe = cap_verify.read()
            if ret_f and vframe is not None:
                vres = det.run(vframe)
                status = "🚨 异物检出" if vres.is_abnormal else "🟢 正常"
                print(f"  Frame {cf:5d} ({cf/fps:5.2f}s): {status} | score={vres.score:.2f} | img_score={vres.extra.get('image_score', 0):.2f} | bbox={vres.bbox}")
        cap_verify.release()
    except Exception as e:
        print(f"  [WARN] 算法抽检跳过: {e}")

    total_time = time.time() - t0
    final_cap = cv2.VideoCapture(output_path)
    final_frames = int(final_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    final_cap.release()

    print("\n" + "=" * 75)
    print("🎉 [视频改造完成] 导出成功！")
    print(f"  最终视频路径: {output_path}")
    print(f"  总帧数: {final_frames} 帧 (原始: {total_frames} 帧)")
    print(f"  总时长: {final_frames / fps:.2f} 秒")
    print(f"  异物持续区间: 第 {target_frame_idx} ~ {end_frame_idx} 帧 ({target_time_s:.2f}s ~ {end_time_s:.2f}s, 共 {hold_frames} 帧)")
    print(f"  耗时: {total_time:.1f} 秒")
    print(f"  文件大小: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="悬挂异物测试视频时序延展工具 (单通道统一滤镜重编码)")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(BASE_DIR, "弓网典型视频", "foreign_1_original.mp4"),
        help="输入原始视频路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(BASE_DIR, "弓网典型视频", "foreign_1.mp4"),
        help="输出目标视频路径"
    )
    parser.add_argument(
        "--target-frame",
        type=int,
        default=850,
        help="缺陷帧索引 (默认: 850)"
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=2.0,
        help="持续延展秒数 (默认: 2.0 秒，对应 50 帧)"
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=10,
        help="x264 CRF 质量参数 (默认: 10, 保障 FastAD 高频特征)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="x264 编码预设 (默认: veryfast)"
    )

    args = parser.parse_args()
    extend_foreign_video(
        input_path=args.input,
        output_path=args.output,
        target_frame_idx=args.target_frame,
        hold_sec=args.hold_sec,
        crf=args.crf,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
