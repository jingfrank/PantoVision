#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_alarm_verifier.py — 受电弓故障报警 Qwen 大模型二级复核引擎

核心功能：
1. 专为大火花 (spark) 与受电弓结构异常 (structural_anomaly) 设计的二级高精度复核防线；
2. 注入针对铁路现场典型伪影的显式负样本物理约束（排除站台探照灯、路灯、金属反光拉丝、螺栓接头、阳光阴影）；
3. 注入反先验去偏指令，免疫工控初检 HUD 标记红框与告警文字的干扰；
4. 保留原图微观放电细节，利用 Qwen3-VL 动态分辨率切片（Dynamic Patching）进行无损精准判定；
5. 提供三种调用形态：
   - 类导入调用：在 ai_detector.py 中直接 verify_image(pic_path, fault_type) 判定并拦截 UDP；
   - 独立守护进程：实时监听板子报警图片落盘目录，自动对新抓拍图片执行二次核验并沉淀 JSON/CSV；
   - 离线批量/单图测试：方便随时在终端运行测试命令。
"""

import os
import sys
import time
import glob
import json
import re
import csv
import logging
import argparse
import threading
from typing import Optional, Dict, Any, Union, Tuple
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [QwenVerifier] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("QwenVerifier")

# 默认模型路径优先级搜索
DEFAULT_MODEL_CANDIDATES = [
    "./models/Qwen3-VL-4B-Instruct",
    "/mnt/T4/jyb/gongwang/models/Qwen3-VL-4B-Instruct",
    "/home/ibd/jyb/gongwang/models/Qwen3-VL-4B-Instruct",
    "/home/ibd/jyb/gongwang_jetson_vllm/models/Qwen3-VL-4B-Instruct",
]

# 严格对齐基准评测验证过的专业复核 Prompt
PROMPT_VERIFY = """你是一个专业的铁路受电弓与接触网故障监测及智能复核安全专家。
请仔细观察图像中受电弓滑板、弓角、支架结构与接触网接触状态。
【注意】：图像上可能叠加有工控系统的初检标记框或顶部告警文字。请你作为独立复核专家，严格基于受电弓本身的物理形态进行真实性研判，甄别出算法可能存在的误报：

1. 【大火花(spark)】:
   - 真实火花判定标准：受电弓碳滑板与接触线之间出现强烈的放电电弧、大面积刺眼白光或蓝色爆燃辉光。
   - 排除误报约束：外界站台照明大灯、夜间路灯、反光镜、车顶金属高反光拉丝绝不是火花！正常运行微弱蓝白小星火不算大火花。

2. 【受电弓结构异常(structural_anomaly)】:
   - 真实结构异常标准：弓角(羊角)明显折断缺失、碳滑板严重剥落碎裂变形、上导杆断脱、金属支架框架扭曲断裂。
   - 排除误报约束：受电弓正常的金属螺栓、固定线夹、卡扣、接头属于标准机械部件，绝不是异常或断裂！因阳光斜射产生的局部阴影或反光亮斑绝不是结构异常！

请严格按以下 JSON 格式输出判定结果，不要输出任何多余废话：
```json
{
  "reasoning": "50字以内的专业视觉研判依据",
  "spark": 0或1,
  "structural_anomaly": 0或1
}
```
"""


class QwenAlarmVerifier:
    """
    Qwen 大模型受电弓故障二级复核器
    支持线程安全并发调用，自动管理 GPU 显存配额与异常重试。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        gpu_mem_fraction: float = 0.80,
        enable_logging: bool = True
    ):
        self.device = device
        self.enable_logging = enable_logging
        self.lock = threading.Lock()

        # 1. 寻找有效模型路径
        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"正在初始化 Qwen 二级复核引擎, 模型目录: {self.model_path}")

        # 2. 导入深度学习依赖
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            self.torch = torch
            self.AutoProcessor = AutoProcessor
            self.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
        except ImportError as e:
            logger.error(f"导入 PyTorch 或 Transformers 失败: {e}. 请确保运行环境满足要求！")
            raise

        # 3. 显存配额保护
        if self.torch.cuda.is_available() and "cuda" in self.device:
            try:
                dev_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
                self.torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=dev_idx)
                logger.info(f"GPU 显存配额限制已设定: {gpu_mem_fraction * 100:.0f}%")
            except Exception as e:
                logger.warning(f"设置 GPU 显存配额失败 (非致命): {e}")

        # 4. 加载 Processor 与 Model (FP16)
        t0 = time.time()
        logger.info("正在加载模型权重与预处理器，请稍候...")
        self.processor = self.AutoProcessor.from_pretrained(self.model_path)
        self.model = self.Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True
        ).eval()

        # 预编译通用 Prompt 文本输入模板
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT_VERIFY}
                ]
            }
        ]
        self.compiled_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        t_cost = time.time() - t0
        logger.info(f"✅ Qwen3-VL-4B 二级复核引擎就绪！加载耗时: {t_cost:.2f} 秒")

    def _resolve_model_path(self, path: Optional[str]) -> str:
        if path and os.path.exists(path):
            return os.path.abspath(path)
        for cand in DEFAULT_MODEL_CANDIDATES:
            if os.path.exists(cand):
                return os.path.abspath(cand)
        raise FileNotFoundError(
            f"未找到 Qwen3-VL 模型权重！候选搜索路径均不存在: {DEFAULT_MODEL_CANDIDATES}。请指定正确的 --model-path"
        )

    def _parse_model_output(self, out_text: str) -> Tuple[str, int, int]:
        """解析模型输出中的 JSON 结构，并提供鲁棒的降级容错提取"""
        reasoning = ""
        spark = 0
        structural_anomaly = 0

        json_match = re.search(r'\{[^{}]*"reasoning"[^{}]*\}', out_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{[^{}]*"spark"[^{}]*\}', out_text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(0))
                reasoning = str(data.get("reasoning", "")).strip()
                spark = 1 if int(data.get("spark", 0)) > 0 else 0
                structural_anomaly = 1 if int(data.get("structural_anomaly", 0)) > 0 else 0
                return reasoning, spark, structural_anomaly
            except Exception:
                pass

        # 降级容错提取
        if re.search(r'"spark"\s*:\s*1', out_text):
            spark = 1
        if re.search(r'"structural_anomaly"\s*:\s*1', out_text):
            structural_anomaly = 1

        r_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', out_text)
        if r_match:
            reasoning = r_match.group(1).strip()
        else:
            reasoning = out_text[:120].replace("\n", " ").strip()

        return reasoning, spark, structural_anomaly

    def verify_image(
        self,
        image_input: Union[str, Image.Image, Any],
        expected_fault_type: Optional[Union[str, int]] = None
    ) -> Dict[str, Any]:
        """
        对单张告警抓拍图像进行专业深度复核。
        
        参数:
            image_input: 图片路径 (str)、PIL Image 对象或 OpenCV BGR ndarray
            expected_fault_type: 工控初检告警类别 (可选):
                - "big_spark" 或 3
                - "structure_abnormal" 或 4
                - None 或其他 (将返回全量研判)
        
        返回 dict 结构:
            {
                "is_alarm_confirmed": bool,   # 大模型是否确认该报警成立 (True=确认真实报警; False=裁定为虚警)
                "verdict": str,               # 易读的定性结论
                "spark": int,                 # 0 或 1
                "structural_anomaly": int,    # 0 或 1
                "reasoning": str,             # 50字专业视觉研判依据
                "latency_ms": float,          # 推理耗时 (毫秒)
                "raw_output": str             # 模型原始输出文本
            }
        """
        # 1. 规范化为 PIL.Image
        pil_img = None
        img_desc = ""
        if isinstance(image_input, str):
            img_desc = image_input
            if not os.path.exists(image_input):
                return {
                    "is_alarm_confirmed": False,
                    "verdict": f"FILE_NOT_FOUND: {image_input}",
                    "spark": 0, "structural_anomaly": 0,
                    "reasoning": "图片文件不存在", "latency_ms": 0.0, "raw_output": ""
                }
            try:
                pil_img = Image.open(image_input).convert("RGB")
            except Exception as e:
                return {
                    "is_alarm_confirmed": False,
                    "verdict": f"IMAGE_DECODE_ERROR: {e}",
                    "spark": 0, "structural_anomaly": 0,
                    "reasoning": f"图片解码失败: {e}", "latency_ms": 0.0, "raw_output": ""
                }
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            img_desc = "PIL.Image"
        elif hasattr(image_input, "shape"):  # numpy / opencv BGR
            import cv2
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            img_desc = f"ndarray_{pil_img.size}"
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_input)}")

        # 2. 规范化初检故障类型
        norm_type = self._normalize_fault_type(expected_fault_type, img_desc)

        # 3. 线程安全推理
        t0 = time.time()
        with self.lock:
            with self.torch.no_grad():
                inputs = self.processor(
                    text=[self.compiled_prompt],
                    images=[pil_img],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

        latency_ms = (time.time() - t0) * 1000.0
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, prompt_len:]
        out_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

        # 4. 提取与裁决
        reasoning, spark_val, struct_val = self._parse_model_output(out_text)

        is_confirmed = False
        verdict = ""

        if norm_type == "big_spark":
            if spark_val == 1:
                is_confirmed = True
                verdict = "CONFIRMED_SPARK (确认大火花)"
            else:
                is_confirmed = False
                verdict = "SUPPRESSED_SPARK_FALSE_ALARM (大火花虚警已剔除)"

        elif norm_type == "structure_abnormal":
            if struct_val == 1:
                is_confirmed = True
                verdict = "CONFIRMED_STRUCT (确认结构异常)"
            else:
                is_confirmed = False
                verdict = "SUPPRESSED_STRUCT_FALSE_ALARM (结构异常虚警已剔除)"

        else:
            # 未指定类型或非这两种类型（如对照样本）
            is_confirmed = bool(spark_val or struct_val)
            findings = []
            if spark_val == 1:
                findings.append("大火花")
            if struct_val == 1:
                findings.append("结构异常")
            verdict = f"DETECTION: {';'.join(findings)}" if findings else "NORMAL (无大火花或结构异常)"

        res = {
            "is_alarm_confirmed": is_confirmed,
            "verdict": verdict,
            "target_fault_type": norm_type,
            "spark": spark_val,
            "structural_anomaly": struct_val,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 1),
            "raw_output": out_text
        }

        if self.enable_logging:
            symbol = "🔥" if is_confirmed else "🛡️"
            logger.info(
                f"{symbol} [{verdict}] 耗时 {latency_ms:.0f}ms | 研判依据: {reasoning}"
            )

        return res

    def _normalize_fault_type(self, raw_type: Any, filename: str = "") -> str:
        """从字符串、数字代码或文件名识别故障类别"""
        if raw_type in [3, "3", "big_spark", "spark", "大火花"]:
            return "big_spark"
        if raw_type in [4, "4", "structure_abnormal", "structure", "受电弓结构异常"]:
            return "structure_abnormal"

        fn = os.path.basename(str(filename)).lower()
        if "pcm_algo_result-3" in fn or "spark" in fn or "火花" in fn:
            return "big_spark"
        if "pcm_algo_result-4" in fn or "structure" in fn or "结构" in fn:
            return "structure_abnormal"

        return "unknown"


# ==============================================================================
# 目录实时监听守护服务 (Watchdog Mode)
# ==============================================================================

class AlarmDirWatcher:
    """
    报警图片目录监听器:
    定期扫描指定结果目录，发现新落盘的 `-3.jpg` (火花) 和 `-4.jpg` (结构) 抓拍图片，
    并自动调度 Qwen 二级复核引擎进行研判与结果归档。
    """

    def __init__(
        self,
        watch_dir: str,
        verifier: QwenAlarmVerifier,
        poll_interval: float = 2.0,
        output_csv: Optional[str] = None,
        suppress_action: str = "mark"  # "mark", "move", "none"
    ):
        self.watch_dir = os.path.abspath(watch_dir)
        self.verifier = verifier
        self.poll_interval = poll_interval
        self.suppress_action = suppress_action
        self.processed_records_file = os.path.join(self.watch_dir, ".qwen_processed_cache.json")
        self.processed_files = self._load_processed_cache()
        self.output_csv = output_csv or os.path.join(self.watch_dir, "qwen_verification_history.csv")
        self._init_csv()

    def _load_processed_cache(self) -> set:
        if os.path.exists(self.processed_records_file):
            try:
                with open(self.processed_records_file, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def _save_processed_cache(self):
        try:
            with open(self.processed_records_file, "w", encoding="utf-8") as f:
                json.dump(list(self.processed_files), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存处理缓存失败: {e}")

    def _init_csv(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "image_path", "target_fault", "is_confirmed",
                    "verdict", "spark", "structural_anomaly", "reasoning", "latency_ms"
                ])

    def _append_csv(self, img_path: str, res: Dict[str, Any]):
        try:
            with open(self.output_csv, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    img_path,
                    res.get("target_fault_type", ""),
                    res.get("is_alarm_confirmed", False),
                    res.get("verdict", ""),
                    res.get("spark", 0),
                    res.get("structural_anomaly", 0),
                    res.get("reasoning", ""),
                    res.get("latency_ms", 0)
                ])
        except Exception as e:
            logger.error(f"写入 CSV 失败: {e}")

    def scan_and_verify(self):
        """扫描一次目录并复核所有未处理的新文件"""
        if not os.path.exists(self.watch_dir):
            return

        # 扫描所有符合车载规范的 jpg 抓拍图
        patterns = [
            os.path.join(self.watch_dir, "**", "*PCM_Algo_result-3*.jpg"),
            os.path.join(self.watch_dir, "**", "*PCM_Algo_result-4*.jpg"),
            os.path.join(self.watch_dir, "**", "*big_spark*.jpg"),
            os.path.join(self.watch_dir, "**", "*structure_abnormal*.jpg"),
        ]

        found_imgs = []
        for pat in patterns:
            found_imgs.extend(glob.glob(pat, recursive=True))

        for img_path in sorted(list(set(found_imgs))):
            abs_path = os.path.abspath(img_path)
            if abs_path in self.processed_files:
                continue

            # 避开临时写入未完成的文件 (文件大小检查)
            try:
                s1 = os.path.getsize(abs_path)
                time.sleep(0.1)
                s2 = os.path.getsize(abs_path)
                if s1 != s2 or s1 == 0:
                    continue  # 文件仍在写入中
            except OSError:
                continue

            logger.info(f"🔍 捕获到新报警抓拍图，开始 Qwen 二级复核: {abs_path}")
            res = self.verifier.verify_image(abs_path)

            # 在抓拍图同级目录下保存结构化复核元数据
            meta_path = os.path.splitext(abs_path)[0] + "_qwen_verdict.json"
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"保存元数据失败 {meta_path}: {e}")

            # 写入全局 CSV 汇总
            self._append_csv(abs_path, res)

            # 虚警处理动作
            if not res["is_alarm_confirmed"] and self.suppress_action == "mark":
                logger.info(f"🛡️ [虚警拦截归档] 标记图片: {os.path.basename(abs_path)}")

            self.processed_files.add(abs_path)
            self._save_processed_cache()

    def run_forever(self):
        """持续后台监听运行"""
        logger.info(f"🚀 启动报警目录实时监听服务, 监听目录: {self.watch_dir}")
        logger.info(f"   • 轮询间隔: {self.poll_interval}s")
        logger.info(f"   • 汇总记录 CSV: {self.output_csv}")
        try:
            while True:
                self.scan_and_verify()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("收到退出信号，停止监听服务。")


# ==============================================================================
# CLI 入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen 大模型受电弓大火花与结构异常二级复核引擎")
    parser.add_argument("--model-path", type=str, default=None, help="Qwen3-VL-4B 模型目录路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备 (如 cuda:0)")
    parser.add_argument("--watch-dir", type=str, default=None, help="启用实时目录监听模式，传入告警图片根目录")
    parser.add_argument("--scan-dir", type=str, default=None, help="一次性离线扫描并复核指定目录内全量图片")
    parser.add_argument("--image", type=str, default=None, help="单张图片测试路径")
    parser.add_argument("--fault-type", type=str, default=None, help="故障类别: big_spark / structure_abnormal")
    parser.add_argument("--output-csv", type=str, default=None, help="汇总 CSV 记录路径")
    parser.add_argument("--interval", type=float, default=2.0, help="监听模式轮询周期 (秒)")
    args = parser.parse_args()

    # 初始化引擎
    verifier = QwenAlarmVerifier(model_path=args.model_path, device=args.device)

    # 1. 单图测试模式
    if args.image:
        res = verifier.verify_image(args.image, expected_fault_type=args.fault_type)
        print("\n" + "=" * 60)
        print("📊 [单图核验详细结果]")
        print(f"  • 输入图像: {args.image}")
        print(f"  • 复核结论: {res['verdict']}")
        print(f"  • 报警确认: {res['is_alarm_confirmed']}")
        print(f"  • 大火花: {res['spark']} | 结构异常: {res['structural_anomaly']}")
        print(f"  • 研判依据: {res['reasoning']}")
        print(f"  • 推理耗时: {res['latency_ms']} ms")
        print("=" * 60 + "\n")
        return

    # 2. 实时目录监听守护模式
    if args.watch_dir:
        watcher = AlarmDirWatcher(
            watch_dir=args.watch_dir,
            verifier=verifier,
            poll_interval=args.interval,
            output_csv=args.output_csv
        )
        watcher.run_forever()
        return

    # 3. 一次性离线扫描模式
    if args.scan_dir:
        watcher = AlarmDirWatcher(
            watch_dir=args.scan_dir,
            verifier=verifier,
            output_csv=args.output_csv
        )
        logger.info(f"开始一次性离线扫描: {args.scan_dir}")
        watcher.scan_and_verify()
        logger.info("离线扫描完成。")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
