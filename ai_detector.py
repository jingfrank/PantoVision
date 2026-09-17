#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 实时故障检测与告警上报引擎 (ai_detector.py).
采用 6 层插件化分层架构 (Layered Plugin Architecture):
1. 数据接入层 (Ingestion): 支持多路 RTSP 异步重连拉流与离线视频播放
2. 检测调度层 (Scheduler): detectors.yaml 驱动、注册表反射、多频调度与沙箱失败隔离
3. 检测器插件层 (Detectors): 统一 BaseDetector (YOLO-火花/飘弓, FastAD-脏污/异物/结构, HRNet-倾斜)
4. 结果融合层 (Fusion): 跨模块互斥抑制、时序平滑与综合告警裁决
5. 状态与环形缓冲 (Buffer): JPEG 环形队列 + 前后 60s 录像无缝合成
6. 输出层 (Output): 0xFF 0x20 标准二进制报文 + 3+2 OSD 水印与工业 HUD 标注
"""

import os
import sys
import time
import logging
import threading
from typing import Optional, Dict, Any, Set

# 确保本工程根目录与 fastad_core 在 sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
FASTAD_CORE = os.path.join(BASE_DIR, "fastad_core")
if os.path.exists(FASTAD_CORE) and FASTAD_CORE not in sys.path:
    sys.path.append(FASTAD_CORE)

from core.data_types import FramePacket, FusionResult
from core.scheduler import DetectionScheduler
from core.fusion import FusionEngine, FAULT_NAME_MAP
from core.ingestion import RTSPStreamReader, VideoFileReader, RingFrameBuffer
from core.output import UDPAlarmSender, StorageManager, Visualizer, draw_osd_watermark, AsyncNasUploader
from core.perf_monitor import PerformanceMonitor

logger = logging.getLogger("ai_detector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FAULT_CODE_MAP = {
    "big_spark": 3,
    "structure_abnormal": 4,
    "dirty": 10,
    "tilt": 12,
    "foreign_object": 14,
    "abnormal_float": 15,
}


class PantographFaultDetector:
    """
    分层架构受电弓故障检测器服务.
    对 monitor_main.py, phm_service.py 和部署脚本保持 100% 向后兼容.
    """

    def __init__(
        self,
        global_cfg: Dict[str, Any],
        stream_cfg: Dict[str, Any],
        udp_sock: Any,
        stop_event: threading.Event,
        vlm_engine: Optional[Any] = None,
        detectors_yaml_path: str = "config/detectors.yaml"
    ):
        self.global_cfg = global_cfg
        self.stream_cfg = stream_cfg
        self.udp_sock = udp_sock
        self.stop_event = stop_event

        self.hw_cfg = global_cfg.get("hardware_config", {})
        self.vs_cfg = global_cfg.get("video_stream_config", {})
        self.alarm_cfg = global_cfg.get("alarm_push_config", {})

        self.rtsp_url = stream_cfg.get("rtsp_url", "")
        self.camera_id = stream_cfg.get("camera_id", self.hw_cfg.get("camera_id", 0))
        self.car_number = stream_cfg.get("car_number", self.hw_cfg.get("car_number", 6))
        self.result_base_dir = stream_cfg.get("result_base_dir", "./local_cache/Algo-result")
        self.ftp_remote_dir = stream_cfg.get(
            "ftp_remote_dir",
            global_cfg.get("ftp_config", {}).get("monitor_remote_dir", "nvme1/Algo-result")
        )
        self.train_id = stream_cfg.get(
            "train_id",
            global_cfg.get("train_id", self.hw_cfg.get("train_id", "CR400BF"))
        )

        # 1. 初始化检测调度层 (配置驱动 + 注册表自动装配 + 失败隔离)
        config_file = detectors_yaml_path
        if not os.path.isabs(config_file):
            config_file = os.path.join(BASE_DIR, config_file)
        self.scheduler = DetectionScheduler(config_file)

        # 2. 初始化结果融合层
        self.fusion = FusionEngine()

        # 3. 初始化异步 NAS 上传流水线 (Stage 2: 独立上传 Worker + 任务队列)
        self.async_uploader = AsyncNasUploader(
            max_queue_size=20,
            worker_name=f"NasUploadWorker-CAR{self.car_number}-CAM{self.camera_id}"
        )

        # 4. 初始化环形缓冲与异步视频合成流水线 (Stage 1: 独立合成 Worker + 背压保护)
        fps = float(self.vs_cfg.get("fps", 25.0))
        self.ring_buffer = RingFrameBuffer(
            fps=fps,
            pre_record_sec=60,
            post_record_sec=60,
            max_synthesis_queue=5,
            on_video_completed=self._ftp_upload_local_file,
            worker_name=f"RingBuffer-Synth-CAR{self.car_number}-CAM{self.camera_id}"
        )

        # 5. 初始化输出层组件
        udp_ip = self.alarm_cfg.get("udp_ip", "127.0.0.1")
        udp_port = int(self.alarm_cfg.get("udp_port", 8080))
        self.udp_sender = UDPAlarmSender(udp_ip=udp_ip, udp_port=udp_port)
        self.storage = StorageManager(
            base_result_dir=self.result_base_dir,
            pre_sec=60,
            post_sec=60,
            train_id=self.train_id
        )
        self.visualizer = Visualizer()

        # 5. 状态寄存器 (对齐 usrcode.py 与 PHM)
        if "input_parameters" not in self.global_cfg:
            self.global_cfg["input_parameters"] = {}
        self.float_abnormal_cnt = 0
        self.last_frame_ts = 0.0
        self._current_active_fault_types: Set[str] = set()
        self._last_active_fault_types: Set[str] = set()

        # 6. 实时性能监控器
        perf_cfg = global_cfg.get("log_config", {}).get("perf_monitor", {})
        self.perf_monitor = PerformanceMonitor(
            source_fps=float(self.vs_cfg.get("fps", 25.0)),
            report_interval_sec=float(perf_cfg.get("report_interval_sec", 10.0)),
            enabled=bool(perf_cfg.get("enabled", True)),
            logger_name=f"perf_monitor.CAR{self.car_number}.CAM{self.camera_id}",
        )

    # 供 usrcode.py 和 PHM 周期心跳读取与设置的属性映射
    @property
    def spark_alarmed(self) -> bool:
        return "big_spark" in self._current_active_fault_types

    @spark_alarmed.setter
    def spark_alarmed(self, val: bool) -> None:
        if val:
            self._current_active_fault_types.add("big_spark")
        else:
            self._current_active_fault_types.discard("big_spark")

    @property
    def structure_alarmed(self) -> bool:
        return "structure_abnormal" in self._current_active_fault_types

    @structure_alarmed.setter
    def structure_alarmed(self, val: bool) -> None:
        if val:
            self._current_active_fault_types.add("structure_abnormal")
        else:
            self._current_active_fault_types.discard("structure_abnormal")

    @property
    def dirty_alarmed(self) -> bool:
        return "dirty" in self._current_active_fault_types

    @dirty_alarmed.setter
    def dirty_alarmed(self, val: bool) -> None:
        if val:
            self._current_active_fault_types.add("dirty")
        else:
            self._current_active_fault_types.discard("dirty")

    @property
    def tilt_alarmed(self) -> bool:
        return "tilt" in self._current_active_fault_types

    @tilt_alarmed.setter
    def tilt_alarmed(self, val: bool) -> None:
        if val:
            self._current_active_fault_types.add("tilt")
        else:
            self._current_active_fault_types.discard("tilt")

    @property
    def foreign_alarmed(self) -> bool:
        return "foreign_object" in self._current_active_fault_types

    @foreign_alarmed.setter
    def foreign_alarmed(self, val: bool) -> None:
        if val:
            self._current_active_fault_types.add("foreign_object")
        else:
            self._current_active_fault_types.discard("foreign_object")

    @property
    def float_alarmed(self) -> bool:
        return "abnormal_float" in self._current_active_fault_types

    @float_alarmed.setter
    def float_alarmed(self, val: bool) -> None:
        if val:
            self._current_active_fault_types.add("abnormal_float")
        else:
            self._current_active_fault_types.discard("abnormal_float")

    def reset_alarm_states(self) -> None:
        """重置所有故障报警标志与跨帧缓存，防止断流或异常期间旧告警驻留虚警"""
        self._current_active_fault_types.clear()
        self._last_active_fault_types.clear()
        self.float_abnormal_cnt = 0
        if hasattr(self, "fusion"):
            self.fusion.reset()

    def is_streaming_active(self) -> bool:
        """判断当前视频流是否处于正常活跃接收状态（超过 6 秒未收到有效帧判定为流中断）"""
        if self.stop_event.is_set():
            return False
        if getattr(self, "last_frame_ts", 0.0) <= 0:
            return True
        return (time.time() - self.last_frame_ts) <= 6.0

    def detect_arc_height(self, result: Any, is_panto_raised: bool = True) -> bool:
        """
        异常飘弓（abnormal_float）状态门控判断：
        • 正常升弓状态（is_panto_raised = True）：受电弓处于接触网工作高度，属于正常工况，重置飘弓计数器，不触发异常飘弓告警；
        • 降弓状态（is_panto_raised = False）：若视觉检测到受电弓高度超过降弓高度限值（panto_down_height_limit），
          说明受电弓在降弓时发生风动漂浮或机械弹起，连续超标帧数后触发“异常飘弓故障”。
        """
        if is_panto_raised:
            self.float_abnormal_cnt = 0
            self.float_alarmed = False
            return False

        panto_down_height_limit = float(
            self.global_cfg.get("algo_config", {}).get("panto_down_height_limit", 70.0)
        )
        fps = float(self.vs_cfg.get("fps", 25.0))
        common_response_frames = int(
            self.alarm_cfg.get("common_response_time_s", 3) * fps
        )

        panto_boxes = []
        if hasattr(result, "boxes") and result.boxes is not None:
            panto_boxes = [box for box in result.boxes if int(box.cls.item()) == 1]
        elif isinstance(result, (list, tuple)):
            panto_boxes = result

        if panto_boxes:
            if hasattr(panto_boxes[0], "xyxy"):
                box = panto_boxes[0].xyxy[0].tolist()
            else:
                box = panto_boxes[0]
            current_height = float(box[3] - box[1])
            if current_height > panto_down_height_limit:
                self.float_abnormal_cnt += 1
            else:
                self.float_abnormal_cnt = 0
        else:
            self.float_abnormal_cnt = 0

        if self.float_abnormal_cnt >= common_response_frames:
            self.float_alarmed = True
            return True
        else:
            self.float_alarmed = False
            return False

    def _ftp_upload_local_file(self, local_path: str) -> None:
        """异步将抓拍图片或完整视频上传到 NAS，并按策略清理"""
        try:
            from ftp_uploader import get_ftp_uploader

            uploader = get_ftp_uploader()
            if (
                not uploader
                or not uploader.enabled
                or not local_path
                or not os.path.exists(local_path)
            ):
                return

            try:
                rel = os.path.relpath(local_path, self.result_base_dir).replace("\\", "/")
            except ValueError:
                rel = os.path.basename(local_path)

            remote_path = f"{self.ftp_remote_dir.rstrip('/')}/{rel}"
            # 投递至 Stage 2 异步上传流水线 (带背压与重试，解耦网络 I/O)
            submitted = self.async_uploader.submit(local_path, remote_path)
            if submitted:
                logger.info(f"[AI-FTP] 任务已提交上传流水线: {local_path} -> {remote_path}")
        except Exception as e:
            logger.error(f"[AI-FTP] 提交上传流水线异常: {e}")

    def _create_stream_source(self):
        """根据配置自动适配 RTSP 实时网络流或本地测试视频文件"""
        fps = self.vs_cfg.get("fps", 30)
        sample_interval = 1.0 / max(fps, 1)
        hang_timeout = float(self.vs_cfg.get("hang_timeout_sec", 30.0))
        if self.rtsp_url.startswith("rtsp://") or self.rtsp_url.startswith("rtsps://"):
            return RTSPStreamReader(
                rtsp_url=self.rtsp_url,
                car_number=self.car_number,
                camera_id=self.camera_id,
                train_id=self.train_id,
                sample_interval=sample_interval,
                hang_timeout_sec=hang_timeout,
            )
        else:
            local_path = self.rtsp_url
            if not os.path.isabs(local_path):
                local_path = os.path.join(BASE_DIR, local_path)
            return VideoFileReader(
                file_path=local_path,
                car_number=self.car_number,
                camera_id=self.camera_id,
                train_id=self.train_id
            )

    def run(self) -> None:
        """主检测服务运行循环"""
        logger.info(f"[AI-CAR{self.car_number}-CAM{self.camera_id}] 启动分层检测服务, 视频源: {self.rtsp_url}")
        stream_source = self._create_stream_source()
        stream_source.start()

        try:
            consecutive_none_packets = 0
            while not self.stop_event.is_set():
                packet: Optional[FramePacket] = stream_source.get_packet(timeout=1.0)
                if packet is None:
                    if isinstance(stream_source, VideoFileReader):
                        logger.info(f"[AI-CAR{self.car_number}-CAM{self.camera_id}] 离线视频处理完毕")
                        break

                    consecutive_none_packets += 1

                    # 断流期间持续检查是否有已达 60 秒的录像任务需要收尾并投递合成
                    if len(self.ring_buffer.pending_tasks) > 0:
                        completed_videos = self.ring_buffer.finalize_recordings(time.time())
                        if completed_videos:
                            for vid_path in completed_videos:
                                self._ftp_upload_local_file(vid_path)

                    # 若连续 3 秒无有效视频帧（确诊断流），立即复位残留故障状态与跨帧缓存，避免断流后虚假报故障
                    if consecutive_none_packets >= 3 and self._current_active_fault_types:
                        logger.warning(
                            f"[AI-CAR{self.car_number}-CAM{self.camera_id}] "
                            f"视频流中断超过 3 秒，自动清空残留告警状态与融合缓存，恢复正常无故障状态"
                        )
                        self._current_active_fault_types.clear()
                        self._last_active_fault_types.clear()
                        self.fusion.reset()

                    if consecutive_none_packets > 100:
                        consecutive_none_packets = 100
                    continue

                self.last_frame_ts = time.time()
                consecutive_none_packets = 0

                # ── 动态注入 TCMS/PHM 融合平台实时下发的输入变量 ─────────────────
                # 参考《网络通信协议-融合主机相关.xls》与 AppConfig.xlsx 定义：
                # 序号 0: 列车速度 (Unsigned16, 比例 0.01 km/h/bit)
                # 序号 1~7: 年、月、日、时、分、秒、时间有效 (TCMS 物理时标)
                # 序号 8: TP06升弓信号 (Boolean1, 1=升弓, 0=降弓)
                # 序号 9: 公里标（HH/HL/LH/LL）(Unsigned32, 单位: 米, 0xFFFFFFFF 为未知)
                input_params = self.global_cfg.get("input_parameters", {})

                # 1. 升降弓状态信号注入 (序号 8: TP06升弓信号；兼容旧序号 3: wlpantoup06)
                if 8 in input_params:
                    packet.is_panto_raised = bool(input_params[8])
                elif 3 in input_params:
                    packet.is_panto_raised = bool(input_params[3])

                # 2. 列车速度注入 (序号 0: 列车速度, 比例 0.01 km/h/bit；兼容旧序号 2)
                if 0 in input_params:
                    try:
                        raw_spd = float(input_params[0])
                        packet.speed = raw_spd * 0.01 if raw_spd > 655.35 else raw_spd
                    except (ValueError, TypeError):
                        pass
                elif 2 in input_params:
                    try:
                        packet.speed = float(input_params[2])
                    except (ValueError, TypeError):
                        pass

                # 3. 公里标注入 (序号 9: 公里标, 单位: 米；兼容旧序号 1)
                if 9 in input_params:
                    kp = input_params[9]
                    if kp is not None and kp != 0xFFFFFFFF:
                        packet.kilometer_post = int(kp)
                elif 1 in input_params:
                    kp = input_params[1]
                    if kp is not None:
                        try:
                            packet.kilometer_post = int(kp)
                        except (ValueError, TypeError):
                            pass

                # 4. TCMS 时间戳对齐 (序号 7 为时间有效信号，序号 1~6 为年月日时分秒)
                if input_params.get(7) == 1:
                    try:
                        import datetime
                        yr = int(input_params.get(1, 0))
                        yr = (2000 + yr) if yr < 100 else yr
                        mo = int(input_params.get(2, 1))
                        dy = int(input_params.get(3, 1))
                        hr = int(input_params.get(4, 0))
                        mn = int(input_params.get(5, 0))
                        sc = int(input_params.get(6, 0))
                        dt = datetime.datetime(yr, mo, dy, hr, mn, sc)
                        packet.timestamp = dt.timestamp()
                    except Exception:
                        pass

                # ── 帧处理端到端计时开始 ──────────────────────────────────
                t_frame_start = time.perf_counter()

                # 1. 调度层：各检测器插件沙箱安全推理
                raw_results = self.scheduler.dispatch(packet)

                # 2. 融合层：跨模块抑制、时序平滑与仲裁
                fusion_res: FusionResult = self.fusion.decide(raw_results, packet)

                # 3. 更新当前生效的告警状态
                self._current_active_fault_types = {
                    alarm["fault_type"] for alarm in fusion_res.active_alarms
                }

                # 4. 边沿触发检测 (仅在故障首次由 0 -> 1 时触发抓拍落盘与报文发射)
                newly_triggered = self._current_active_fault_types - self._last_active_fault_types

                hud_frame = None
                if fusion_res.has_fault or newly_triggered or len(self.ring_buffer.pending_tasks) > 0:
                    hud_frame = self.visualizer.render_hud(packet.frame, fusion_res, packet)

                if newly_triggered:
                    for f_type in newly_triggered:
                        chinese_name = FAULT_NAME_MAP.get(f_type, f_type)
                        fault_code = FAULT_CODE_MAP.get(f_type, 0)
                        cfg_level = self.alarm_cfg.get("fault_levels", {}).get(chinese_name, 1)

                        # (a) 生成车载规范化基名与独立事件目录
                        base_name, save_dir = self.storage.generate_base_name(packet.timestamp, fault_code)

                        # (b) 保存故障时刻高质量抓拍并异步上传 NAS
                        pic_path = self.storage.save_snapshot(
                            frame=hud_frame if hud_frame is not None else packet.frame,
                            timestamp=packet.timestamp,
                            fault_code=fault_code,
                            base_name=base_name,
                            save_dir=save_dir
                        )
                        self._ftp_upload_local_file(pic_path)

                        # (c) 发射 0xFF 0x20 车端标准二进制报文 (携带规范化图片绝对路径)
                        self.udp_sender.send_alarm(
                            event_name=chinese_name,
                            car_number=self.car_number,
                            camera_id=self.camera_id,
                            picture_path=pic_path,
                            level=cfg_level,
                            fault_code=fault_code
                        )

                        # (d) 触发前后 60 秒录像收集任务
                        self.ring_buffer.trigger_recording(
                            event_name=chinese_name,
                            fault_code=fault_code,
                            fault_ts=packet.timestamp,
                            save_dir=save_dir,
                            base_name=base_name
                        )

                self._last_active_fault_types = set(self._current_active_fault_types)

                # 5. 环形缓冲：收集后60秒帧与定期合成完整视频并上传
                cache_frame = hud_frame if hud_frame is not None else packet.frame
                self.ring_buffer.append(packet.timestamp, cache_frame)
                self.ring_buffer.collect_post_frame(packet.timestamp, cache_frame)
                completed_videos = self.ring_buffer.finalize_recordings(packet.timestamp)
                if completed_videos:
                    for vid_path in completed_videos:
                        self._ftp_upload_local_file(vid_path)

                # 6. 性能监控 tick（在本帧所有处理完成后计算端到端耗时）
                wall_ms = (time.perf_counter() - t_frame_start) * 1000.0
                queue_depth = 0
                if hasattr(stream_source, "frame_queue"):
                    try:
                        queue_depth = stream_source.frame_queue.qsize()
                    except Exception:
                        pass
                self.perf_monitor.tick(packet, raw_results, wall_ms, queue_depth)

        except Exception as e:
            logger.error(f"[AI-CAR{self.car_number}-CAM{self.camera_id}] 主循环异常: {e}", exc_info=True)
        finally:
            stream_source.stop()
            self.ring_buffer.stop()
            self.async_uploader.stop()
            self.udp_sender.close()
            logger.info(f"[AI-CAR{self.car_number}-CAM{self.camera_id}] 检测服务已安全停止")
