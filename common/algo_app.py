"""UDP 算法应用基类：心跳上报与结果上报。"""

import os
import socket
import struct
import threading
import time

from .config import load_app_config, resolve_log_root
from .protocol import (
    HEADER_FORMAT,
    HEADER_SIZE,
    HOST_SOCKET_ADDR_DEFAULT,
    PROG_ALGO_DATA,
    PROG_ALGO_STOP,
    PROG_ALGO_STAT,
    RECV_BUFFER_SIZE,
    SEND_INTERVAL_SECONDS,
    STAT_DATA_LEN,
    STAT_NORMAL,
    STAT_OFFLINE,
    STAT_TIMEOUT,
    STATUS_FORMAT,
    build_result_packet,
    format_status_packet,
    format_result_packet,
    pack_message_header,
)


class RunLog:
    """运行日志记录器（线程安全）。

    日志路径: {log_root}/{algo_id}/algo_run.log
      板卡: /hirain/hi_opt/algo_log/{algo_id}/algo_run.log
      开发机: local_cache/logs/{algo_id}/algo_run.log

    超过 MAX_ENTRIES 条时自动减半（保留后半段），写失败不影响算法主逻辑。
    """

    MAX_ENTRIES = 10000

    def __init__(self, algo_id: int, enabled: bool = True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.path = None
        self.entries = 0
        if not enabled:
            return
        try:
            log_dir = os.path.join(resolve_log_root(), str(algo_id))
            os.makedirs(log_dir, exist_ok=True)
            self.path = os.path.join(log_dir, "algo_run.log")
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8-sig", errors="ignore") as f:
                    self.entries = sum(1 for _ in f)
        except OSError:
            self.path = None

    def log(self, message: str) -> None:
        """追加一条带时间戳的日志记录。"""
        if not self.enabled or not self.path:
            return
        try:
            with self.lock:
                with open(self.path, "a", encoding="utf-8-sig") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
                self.entries += 1
                if self.entries > self.MAX_ENTRIES:
                    self._trim()
        except OSError:
            pass

    def _trim(self) -> None:
        """超限后保留后半段，防止日志文件无限增长。"""
        try:
            with open(self.path, "r", encoding="utf-8-sig", errors="ignore") as f:
                lines = f.readlines()
            keep = lines[-(self.MAX_ENTRIES // 2):]
            with open(self.path, "w", encoding="utf-8-sig") as f:
                f.writelines(keep)
            self.entries = len(keep)
        except OSError:
            pass


class AlgoUdpAppBase:
    """UDP 算法服务公共基类。"""

    def __init__(
        self,
        algo_id,
        algo_port,
        app_config_path,
        host_socket_addr=HOST_SOCKET_ADDR_DEFAULT,
        *,
        recv_timeout=SEND_INTERVAL_SECONDS,
        log_status=False,
        log_enabled=True,
        shared_cfg=None,
    ):
        algo_id, algo_version, _ = load_app_config(app_config_path, algo_id)
        self.log_status = log_status
        self.run_log = RunLog(algo_id, enabled=log_enabled)
        self.shared_cfg = shared_cfg if shared_cfg is not None else {}
        if "input_parameters" not in self.shared_cfg:
            self.shared_cfg["input_parameters"] = {}

        self.stat_list = [algo_id, algo_version, os.getpid(), 0, STAT_NORMAL]
        self.running = True
        self.result_sequence = 0

        self.local_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self.local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
        self.local_socket.bind(("0.0.0.0", algo_port))
        self.local_socket.settimeout(recv_timeout)
        self.stat_list[3] = self.local_socket.getsockname()[1]

        self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if host_socket_addr and isinstance(host_socket_addr[0], (list, tuple)):
            self.host_socket_addrs = [tuple(addr) for addr in host_socket_addr]
        else:
            self.host_socket_addrs = [tuple(host_socket_addr)]
        self.host_socket_addr = self.host_socket_addrs[0]

    def _send_to_hosts(self, packet):
        for addr in self.host_socket_addrs:
            try:
                self.host_socket.sendto(packet, addr)
            except socket.error as e:
                print(f"[PHM] 发送到 {addr[0]}:{addr[1]} 失败: {e}")

    def report_status_loop(self):
        packet_header = pack_message_header(PROG_ALGO_STAT, STAT_DATA_LEN)

        while self.running:
            try:
                status_payload = struct.pack(STATUS_FORMAT, *self.stat_list)
                packet = packet_header + status_payload
                self._send_to_hosts(packet)

                if self.log_status:
                    targets = ", ".join(f"{addr[0]}:{addr[1]}" for addr in self.host_socket_addrs)
                    print(f"{format_status_packet(packet)}\n  目标: {targets}", flush=True)
            except socket.error as e:
                print(f"[PHM] 发送心跳失败: {e}")

            time.sleep(SEND_INTERVAL_SECONDS)

    def send_offline_status(self):
        self.stat_list[4] = STAT_OFFLINE
        packet_header = pack_message_header(PROG_ALGO_STAT, STAT_DATA_LEN)
        status_payload = struct.pack(STATUS_FORMAT, *self.stat_list)
        packet = packet_header + status_payload
        self._send_to_hosts(packet)
        if self.log_status:
            print(format_status_packet(packet), flush=True)

    def close(self):
        self.running = False
        try:
            self.local_socket.close()
        except OSError:
            pass
        try:
            self.host_socket.close()
        except OSError:
            pass

    def _self_stop(self):
        """收到应用管理程序的停止指令后退出进程（与 shiwen 一致）。

        先上报离线状态，关闭套接字，再 os._exit(0) 干净退出。
        下次由应用管理程序重新拉起 start.sh，算法以全新进程运行。
        """
        print("[停止] 收到停止指令，算法进程即将退出...", flush=True)
        try:
            self.send_offline_status()
        except Exception:
            pass
        self.running = False
        for sock in (self.local_socket, self.host_socket):
            try:
                sock.close()
            except OSError:
                pass
        os._exit(0)

    def recv_command_loop(self):
        from .protocol import FIELD_HEADER_FORMAT, FIELD_HEADER_SIZE, decode_field_value

        while self.running:
            try:
                recv_data, _ = self.local_socket.recvfrom(RECV_BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(recv_data) < HEADER_SIZE:
                continue

            header_data = struct.unpack(HEADER_FORMAT, recv_data[:HEADER_SIZE])
            command = header_data[2]
            data_length = header_data[4]

            if command == PROG_ALGO_DATA:
                payload = recv_data[HEADER_SIZE : HEADER_SIZE + data_length]
                offset = 0
                while offset + FIELD_HEADER_SIZE <= len(payload):
                    field_idx, field_len = struct.unpack(
                        FIELD_HEADER_FORMAT, payload[offset : offset + FIELD_HEADER_SIZE]
                    )
                    offset += FIELD_HEADER_SIZE
                    if offset + field_len > len(payload):
                        break
                    field_bytes = payload[offset : offset + field_len]
                    try:
                        four_byte_format = "<f" if field_idx in (0, 1, 2) and field_len == 4 else "<I"
                        val = decode_field_value(field_len, field_bytes, four_byte_format=four_byte_format)
                        self.shared_cfg["input_parameters"][field_idx] = val
                    except Exception:
                        pass
                    offset += field_len
                continue
            if command == PROG_ALGO_STOP:
                self._self_stop()


class PhmBackgroundReporter(AlgoUdpAppBase):
    """后台 PHM 心跳 + 周期性故障/结果上报（不阻塞主业务）。"""

    def __init__(
        self,
        algo_id,
        algo_port,
        app_config_path,
        output_provider,
        host_socket_addr=HOST_SOCKET_ADDR_DEFAULT,
        *,
        output_interval_ms=1000,
        log_status=True,
        log_output=False,
        log_enabled=True,
        output_field_names=None,
        shared_cfg=None,
    ):
        super().__init__(
            algo_id,
            algo_port,
            app_config_path,
            host_socket_addr,
            log_status=log_status,
            log_enabled=log_enabled,
            shared_cfg=shared_cfg,
        )
        self.output_provider = output_provider
        self.output_interval_ms = output_interval_ms
        self.log_output = log_output
        self.output_field_names = output_field_names or []
        self._threads = []


    def send_algo_result_loop(self):
        while self.running:
            time.sleep(self.output_interval_ms / 1000.0)

            try:
                result_len, result_payload = self.output_provider()
            except Exception as e:
                print(f"[PHM] 构建结果包失败: {e}")
                continue

            if result_len <= 0:
                continue

            self.result_sequence += 1
            try:
                packet = build_result_packet(
                    self.stat_list, self.result_sequence, result_len, result_payload
                )
                self._send_to_hosts(packet)
                if self.log_output:
                    print(format_result_packet(packet, self.output_field_names), flush=True)
            except socket.error as e:
                print(f"[PHM] 发送故障/结果包失败: {e}")

    def start(self):
        self._threads = [
            threading.Thread(target=self.report_status_loop, daemon=True, name="phm-heartbeat"),
            threading.Thread(target=self.send_algo_result_loop, daemon=True, name="phm-result"),
            threading.Thread(target=self.recv_command_loop, daemon=True, name="phm-recv"),
        ]
        for thread in self._threads:
            thread.start()
        print(
            f"[PHM] 已启动心跳与故障上报 (算法ID={self.stat_list[0]}, "
            f"监听端口={self.stat_list[3]})"
        )

    def stop(self):
        try:
            self.send_offline_status()
        except Exception:
            pass
        self.close()
