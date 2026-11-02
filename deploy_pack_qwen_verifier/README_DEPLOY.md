# Qwen 大模型受电弓大火花与结构异常二级复核交付包

本交付包用于将经过充分实测调优的 **Qwen3-VL-4B 二级复核防线** 部署到板卡（NVIDIA Jetson AGX Orin）上，对第一道工控防线产生的报警图片进行高精度复核过滤。

---

## 一、文件清单与说明

| 文件名 | 类型 | 说明 |
| :--- | :--- | :--- |
| [`qwen_alarm_verifier.py`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/qwen_alarm_verifier.py) | **新增核心** | Qwen 二级复核核心引擎。内置最佳 Prompt 负样本约束与去偏机制；支持 Python 导入调用、单图测试、批量测试及后台目录监听守护。 |
| [`ai_detector.py`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/ai_detector.py) | **覆盖替换** | 增强版检测服务。内嵌异步复核队列与 Worker 线程，主视频流保持 25fps 零卡顿；大火花/结构异常经 Qwen 复核裁决后放行或拦截 UDP。 |
| [`monitor_main.py`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/monitor_main.py) | **覆盖替换** | 主入口。多路流共享单例 Qwen 复核引擎，避免重复占满 Orin 显存。 |
| [`config_prod.yaml`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/config_prod.yaml) | **覆盖/参考** | 新增 `qwen_verification` 配置段，可一键开关复核功能。 |
| [`start_qwen_verifier_daemon.sh`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/start_qwen_verifier_daemon.sh) | **新增脚本** | 独立目录监听守护启动脚本，用于“外挂零侵入模式”。 |

---

## 二、部署方式选择（按需二选一）

### 方案 1：独立守护监听模式（强烈推荐 ⭐，零侵入第一道防线）
* **特点**：
  * 第一道工控防线代码完全不用动，保证实时拉流 100% 稳定；
  * 新建一个后台进程，实时监听报警落盘目录（如 `/hirain/hi_opt/nvme1/Algo-result`）；
  * 只要第一道防线落盘了大火花（`-3.jpg`）或结构异常（`-4.jpg`），Qwen 立即自动二次核验，并在图片旁生成 `_qwen_verdict.json` 与历史汇总 CSV。
* **操作步骤**：
  1. 将 [`qwen_alarm_verifier.py`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/qwen_alarm_verifier.py) 和 [`start_qwen_verifier_daemon.sh`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/start_qwen_verifier_daemon.sh) 拷贝到板子工程根目录。
  2. 启动后台守护服务：
     ```bash
     nohup ./start_qwen_verifier_daemon.sh /hirain/hi_opt/nvme1/Algo-result > qwen_verifier.log 2>&1 &
     ```

---

### 方案 2：主服务深度内嵌联动模式（自动拦截虚警 UDP 报文）
* **特点**：
  * 第一道防线检测出大火花/结构异常并落盘图片后，自动推入内部异步队列；
  * Qwen 复核确认真故障才发送车载 UDP 8080 报文与触发录像；
  * Qwen 判定为虚警则直接拦截抑制，彻底避免司机室误报！
  * 脏污、异物、倾斜、飘弓不受影响，照常实时上报。
* **操作步骤**：
  1. 将 [`qwen_alarm_verifier.py`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/qwen_alarm_verifier.py) 复制到板子工程根目录。
  2. 覆盖替换板子上的 [`ai_detector.py`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/ai_detector.py) 和 [`monitor_main.py`](file:///mnt/T4/jyb/gongwang_clean/deploy_pack_qwen_verifier/monitor_main.py)。
  3. 在 `config_prod.yaml` 末尾添加配置：
     ```yaml
     qwen_verification:
       enabled: true                    # 开启复核拦截
       model_path: "./models/Qwen3-VL-4B-Instruct"
       device: "cuda:0"
       gpu_mem_fraction: 0.80
       suppress_false_alarm_udp: true    # 虚警拦截 UDP 发送
       target_faults:
         - "big_spark"
         - "structure_abnormal"
     ```
  4. 按照常规方式运行 `./start.sh` 或 `python3 main.py config_prod.yaml` 即可。

---

## 三、命令行常用验证指令

在板子上随时可手动单图或批量测试：

1. **单图测试**：
   ```bash
   python3 qwen_alarm_verifier.py --image /path/to/spark_image.jpg --fault-type big_spark
   ```
2. **离线批量核验目录内所有历史抓拍**：
   ```bash
   python3 qwen_alarm_verifier.py --scan-dir /hirain/hi_opt/nvme1/Algo-result --output-csv ./history_verified.csv
   ```
