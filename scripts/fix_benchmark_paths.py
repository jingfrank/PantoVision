#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规范化修复 tests_and_benchmarks 中脚本的便携式路径
使得脚本在任何笔记本或机器上均可直接运行，自动寻找本地工程目录下的 models 和 fastad_core
"""

import os
import re

BENCH_DIR = "/mnt/T4/jyb/qwen_vlm_defense_workspace/tests_and_benchmarks"

PATH_HEADER = '''
# --- 自动解析便携式工程根路径 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_FASTAD_CORE_DIR = os.path.join(_PROJECT_ROOT, "fastad_core")
if _FASTAD_CORE_DIR not in sys.path:
    sys.path.insert(0, _FASTAD_CORE_DIR)
# -----------------------------
'''

for fname in os.listdir(BENCH_DIR):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(BENCH_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    orig_content = content

    # 替换 fastad 路径引用
    content = re.sub(
        r'sys\.path\.insert\(0,\s*["\']/mnt/T4/jyb/gongwang_fastad["\']\)',
        '# 便携式 fastad 路径注入已由上方接管',
        content
    )

    # 替换模型硬编码路径
    content = content.replace(
        '"/mnt/T4/jyb/gongwang_fastad/models/fastad_layer2_256_reg1e4_nq3_fmq1.pt"',
        'os.path.join(_PROJECT_ROOT, "models", "fastad_layer2_256_reg1e4_nq3_fmq1.pt")'
    )
    content = content.replace(
        '"/mnt/T4/jyb/gongwang/models/Qwen3-VL-4B-Instruct"',
        'os.path.join(_PROJECT_ROOT, "models", "Qwen3-VL-4B-Instruct-AWQ-4bit")'
    )
    content = content.replace(
        '"/mnt/T4/jyb/gongwang_clean/models/Qwen3-VL-4B-Instruct"',
        'os.path.join(_PROJECT_ROOT, "models", "Qwen3-VL-4B-Instruct-AWQ-4bit")'
    )
    content = content.replace(
        'DEFAULT_MODEL_PATH = "/mnt/T4/jyb/gongwang/models/Qwen3-VL-4B-Instruct"',
        'DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "Qwen3-VL-4B-Instruct-AWQ-4bit")'
    )

    # 注入 PATH_HEADER
    if "_PROJECT_ROOT" not in content:
        # 在 import sys 之后插入
        if "import sys" in content:
            content = content.replace("import sys", "import sys" + PATH_HEADER, 1)

    if content != orig_content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 成功规范化便携式路径: {fname}")

print("全部脚本便携化处理完毕！")
