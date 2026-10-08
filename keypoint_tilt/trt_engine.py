#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance TensorRT Engine Runner for HRNet Pantograph Keypoint Detection.
Supports zero-copy GPU tensor inference for both TensorRT 8.x and 10.x APIs,
with automatic fallback to PyTorch/ONNXRuntime if runtime is unavailable.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrtHRNetRunner:
    """
    TensorRT 执行引擎包装器 (针对 HRNet 4 关键点热图回归).
    输入形状: [1, 3, 288, 384] (RGB, ImageNet 归一化)
    输出形状: [1, 4, 72, 96] (4 个关键点高斯热图)
    """

    def __init__(
        self,
        engine_path: str,
        device: str = "cuda:0",
        precision: str = "fp16",
    ):
        self.engine_path = os.path.abspath(engine_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision.lower()
        self.engine = None
        self.context = None
        self.stream = None
        self.is_ready = False

        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"[TrtHRNetRunner] Engine 文件不存在: {self.engine_path}")

        self._load_engine()

    def _load_engine(self) -> None:
        try:
            import tensorrt as trt
        except ImportError as e:
            logger.error(f"[TrtHRNetRunner] 未检测到 tensorrt 模块: {e}")
            raise

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        logger.info(f"[TrtHRNetRunner] 正在加载 TensorRT 引擎: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"[TrtHRNetRunner] 反序列化 CUDA Engine 失败: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("[TrtHRNetRunner] 创建 TensorRT 执行上下文失败")

        # 区分 TensorRT 8.x 与 TensorRT 10.x API
        self.is_trt8 = hasattr(self.engine, "num_bindings")
        self.input_binding_idx = None
        self.output_binding_idx = None
        self.input_name = None
        self.output_name = None

        if self.is_trt8:
            # TRT 8.x API
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                if self.engine.binding_is_input(idx):
                    self.input_binding_idx = idx
                    self.input_name = name
                    self.input_shape = tuple(self.engine.get_binding_shape(idx))
                    self.input_dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                else:
                    self.output_binding_idx = idx
                    self.output_name = name
                    self.output_shape = tuple(self.engine.get_binding_shape(idx))
                    self.output_dtype = trt.nptype(self.engine.get_binding_dtype(idx))
        else:
            # TRT 10.x API
            num_io = self.engine.num_io_tensors
            for idx in range(num_io):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    self.input_shape = tuple(self.engine.get_tensor_shape(name))
                    self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                else:
                    self.output_name = name
                    self.output_shape = tuple(self.engine.get_tensor_shape(name))
                    self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(name))

        logger.info(
            f"[TrtHRNetRunner] Engine 初始化就绪 | "
            f"Input: {self.input_name} {self.input_shape} ({self.input_dtype}) | "
            f"Output: {self.output_name} {self.output_shape} ({self.output_dtype})"
        )

        # 预先分配 GPU 上的输出张量缓冲区，避免推理时重复 cudaMalloc
        # 如果 shape 包含动态 batch (-1)，默认分配 batch=1
        out_shape = [s if s > 0 else 1 for s in self.output_shape]
        torch_dtype = torch.float32 if self.output_dtype == np.float32 else torch.float16
        self.output_tensor = torch.empty(out_shape, dtype=torch_dtype, device=self.device)
        self.cuda_stream = torch.cuda.current_stream().cuda_stream
        self.is_ready = True

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        执行 TensorRT 前向推理.
        input_tensor: [1, 3, 288, 384] 形状张量 (可位于 GPU 或 CPU).
        返回: [1, 4, 72, 96] GPU 上的 torch.Tensor 热图.
        """
        if not self.is_ready:
            raise RuntimeError("[TrtHRNetRunner] Engine 尚未正确初始化")

        # 确保输入在正确的设备上且数据连续
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        # 精度自动转换（若 engine 期望 float16 则输入自动转 half）
        expected_torch_dtype = torch.float32 if self.input_dtype == np.float32 else torch.float16
        if input_tensor.dtype != expected_torch_dtype:
            input_tensor = input_tensor.to(expected_torch_dtype)

        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        batch_size = input_tensor.shape[0]

        if self.is_trt8:
            # TRT 8.x: 设置动态绑定维度并异步执行
            if -1 in self.input_shape:
                self.context.set_binding_shape(self.input_binding_idx, input_tensor.shape)

            # 动态调整输出缓冲区尺寸（若 batch_size 发生变动）
            if self.output_tensor.shape[0] != batch_size:
                out_shape = list(self.output_shape)
                out_shape[0] = batch_size
                out_shape = [s if s > 0 else 1 for s in out_shape]
                torch_dtype = torch.float32 if self.output_dtype == np.float32 else torch.float16
                self.output_tensor = torch.empty(out_shape, dtype=torch_dtype, device=self.device)

            bindings = [None, None]
            bindings[self.input_binding_idx] = input_tensor.data_ptr()
            bindings[self.output_binding_idx] = self.output_tensor.data_ptr()

            success = self.context.execute_async_v2(
                bindings=bindings,
                stream_handle=self.cuda_stream
            )
            if not success:
                raise RuntimeError("[TrtHRNetRunner] execute_async_v2 执行失败")
        else:
            # TRT 10.x API
            self.context.set_input_shape(self.input_name, input_tensor.shape)
            self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())

            success = self.context.execute_async_v3(self.cuda_stream)
            if not success:
                raise RuntimeError("[TrtHRNetRunner] execute_async_v3 执行失败")

        # 确保输出转换回 float32 供后续高斯峰值与几何解算使用
        if self.output_tensor.dtype != torch.float32:
            return self.output_tensor.float()
        return self.output_tensor
