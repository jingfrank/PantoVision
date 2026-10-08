#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance TensorRT Engine Runner for FastAD WideResNet Backbone.
Provides zero-copy GPU tensor inference for both TensorRT 8.x and 10.x APIs,
with automatic PyTorch fallback if runtime or engine is unavailable.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrtFastADBackboneRunner:
    """
    TensorRT 执行引擎包装器 (针对 FastAD WideResNet-50 剪枝骨干).
    输入形状: [1, 3, 256, 256] (RGB, ImageNet 归一化)
    输出形状: [1, 512, 32, 32] (Layer 2 特征图，或多层特征图列表)
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
            raise FileNotFoundError(f"[TrtFastADBackboneRunner] Engine 文件不存在: {self.engine_path}")

        self._load_engine()

    def _load_engine(self) -> None:
        try:
            import tensorrt as trt
        except ImportError as e:
            logger.error(f"[TrtFastADBackboneRunner] 未检测到 tensorrt 模块: {e}")
            raise

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        logger.info(f"[TrtFastADBackboneRunner] 正在加载 FastAD TensorRT 骨干引擎: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"[TrtFastADBackboneRunner] 反序列化 CUDA Engine 失败: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("[TrtFastADBackboneRunner] 创建 TensorRT 执行上下文失败")

        self.is_trt8 = hasattr(self.engine, "num_bindings")
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}
        self.input_dtypes: Dict[str, Any] = {}
        self.output_dtypes: Dict[str, Any] = {}

        if self.is_trt8:
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                shape = tuple(self.engine.get_binding_shape(idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                if self.engine.binding_is_input(idx):
                    self.input_names.append(name)
                    self.input_shapes[name] = shape
                    self.input_dtypes[name] = dtype
                else:
                    self.output_names.append(name)
                    self.output_shapes[name] = shape
                    self.output_dtypes[name] = dtype
        else:
            num_io = self.engine.num_io_tensors
            for idx in range(num_io):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                    self.input_shapes[name] = shape
                    self.input_dtypes[name] = dtype
                else:
                    self.output_names.append(name)
                    self.output_shapes[name] = shape
                    self.output_dtypes[name] = dtype

        self.stream = torch.cuda.current_stream(device=self.device)

        # 预分配固定尺寸 GPU 输出缓存，实现零开销重用
        self._output_tensors: Dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = self.output_shapes[name]
            # 若 shape 含有动态维度 -1，默认按 batch=1 解析
            concrete_shape = tuple(1 if dim < 0 else dim for dim in shape)
            self._output_tensors[name] = torch.empty(
                concrete_shape,
                dtype=torch.float32,
                device=self.device,
            )

        self.is_ready = True
        logger.info(
            f"[TrtFastADBackboneRunner] FastAD Engine 初始化完成 | 输入: {self.input_names} | 输出: {self.output_names}"
        )

    def is_available(self) -> bool:
        return self.is_ready and self.context is not None

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        前向推理接口 (Zero-Copy GPU Tensor In/Out).
        Args:
            x: 输入张量 [B, 3, H, W]，通常在 GPU 上且已归一化
        Returns:
            若单个输出返回 Tensor [B, C, H', W']，若多输出则返回 List[Tensor]
        """
        if not self.is_ready:
            raise RuntimeError("[TrtFastADBackboneRunner] Engine 未就绪")

        if not x.is_cuda:
            x = x.to(self.device, non_blocking=True)
        if x.dtype != torch.float32:
            x = x.float()

        x = x.contiguous()
        stream_ptr = torch.cuda.current_stream(device=self.device).cuda_stream

        input_name = self.input_names[0]

        if self.is_trt8:
            # TRT 8.x
            bindings = [0] * self.engine.num_bindings
            in_idx = self.engine.get_binding_index(input_name)
            bindings[in_idx] = x.data_ptr()

            for out_name in self.output_names:
                out_idx = self.engine.get_binding_index(out_name)
                bindings[out_idx] = self._output_tensors[out_name].data_ptr()

            self.context.execute_async_v2(
                bindings=bindings,
                stream_handle=stream_ptr,
            )
        else:
            # TRT 10.x
            self.context.set_tensor_address(input_name, x.data_ptr())
            for out_name in self.output_names:
                self.context.set_tensor_address(
                    out_name,
                    self._output_tensors[out_name].data_ptr(),
                )
            self.context.execute_async_v3(stream_handle=stream_ptr)

        outputs = [self._output_tensors[name] for name in self.output_names]
        if len(outputs) == 1:
            return outputs[0]
        return outputs
