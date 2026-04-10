# src/models/wrappers/trt_dinov2_wrapper.py
#
# TensorRT runtime wrapper для DINOv2 ViT-L/14.
# Завантажує скомпільований .engine файл та виконує інференс без PyTorch overhead.

from pathlib import Path

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# TensorRT доступний не на всіх системах
try:
    import pycuda.autoinit  # noqa: F401 — ініціалізує CUDA context
    import pycuda.driver as cuda
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False


def is_trt_available() -> bool:
    """Перевіряє чи TensorRT runtime доступний."""
    return _TRT_AVAILABLE


class TensorRTDINOv2Wrapper:
    """Runtime wrapper для TensorRT DINOv2 engine.

    Забезпечує інтерфейс forward(image_tensor) -> np.ndarray (1024-dim)
    сумісний із PyTorch DINOv2 wrapper.

    Використання:
        wrapper = TensorRTDINOv2Wrapper("models/engines/dinov2_vitl14_fp16.engine")
        descriptor = wrapper.forward(image_np)  # (1024,) float32
    """

    def __init__(self, engine_path: str, input_size: int = 336):
        if not _TRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install tensorrt and pycuda packages.")

        engine_file = Path(engine_path)
        if not engine_file.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}. "
                f"Run: python scripts/compile_dinov2_trt.py --output {engine_file.parent}"
            )

        self.input_size = input_size
        self._load_engine(str(engine_file))
        logger.success(f"TensorRT DINOv2 engine loaded: {engine_path}")

    def _load_engine(self, engine_path: str):
        """Завантажує TensorRT engine та виділяє GPU пам'ять."""
        trt_logger = trt.Logger(trt.Logger.SEVERE)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Виділення пам'яті для input та output
        self.input_shape = (1, 3, self.input_size, self.input_size)
        self.output_shape = (1, 1024)  # DINOv2 ViT-L/14 output dim

        input_nbytes = int(np.prod(self.input_shape) * np.float32(0).nbytes)
        output_nbytes = int(np.prod(self.output_shape) * np.float32(0).nbytes)

        # GPU буфери
        self.d_input = cuda.mem_alloc(input_nbytes)
        self.d_output = cuda.mem_alloc(output_nbytes)

        # CPU буфери (page-locked для швидкого копіювання)
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

        self.stream = cuda.Stream()
        logger.debug(f"TRT buffers allocated: input={input_nbytes}B, output={output_nbytes}B")

    def forward(self, image_chw: np.ndarray) -> np.ndarray:
        """Виконує інференс TensorRT engine.

        Args:
            image_chw: нормалізоване зображення (3, H, W) float32
                       (ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        Returns:
            global_descriptor: (1024,) float32
        """
        # Копіюємо дані у page-locked буфер
        np.copyto(self.h_input, image_chw.reshape(self.input_shape).astype(np.float32))

        # Host → Device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Інференс
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )

        # Device → Host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.reshape(-1).copy()  # (1024,)

    @property
    def output_dim(self) -> int:
        """Повертає розмірність вихідного дескриптора."""
        return self.output_shape[-1]

    def __del__(self):
        """Звільнює GPU ресурси."""
        try:
            if hasattr(self, "d_input"):
                self.d_input.free()
            if hasattr(self, "d_output"):
                self.d_output.free()
        except Exception:
            pass  # Ігноруємо помилки при garbage collection
