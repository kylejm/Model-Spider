# hardware_token.py
import torch
import numpy as np

class HardwareTokenizer:
    """
    Produce a numeric hardware vector that includes:
      - One-hot GPU type (covering Apple M1–M4, various Nvidia cards, IoT devices).
      - Other numeric fields for VRAM, RAM, power usage, latency, etc.
      - Output dimension should be 1024
    """

    GPU_TYPE_LIST = [

    # Apple Silicon
    "Apple-M1", "Apple-M1-Pro", "Apple-M1-Max", "Apple-M1-Ultra",
    "Apple-M2", "Apple-M2-Pro", "Apple-M2-Max", "Apple-M2-Ultra",
    "Apple-M3", "Apple-M3-Pro", "Apple-M3-Max", "Apple-M3-Ultra",  # hypothetical
    "Apple-M4", "Apple-M4-Pro", "Apple-M4-Max", "Apple-M4-Ultra",  # hypothetical

    # Nvidia Consumer (GTX/RTX series)
    "NVIDIA-GTX-970", "NVIDIA-GTX-980", "NVIDIA-GTX-1070", "NVIDIA-GTX-1070Ti", "NVIDIA-GTX-1080","NVIDIA-GTX-1080Ti",
    "NVIDIA-RTX-2060", "NVIDIA-RTX-2070", "NVIDIA-RTX-2080", "NVIDIA-RTX-2080Ti",
    "NVIDIA-RTX-3060", "NVIDIA-RTX-3070", "NVIDIA-RTX-3080", "NVIDIA-RTX-3090",
    "NVIDIA-RTX-4060", "NVIDIA-RTX-4070", "NVIDIA-RTX-4080", "NVIDIA-RTX-4090",

    # Nvidia Data Center
    "NVIDIA-T4", "NVIDIA-A10", "NVIDIA-A100",
    "NVIDIA-V100", "NVIDIA-P100", "NVIDIA-P4", "NVIDIA-P40",

    # IoT / Edge devices
    "Raspberry-Pi-3", "Raspberry-Pi-4", 
    "NVIDIA-Jetson-Nano", "NVIDIA-Jetson-Xavier-NX",
    
    ## Nvidia Mobile GPUs
    "NVIDIA-RTX-3050", "NVIDIA-RTX-3050Ti",
    
    ## Nvidia Quadro Series 
    "NVIDIA-Quadro-P2000", "NVIDIA-Quadro-P4000", "NVIDIA-Quadro-P5000", "NVIDIA-Quadro-P6000",
    "NVIDIA-Quadro-RTX-4000", "NVIDIA-Quadro-RTX-5000", "NVIDIA-Quadro-RTX-6000", "NVIDIA-Quadro-RTX-8000",

    ## AMD Radeon Consumer GPUs
    "AMD-Radeon-RX-460", "AMD-Radeon-RX-470", "AMD-Radeon-RX-480", "AMD-Radeon-RX-560", "AMD-Radeon-RX-570",
    "AMD-Radeon-RX-580", "AMD-Radeon-RX-5500", "AMD-Radeon-RX-5600", "AMD-Radeon-RX-5700",
    "AMD-Radeon-RX-6600", "AMD-Radeon-RX-6650", "AMD-Radeon-RX-6700", "AMD-Radeon-RX-6750", 
    "AMD-Radeon-RX-6800", "AMD-Radeon-RX-6800XT", "AMD-Radeon-RX-6900XT",

    ## AMD Radeon Pro / Data Center GPUs
    "AMD-Radeon-Pro-W5500", "AMD-Radeon-Pro-W5700", "AMD-Radeon-Pro-W6600",
    "AMD-MI50", "AMD-MI60", 

    ## Intel Integrated GPUs
    "Intel-UHD-Graphics-620", "Intel-UHD-Graphics-630", "Intel-Iris-Plus", "Intel-Iris-Xe", 
    "Intel-Iris-Xe-LP", "Intel-Iris-Xe-HP",

    ## Other Embedded / Mobile GPUs
    "Imagination-PowerVR-GPU", "Vivante-GPU",

    ## Cloud Provider Instance GPUs
    "AWS-G4DN", "AWS-P3", "AWS-P4", "GCP-T4", "GCP-A100",

    ## Hypothetical / Future Models
    "NVIDIA-RTX-5090", "AMD-Radeon-RX-7900XT",

    # Catch-all or unknown
    "Unknown-GPU"
]
    GPU_TYPE_MAP = {name: idx for idx, name in enumerate(GPU_TYPE_LIST)}

    def __init__(self):
        pass

    def _get_gpu_one_hot(self, gpu_type: str):
        """
        Return a one-hot vector for the GPU type.
        If the gpu_type doesn't exist in GPU_TYPE_MAP, fallback to 'Unknown-GPU'.
        """
        index = self.GPU_TYPE_MAP.get(gpu_type, self.GPU_TYPE_MAP["Unknown-GPU"])
        one_hot = [0.0] * len(self.GPU_TYPE_LIST)
        one_hot[index] = 1.0
        return one_hot

    def get_hardware_vector_manual(
        self,
        gpu_type: str = "Apple-M1",
        vram_gb: float = 8.0,
        ram_gb: float = 16.0,
        power_watts: float = 30.0,
        latency_ms: float = 100.0
    ) -> torch.Tensor:
        """
        Return a numeric vector describing hardware constraints:
            - one-hot for GPU type
            - scaled VRAM, system RAM, power usage, latency
        """
        gpu_one_hot = self._get_gpu_one_hot(gpu_type)

        # Scaling
        vram_scaled    = vram_gb / 100.0    
        ram_scaled     = ram_gb / 512.0     
        power_scaled   = power_watts / 500.0   
        latency_scaled = latency_ms / 1000.0   

        feats = gpu_one_hot + [vram_scaled, ram_scaled, power_scaled, latency_scaled]

        return torch.tensor(feats, dtype=torch.float32)