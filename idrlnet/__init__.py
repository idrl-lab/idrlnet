import torch
from .header import logger

GPU_AVAILABLE = False
GPU_ENABLED = False
# todo more careful check
if torch.cuda.is_available():
    try:
        _ = torch.Tensor([0.0, 0.0]).cuda()
        logger.info("GPU available")
        GPU_AVAILABLE = True
    except:
        logger.info("GPU not available")
        GPU_AVAILABLE = False
else:
    logger.info("GPU not available")
    GPU_AVAILABLE = False


def use_gpu(device=0):
    """Use GPU with device `device`.

    Args:
        device (torch.device or int): selected device.
    """
    if GPU_AVAILABLE:
        try:
            torch.cuda.set_device(device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            logger.info(f"Using GPU device {device}")
            global GPU_ENABLED
            GPU_ENABLED = True
        except:
            logger.warning("Invalid device ordinal")


def use_cpu():
    """
    Use CPU.
    """
    if GPU_ENABLED:
        torch.set_default_tensor_type("torch.FloatTensor")
    logger.info(f"Using CPU")
