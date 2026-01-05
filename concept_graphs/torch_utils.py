import torch
import numpy as np

_torch_device_name = None
def maybe_set_mps_compatibility_flags(device: str) -> None:
    global _torch_device_name
    _torch_device_name = device

    if device == "mps":
        import torch
        torch.set_default_dtype(torch.float32)

def np_to_torch(np_array: np.ndarray) -> torch.Tensor:
    assert _torch_device_name is not None, "Torch device name is not set"

    if _torch_device_name == "mps":
        np_array = np_array.astype(np.float32)

    ret = torch.from_numpy(np_array)

    if _torch_device_name == "mps":
        assert ret.dtype == torch.float32

    return ret
