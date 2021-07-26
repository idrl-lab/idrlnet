import torch

# todo more careful check
GPU_ENABLED = True
if torch.cuda.is_available():
    try:
        _ = torch.Tensor([0.0, 0.0]).cuda()
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("gpu available")
        GPU_ENABLED = True
    except:
        print("gpu not available")
        GPU_ENABLED = False
else:
    print("gpu not available")
    GPU_ENABLED = False
