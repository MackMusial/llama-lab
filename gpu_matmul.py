#prediction: CPU 137ms, GPU 4.6ms, speedup x
import time
import torch

start = time.perf_counter()
torch.cuda.synchronize()
end = time.perf_counter()
elapsed_ms = (end - start) * 1000  #seconds -> milliseconds
print(f"took {elapsed_ms:.1f} ms")  #:.1f = 1 decimal placeXZ
