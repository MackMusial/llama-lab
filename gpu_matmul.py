#  Prediction (matrix x matrix): CPU 137ms, GPU 4.6ms, speedup ~30x
#  Prediction (matrix x vector): CPU 1.22ms, GPU 0.13ms, speedup ~9x

#  My suggestion to anyone who is reading this...

#  If you have not done something like this, I would highly recommend it.
#  Its a great brain workout and you WILL learn alot


import time
import torch

A = torch.randn(4096,4096)
B = torch.randn(4096,4096)

start = time.perf_counter()  #start the timer
C = A @ B  #this is what we are timing... This is computed on the CPU
end = time.perf_counter()  #end the timer
cpu_elapsed_ms = (end - start)*1000  #convert seconds to ms
print(f"CPU matrix computed in {cpu_elapsed_ms:.1f} ms")  #print

A_gpu = A.to("cuda")  #copy of a that was originally on the cpu has now been copied too
B_gpu = B.to("cuda")  #a copy of b on the gpu. the original copy stays one the cpu
warmup_gpu = A_gpu @ B_gpu  #warmup for the GPU. He needs to stretch his legs first
torch.cuda.synchronize()  #wait for gpu to finish
start = time.perf_counter()  #start timer
D_gpu = A_gpu @ B_gpu  #perform the matrix multiplications
torch.cuda.synchronize()  #wait for the GPU to finish calculations. Without this python would go to the next line immediatly and not wait for the GPU to finish calc's
end = time.perf_counter()  #end timer
gpu_elapsed_ms = (end - start)*1000  #convert seconds to miliseconds
print(f"GPU matrix computed in {gpu_elapsed_ms:.1f} ms")  #print time taken
print(f"Matrix speedup: {cpu_elapsed_ms / gpu_elapsed_ms:.1f}x")

#-------------------------------------------------------------------------------
# matrix x vector: each number in A is used once, so this is memory-bound (like generation)
runs = 100

V = torch.randn(4096)
start = time.perf_counter()
for _ in range(runs):
    C = A @ V
end = time.perf_counter()
cpu_elapsed_ms = (end - start)*1000 / runs  #seconds -> ms, then average per run
print(f"CPU vector computed in {cpu_elapsed_ms:.3f} ms")

V_gpu = V.to("cuda")  #copy of V on the gpu
A_gpu @ V_gpu  #warmup
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(runs):
    C_gpu = A_gpu @ V_gpu
torch.cuda.synchronize()
end = time.perf_counter()
gpu_elapsed_ms = (end - start)*1000 / runs
print(f"GPU vector computed in {gpu_elapsed_ms:.3f} ms")
print(f"Vector speedup: {cpu_elapsed_ms / gpu_elapsed_ms:.1f}x")
