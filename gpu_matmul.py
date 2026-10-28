#prediction: CPU 137ms, GPU 4.6ms, speedup 29.7826 x

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
print(f"CPU computed in {cpu_elapsed_ms:.1f} ms")  #print

A_gpu = A.to("cuda")  #copy of a that was originally on the cpu has now been copied too
B_gpu = B.to("cuda")  #a copy of b on the gpu. the original copy stays one the cpu
warmup_gpu = A_gpu @ B_gpu  #warmup for the GPU. He needs to stretch his legs first
torch.cuda.synchronize()  #wait for gpu to finish
start = time.perf_counter()  #start timer
D_gpu = A_gpu @ B_gpu  #perform the matrix multiplications
torch.cuda.synchronize()  #wait for the GPU to finish calculations. Without this python would go to the next line immediatly and not wait for the GPU to finish calc's 
end = time.perf_counter()  #end timer
gpu_elapsed_ms = (end - start)*1000  #convert seconds to miliseconds
print(f"GPU computed in {gpu_elapsed_ms:.1f} ms")  #print time taken










# x = torch.randn(3, 3)    # 3x3 grid of random numbers, on the CPU
# y = x @ x                # matrix multiply (@ is the matmul operator)
# x_gpu = x.to("cuda")     # a COPY of x on the GPU; x itself stays on the CPU
# print(x_gpu.device)      # prints: cuda:0

# torch.cuda.synchronize()   # "wait here until the GPU has finished everything I queued"

