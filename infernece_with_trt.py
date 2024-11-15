#inference with .trt files
# source : https://github.com/NVIDIA/TensorRT/blob/release/10.4/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20through%20ONNX.ipynb
# traps : libs a installer. dont numpy qui donne une erreur (2.0.1 marche)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # usefull !
import numpy as np
import time

import torch
from torchvision.transforms import Normalize


print("ça run")
BATCH_SIZE=1024
target_dtype = np.int8

f = open("7_classes_int8.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

input_batch = np.empty([1024, 224, 224, 3], dtype = target_dtype)################# à remplir avc les images


# need to set input and output precisions to FP16 to fully enable it
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

# allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
assert(len(tensor_names) == 2)

context.set_tensor_address(tensor_names[0], int(d_input))
context.set_tensor_address(tensor_names[1], int(d_output))

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()


def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v3(stream.handle)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output




def preprocess_image(img):
    return np.array(torch.from_numpy(img).transpose(0,2).transpose(1,2))

preprocessed_images = np.array([preprocess_image(image) for image in input_batch])

print("Warming up...")

pred = predict(preprocessed_images)

print("Done warming up!")

start_time = time.time()

pred = predict(preprocessed_images)
end_time = time.time()

print("shape of preprocessed_images:", preprocessed_images.shape)
print(f"Inference time {end_time - start_time} seconds")