Scripts to infere with any .onnx model FAST !

convert .onnx to .trt with the trtexec tool (to install if needed!) with the command :
trtexec --onnx=your_model.onnx --saveEngine=7_classes_fp16.trt --fp16

Then run (and/or adapt) the inference script : inference_with_tensort.py

Note : builder and dockerfile are only here to help install tensorRT

