#pragma once

#include "../common.h"
#include "../wrappers/CUDADeviceMemory.h"
#include <NvInfer.h>

void run_transnet(
	nvinfer1::IExecutionContext* transnet_trt_ctx,
	f32* out_scores, // (950, ) of f32 after softmax
	CUDADeviceMemoryUnique<f32>& scratch_f32_1,
	CUDADeviceMemoryUnique<f32>& scratch_f32_2,
	u8 const* const in_frames, // (900, 3, H, W) of u8 original resolution
	u32 width, // =224
	u32 height, // =224
	CUstream stream
);

