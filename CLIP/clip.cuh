#pragma once

#include "../common.h"
#include "../wrappers/CUDADeviceMemory.h"
#include <NvInfer.h>

void run_clip(
	nvinfer1::IExecutionContext* clip_trt_ctx,
	CUDADeviceMemoryUnique<f32>& out_features,
	u8 const* const frames,
	u32 num_frames,
	i32* key_frame_indices, // on CPU
	u32 num_key_frames,
	u32 width, // =288
	u32 height, // =288
	CUDADeviceMemoryUnique<f32>& scratch_f32_1,
	CUDADeviceMemoryUnique<u8>& scratch_u8_1,
	CUDADeviceMemoryUnique<i32>& scratch_i32_1,
	CUstream stream
);
