#pragma once

#include "../common.h"

void nv12_to_rgb_resize(
	f32* out_resized_rgb_frames,
	f32 const* const batched_yuv_frames,
	usize batch_size,
	usize nv12_frame_size_in_bytes,
	i32 input_width,
	i32 input_height,
	i32 output_width,
	i32 output_height,
	CUstream stream
);
