#pragma once

#include "../common.h"

void RgbBytesToRgbFP32(
	u8 const* const input_u8_frames,
	f32* out_fp32_frames,
	usize input_frame_count,
	i32 width,
	i32 height,
	CUstream stream = nullptr
);
