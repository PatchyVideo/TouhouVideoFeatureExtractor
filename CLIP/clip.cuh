#pragma once

#include "../common.h"
#include "../wrappers/NvInferContext.h"

void run_clip(
	NvInferContext *clip_ctx,
	f32 *out_features,
	f32 const* const frames,
	usize batch_size,
	u32 width,
	u32 height,
	CUstream stream
);
