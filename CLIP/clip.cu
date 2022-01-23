
#include "clip.cuh"

void run_clip(
	NvInferContext* clip_ctx,
	f32* out_features,
	f32 const* const frames,
	usize batch_size,
	u32 width,
	u32 height,
	CUstream stream
) {
}
