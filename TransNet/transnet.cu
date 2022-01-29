
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "transnet.cuh"
#include "../wrappers/CUDADeviceMemory.h"

union load_store_4f32
{
	f32 f32s[4];
};

__global__ void create_padding_kernel2(
	f32 *inout_frames, // 950 frames of 3x27x48(CHW) frames
	usize frame_stride // 48*27*3=3888
) {
	// 48*27*3/16=243 threads in a block, 10 blocks
	i32 ix(blockIdx.x * blockDim.x + threadIdx.x), iz(blockIdx.z);
	f32* src_frame, * dst_frame;
	if (iz < 5) {
		// head
		src_frame = inout_frames + frame_stride * 25;
		dst_frame = inout_frames + frame_stride * iz * 5;
	}
	else {
		src_frame = inout_frames + frame_stride * (900 + 25 - 1);
		dst_frame = inout_frames + frame_stride * (900 + 25) + frame_stride * iz * 5;
	}
	load_store_4f32* src(reinterpret_cast<load_store_4f32*>(src_frame) + ix);
	load_store_4f32* dst(reinterpret_cast<load_store_4f32*>(dst_frame) + ix);
	usize frame_stride_4(frame_stride / sizeof(load_store_4f32));
#pragma unroll
	for (u32 i(0); i < 5; ++i) {
		*(dst + i * frame_stride_4) = *(src + i * frame_stride_4);
	}

}

__global__ void create_padding_kernel(
	f32* inout_frames, // 950 frames of 3x27x48(CHW) frames
	u32 frame_stride, // 48*27*3=3888
	u32 num_frames
) {
	i32 ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	f32* src_frame, * dst_frame;
	if (iz < 1) {
		// head
		src_frame = inout_frames + frame_stride * 25;
		dst_frame = inout_frames + frame_stride * iz;
	}
	else {
		src_frame = inout_frames + frame_stride * (num_frames + 25 - 1);
		dst_frame = inout_frames + frame_stride * (num_frames + 25) + frame_stride * (iz - 1);
	}
	if (ix < 48 && iy < 27) {
		for (u32 i(0); i < 25; ++i) {
			*(dst_frame + frame_stride * i + 48 * 27 * 0 + 48 * iy + ix) = *(src_frame + 48 * 27 * 0 + 48 * iy + ix);
			*(dst_frame + frame_stride * i + 48 * 27 * 1 + 48 * iy + ix) = *(src_frame + 48 * 27 * 1 + 48 * iy + ix);
			*(dst_frame + frame_stride * i + 48 * 27 * 2 + 48 * iy + ix) = *(src_frame + 48 * 27 * 2 + 48 * iy + ix);
		}
	}
}

__global__ void downsample_nn_and_to_fp32_1C_kernel(
	f32 *dst,
	i32 dst_width,
	i32 dst_height,
	u8 const *const src,
	i32 src_width,
	i32 src_height,
	u32 num_images
) {
	i32 ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z * blockDim.z + threadIdx.z);
	i32 src_stride = src_width * src_height;
	i32 dst_stride = dst_width * dst_height;
	if (ix < dst_width && iy < dst_height) {
		u32 img_bond((iz + 1) * 10);
		img_bond = std::min(img_bond, num_images);
		float2 uv{ __int2float_rz(ix) / __int2float_rz(dst_width - 1), __int2float_rz(iy) / __int2float_rz(dst_height - 1) };
		int2 src_pos{ __float2int_rz(uv.x * __int2float_rz(src_width - 1) + 0.5f),  __float2int_rz(uv.y * __int2float_rz(src_height - 1) + 0.5f) };
		for (u32 i(iz * 10); i < img_bond; ++i) {
			f32* dst_pix(dst + dst_stride * i + dst_width * iy + ix);
			u8 const* const src_pix(src + src_stride * i + src_width * src_pos.y + src_pos.x);
			*dst_pix = static_cast<f32>(*src_pix) / 255.0f;
		}
	}
}

void downsample_nn_and_to_fp32_1C(
	f32* dst,
	i32 dst_wdith,
	i32 dst_height,
	u8 const* const src,
	i32 src_width,
	i32 src_height,
	u32 num_images,
	CUstream stream
) {
	//dst_height + 1 to make num thread multiple of wrap size
	dim3 block(32, 32, 1);
	dim3 grid(2, 1, (num_images - 1) / 10 + 1);
	downsample_nn_and_to_fp32_1C_kernel<<<grid, block, 0, stream>>>(dst, dst_wdith, dst_height, src, src_width, src_height, num_images);
	ck2(cudaGetLastError());
}

__global__ void extract_score_kernel(
	f32* out_scores, // (950, ) of f32 after softmax
	f32 const* const in_softmax_output // (950, 2) of 32
) {
	// 950 threads
	i32 ix(blockIdx.x * blockDim.x + threadIdx.x);
	if (ix >= 950)
		return;
	out_scores[ix] = in_softmax_output[ix * 2 + 1];
}

void run_transnet(
	nvinfer1::IExecutionContext* transnet_trt_ctx,
	f32 *out_scores, // (num_frames+50, ) of f32 after softmax
	CUDADeviceMemoryUnique<f32>& scratch_f32_1,
	CUDADeviceMemoryUnique<f32>& scratch_f32_2,
	u8 const * const in_frames, // (num_frames, 3, H, W) of u8 original resolution
	u32 width, // =288
	u32 height, // =288
	u32 num_frames,
	CUstream stream
) {
	if (scratch_f32_1.size() < 950 * 3 * 27 * 48)
		scratch_f32_1.reallocate(950 * 3 * 27 * 48);
	if (scratch_f32_2.size() < 950 * 2)
		scratch_f32_2.reallocate(950 * 2);
	// resize to scratch_f32+25frames
	downsample_nn_and_to_fp32_1C(scratch_f32_1.at_offset(27 * 48 * 3, 25), 48, 27, in_frames, width, height, num_frames * 3, stream);


	// pad 25 frames to head and tail
	dim3 block_padding(32, 32, 1);
	dim3 grid_padding(2, 1, 1);
	create_padding_kernel<<<grid_padding, block_padding, 0, stream>>>(reinterpret_cast<f32*>(scratch_f32_1.ptr), 48 * 27 * 3, num_frames);
	ck2(cudaGetLastError());

	std::vector<void*> bindings{
		scratch_f32_1.at_offset(0, 0),
		scratch_f32_2.at_offset(0, 0),
	};

	// enqueue TransNet
	if (!transnet_trt_ctx->enqueueV2(bindings.data(), stream, nullptr)) {
		throw std::runtime_error("enqueue failed!!!");
	}

	// extract score
	dim3 block_score(1024, 1, 1);
	dim3 grid_score(1, 1, 1);
	extract_score_kernel<<<grid_score, block_score, 0, stream>>>(out_scores, scratch_f32_2.at_offset(0, 0));
	ck2(cudaGetLastError());
}
