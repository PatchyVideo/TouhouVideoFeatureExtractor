
#include "clip.cuh"
#include "../Utils/RgbBytesToRgbFP32.cuh"

template<typename T>
__device__ __forceinline__ void write_pixel_gs(T* grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, T val, std::int32_t width, std::int32_t height)
{
	*(grayscale + width * height * batch + width * y + x) = val;
}

template<typename T>
__device__ __forceinline__ void write_pixel(T* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, T r, T g, T b, std::int32_t width, std::int32_t height)
{
	*(rgb + width * height * 3 * batch + width * height * 0 + width * y + x) = r;
	*(rgb + width * height * 3 * batch + width * height * 1 + width * y + x) = g;
	*(rgb + width * height * 3 * batch + width * height * 2 + width * y + x) = b;
}

template<typename T>
__device__ __forceinline__ T read_pixel(T const* const grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t channel, std::int32_t width, std::int32_t height)
{
	return *(grayscale + width * height * 3 * batch + width * height * channel + width * y + x);
}

union load_16bytes
{
	uint4 u128;
	struct
	{
		std::uint8_t u8s[16];
	};
};

// copy extracted frames to contiguous storage space
__global__ void ExtractContiguousTextRegions_kernel(
	std::uint8_t const* const in_frames,
	std::uint8_t* out_frames,
	std::int32_t const* const in_source_index,
	std::int32_t frame_height,
	std::int32_t reduced_frame_width
)
{
	load_16bytes const* const in_16bytes(reinterpret_cast<load_16bytes const* const>(in_frames));
	load_16bytes* out_16bytes(reinterpret_cast<load_16bytes*>(out_frames));
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	if (ix >= reduced_frame_width || iy >= frame_height / 16)
		return;
	std::int32_t source_frame(in_source_index[iz]);
	if (source_frame == -1)
	{
		load_16bytes zeros{};
#pragma unroll
		for (int i(0); i < 16; ++i)
		{
			write_pixel(out_16bytes, iz, ix, iy * 16 + i, zeros, zeros, zeros, reduced_frame_width, frame_height);
		}
	}
	else
	{
#pragma unroll
		for (int i(0); i < 16; ++i)
		{
			auto r(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 0, reduced_frame_width, frame_height));
			auto g(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 1, reduced_frame_width, frame_height));
			auto b(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 2, reduced_frame_width, frame_height));
			write_pixel(out_16bytes, iz, ix, iy * 16 + i, r, g, b, reduced_frame_width, frame_height);
		}
	}
}

void run_clip(
	nvinfer1::IExecutionContext* clip_trt_ctx,
	CUDADeviceMemoryUnique<f32>& out_features,
	u8 const* const frames,
	u32 num_frames,
	i32 *key_frame_indices, // on CPU, always 950 frames
	u32 num_key_frames,
	u32 width, // =288
	u32 height, // =288
	CUDADeviceMemoryUnique<f32>& scratch_f32_1,
	CUDADeviceMemoryUnique<u8>& scratch_u8_1,
	CUDADeviceMemoryUnique<i32>& scratch_i32_1,
	CUstream stream
) {
	constexpr u32 output_batch_size = 8;
	constexpr u32 out_feature_size = 640;
	auto reduced_region_width(width / 16);
	auto num_contiguous_regions(num_key_frames);

	// pad
	if (num_contiguous_regions % output_batch_size != 0)
	{
		auto pad_size(output_batch_size - (num_contiguous_regions % output_batch_size));
		for (std::size_t i(0); i < pad_size; ++i)
			key_frame_indices[num_contiguous_regions++] = -1;
	}

	// upload indices
	if (scratch_i32_1.empty() || scratch_i32_1.size() < num_contiguous_regions)
		scratch_i32_1.reallocate(num_contiguous_regions);
	scratch_i32_1.upload_partial(key_frame_indices, num_contiguous_regions, stream);

	// allocate enough space for contiguous key frames
	if (scratch_u8_1.empty() || scratch_u8_1.size() < num_contiguous_regions * 3 * height * width)
		scratch_u8_1.reallocate(num_contiguous_regions * 3 * height * width);
	if (scratch_f32_1.empty() || scratch_f32_1.size() < num_contiguous_regions * 3 * height * width)
		scratch_f32_1.reallocate(num_contiguous_regions * 3 * height * width);

	// allocate space for features
	if (out_features.empty() || out_features.size() < num_contiguous_regions * out_feature_size)
		out_features.reallocate(num_contiguous_regions * out_feature_size);

	// extract contiguous key frames
	dim3 block(32, 32, 1);
	dim3 grid((reduced_region_width - 1) / 32 + 1, (height / 16 - 1) / 32 + 1, num_contiguous_regions);
	ExtractContiguousTextRegions_kernel<<<grid, block, 0, stream>>>(frames, scratch_u8_1.at_offset(0, 0), scratch_i32_1.at_offset(0, 0), height, reduced_region_width);
	ck2(cudaGetLastError());

	// convert to fp32
	RgbBytesToRgbFP32(scratch_u8_1.at_offset(0, 0), scratch_f32_1.at_offset(0, 0), num_contiguous_regions, width, height, stream);

	// run clip
	for (usize i(0); i < num_contiguous_regions; i += output_batch_size) {
		std::vector<void*> bindings{
			scratch_f32_1.at_offset(width * height * 3, i),
			out_features.at_offset(out_feature_size, i),
		};

		// enqueue TransNet
		if (!clip_trt_ctx->enqueueV2(bindings.data(), stream, nullptr)) {
			throw std::runtime_error("enqueue failed!!!");
		}
	}
}
