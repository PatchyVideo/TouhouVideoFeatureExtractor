
#include "nv12_to_rgb_resize.cuh"
#include <device_launch_parameters.h>

template<typename T>
__device__ __forceinline__ void write_pixel(T* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, T r, T g, T b, std::int32_t width, std::int32_t height)
{
	*(rgb + width * height * 3 * batch + width * height * 0 + width * y + x) = r;
	*(rgb + width * height * 3 * batch + width * height * 1 + width * y + x) = g;
	*(rgb + width * height * 3 * batch + width * height * 2 + width * y + x) = b;
}

__device__ __forceinline__ float3 read_pixel(float* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height)
{
	float r, g, b;
	r = *(rgb + width * height * 3 * batch + width * height * 0 + width * y + x);
	g = *(rgb + width * height * 3 * batch + width * height * 1 + width * y + x);
	b = *(rgb + width * height * 3 * batch + width * height * 2 + width * y + x);
	return float3{ r, g, b };
}

__device__ __forceinline__ uchar3 read_pixel(std::uint8_t* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height)
{
	std::uint8_t r, g, b;
	r = *(rgb + width * height * 3 * batch + width * height * 0 + width * y + x);
	g = *(rgb + width * height * 3 * batch + width * height * 1 + width * y + x);
	b = *(rgb + width * height * 3 * batch + width * height * 2 + width * y + x);
	return uchar3{ r, g, b };
}

__device__ __forceinline__ std::uint8_t saturate_u8(float v)
{
	return static_cast<std::uint8_t>(std::max(std::min(v, 255.0f), 0.0f));
}

__device__ __forceinline__ uchar3 yuv2rgb(std::uint8_t y_, std::uint8_t u_, std::uint8_t v_)
{
	float luma(static_cast<float>(y_));
	float u(static_cast<float>(u_));
	float v(static_cast<float>(v_));
	return uchar3{
		saturate_u8(luma + (1.40200f * (v - 128.0f))),
		saturate_u8(luma - (0.34414f * (u - 128.0f)) - (0.71414f * (v - 128.0f))),
		saturate_u8(luma + (1.77200f * (u - 128.0f))),
	};
}

__device__ __forceinline__ float l1_dist(float3 const& a, float3 const& b)
{
	return std::fabs(a.x - b.x) + std::fabs(a.y - b.y) + std::fabs(a.z - b.z);
}

__device__ __forceinline__ float sqr(float const a)
{
	return a * a;
}

__device__ __forceinline__ float3 operator*(float const& lhs, float3 const& rhs)
{
	return float3{ lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };
}

__device__ __forceinline__ float3 operator/(float3 const& lhs, float const& rhs)
{
	return float3{ lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };
}

__device__ __forceinline__ float3 operator+(float3 const& lhs, float3 const& rhs)
{
	return float3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}

#define RESIZE_BATCH_SIZE 10

__global__ void nv12_to_rgb_resize_kernel(
	u8 const* const nv12,
	u8* out_rgb,
	i32 const input_width,
	i32 const input_height,
	usize const input_frame_count,
	usize const input_frame_size,
	i32 const output_width,
	i32 const output_height
)
{
	i32 ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y);

	// step 1: use border color
	i32 reflected_x(std::min(ix, output_width - 1));
	i32 reflected_y(std::min(iy, output_height - 1));
	// step 2: get sampling location in original image coordinate
	float2 uv{ __int2float_rz(reflected_x) / __int2float_rz(output_width - 1), __int2float_rz(reflected_y) / __int2float_rz(output_height - 1) };
	int2 pos{ __float2int_rz(uv.x * __int2float_rz(input_width - 1) + 0.5f),  __float2int_rz(uv.y * __int2float_rz(input_height - 1) + 0.5f) };
	int2 pos_half{ __float2int_rz(uv.x * __int2float_rz(input_width / 2 - 1) + 0.5f),  __float2int_rz(uv.y * __int2float_rz(input_height / 2 - 1) + 0.5f) };

	i32 const z_max_pos{ std::min(input_frame_count, static_cast<usize>((blockIdx.z + 1) * RESIZE_BATCH_SIZE)) };
	for (std::int32_t iz{ blockIdx.z * RESIZE_BATCH_SIZE }; iz < z_max_pos; ++iz) {
		// step 3: get YUV color from NV12 layout
		u8 color_y(*(nv12 + input_frame_size * iz + pos.y * input_width + pos.x));
		auto chroma_base(reinterpret_cast<uchar2 const* const>(nv12 + input_frame_size * iz + input_width * input_height));
		auto color_uv(chroma_base[input_width / 2 * pos_half.y + pos_half.x]);
		// step 4: convert to RGB bytes
		auto rgb(yuv2rgb(color_y, color_uv.x, color_uv.y));
		// step 5: store in tmp array
		write_pixel(out_rgb, iz, ix, iy, rgb.x, rgb.y, rgb.z, output_width, output_height);
	}
}

void calculate_lpf_params(
	i32 input_width,
	i32 input_height,
	i32 output_width,
	i32 output_height,
	u32 *gaussian_ks,
	f32 *gaussian_sigma
) {

}

void nv12_to_rgb_resize(
	u8 *out_resized_rgb_frames,
	u8 const *const batched_yuv_frames,
	usize batch_size,
	usize nv12_frame_size_in_bytes,
	i32 input_width,
	i32 input_height,
	i32 output_width,
	i32 output_height,
	CUstream stream
) {
	assert(output_width % 32 == 0 && output_height % 32 == 0);

	dim3 block(32, 32, 1);
	dim3 grid(output_width / 32, output_height / 32, (batch_size - 1) / RESIZE_BATCH_SIZE + 1);

	u32 gaussian_ks;
	f32 gaussian_sigma;
	calculate_lpf_params(input_width, input_height, output_width, output_height, std::addressof(gaussian_ks), std::addressof(gaussian_sigma));
	// TODO: apply gaussian blur to remove higher frequencies before downsampling

	nv12_to_rgb_resize_kernel<<<grid, block, 0, stream>>>(
		batched_yuv_frames,
		out_resized_rgb_frames,
		input_width,
		input_height,
		batch_size,
		nv12_frame_size_in_bytes,
		output_width,
		output_height
		);
	ck2(cudaGetLastError());
}

#undef RESIZE_BATCH_SIZE
