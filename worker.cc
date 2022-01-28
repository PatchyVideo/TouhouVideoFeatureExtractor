
#include "worker.h"
#include "wrappers/CUDADeviceMemory.h"
#include "wrappers/CUDAStream.h"
#include "TransNet/transnet.cuh"
#include "CLIP/clip.cuh"

namespace worker_details {

void worker_thread(CUcontext cuda_context, Worker* self) {
	CUDAThreadContext cuda_thread_ctx(cuda_context);
	CUDAStream stream;

	nvinfer1::IExecutionContext *transnet_ctx(self->transnet_engine->createExecutionContext());
	nvinfer1::IExecutionContext* clip_ctx(self->clip_engine->createExecutionContext());

	assert(transnet_ctx->setOptimizationProfile(0));
	assert(clip_ctx->setOptimizationProfile(0));

	auto transnet_dims(nvinfer1::Dims{ .nbDims = 5 });
	transnet_dims.d[0] = 1;
	transnet_dims.d[1] = 950;
	transnet_dims.d[2] = 3;
	transnet_dims.d[3] = 27;
	transnet_dims.d[4] = 48;
	transnet_ctx->setBindingDimensions(0, transnet_dims);
	clip_ctx->setBindingDimensions(0, nvinfer1::Dims4(8, 3, CLIP_HEIGHT, CLIP_WIDTH));

	CUDADeviceMemoryUnique<f32> scratch_f32_1;
	CUDADeviceMemoryUnique<f32> scratch_f32_2;
	CUDADeviceMemoryUnique<i32> scratch_i32_1;
	CUDADeviceMemoryUnique<u8> scratch_u8_1;

	CUDADeviceMemoryUnique<f32> transnet_scores(950);
	CUDAHostMemoryUnique<f32> transnet_scores_cpu(950);

	CUDADeviceMemoryUnique<f32> clip_features;
	CUDAHostMemoryUnique<f32> clip_features_cpu;

	std::vector<u32> key_frames;
	key_frames.reserve(900);
	std::vector<i32> feature_key_frames;
	feature_key_frames.reserve(920);

	constexpr i32 max_key_frame_separation = 300;
	constexpr i32 min_key_frame_separation = 3;

	while (self->running) {
		auto task_opt(self->GetJob());
		if (!task_opt.has_value())
			continue; // self->running == false
		key_frames.clear();
		feature_key_frames.clear();
		FrameBatch fb(task_opt->frames);
		usize custom_data(task_opt->custom_data);
		run_transnet(
			transnet_ctx,
			transnet_scores.at_offset(0, 0),
			scratch_f32_1,
			scratch_f32_2,
			fb.frames_gpu,
			CLIP_WIDTH,
			CLIP_HEIGHT,
			fb.number_of_frames,
			stream
		);
		transnet_scores.download_partial(transnet_scores_cpu, fb.number_of_frames + 50, stream);
		cudaStreamSynchronize(stream);
		key_frames.push_back(0);
		i32 last_frame(0);
		for (usize i(0); i < fb.number_of_frames; ++i) {
			f32 cur_score(transnet_scores_cpu[i + 25]);
			if (cur_score > 0.1 && i > 0 && i < fb.number_of_frames - 1 && i - last_frame > min_key_frame_separation) {
				key_frames.push_back(i);
				last_frame = i;
			}
		}
		key_frames.push_back(fb.number_of_frames - 1);
		for (usize i(1); i < key_frames.size(); ++i) {
			i32 left(key_frames[i - 1]), right(key_frames[i]);
			if (right - left > max_key_frame_separation) {
				i32 n_divide(((right - left - 1) / max_key_frame_separation) + 1);
				i32 frame_advance((right - left) / n_divide);
				while (left + frame_advance / 2 <= right) {
					feature_key_frames.push_back(left + frame_advance / 2);
					left += frame_advance;
				}
			}
			else {
				feature_key_frames.push_back((left + right) / 2);
			}
		}
		usize num_features(feature_key_frames.size());
		run_clip(
			clip_ctx,
			clip_features,
			fb.frames_gpu,
			fb.number_of_frames,
			feature_key_frames.data(),
			num_features,
			CLIP_WIDTH,
			CLIP_HEIGHT,
			scratch_f32_1,
			scratch_u8_1,
			scratch_i32_1,
			stream
		);
		clip_features.download_partial(clip_features_cpu, num_features * CLIP_FEATURE_SIZE, stream);
		cudaStreamSynchronize(stream);
		fb.ConsumeBatch();
		for (auto& idx : feature_key_frames) {
			idx += fb.frame_offset;
		}
		self->SubmitResponse(
			WorkResponse(
				self,
				num_features,
				reinterpret_cast<u8 const* const>(clip_features_cpu.at_offset(0, 0)),
				sizeof(f32) * CLIP_FEATURE_SIZE,
				feature_key_frames.data(),
				sizeof(*feature_key_frames.data()),
				custom_data
			)
		);
	}

	delete transnet_ctx;
	delete clip_ctx;
}

};

void WorkResponse::ConsumeResponse() {
	std::lock_guard<std::mutex> guard(*associated_worker->free_workers_mutex);
	associated_worker->free_workers->push_back(associated_worker->worker_id);
	associated_worker->doing_work = false;
}
