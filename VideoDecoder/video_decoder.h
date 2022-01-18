#pragma once

#include "../common.h"
#include "../wrappers/CUDADeviceMemory.h"
#include "../wrappers/CUDAHostMemory.h"
#include "../wrappers/CUDAThreadContext.h"

namespace video_deocder_impl {
	void video_deocder_thread(
		CUcontext cuda_context,
		VideoDecoderSyncState* states
	);
}

enum class VideoDecoderMemoryBlockState {
	Ready,
	Filling,
	Filled,
	InUse
};

struct VideoDecoderSyncState {
	VideoDecoderSyncState(usize num_buffers, usize frames_per_buffer, u32 resize_width, u32 resize_height) {
		this->num_buffers = num_buffers;
		frame_stride = 3 * resize_width * resize_height;
		memory_block_stride = frames_per_buffer * frame_stride;
		frames.reallocate(memory_block_stride * num_buffers);
		memory_block_states.reserve(num_buffers);
		for (usize i{ 0 }; i != num_buffers; ++i)
			memory_block_states.push_back(VideoDecoderMemoryBlockState::Ready);
	}

	void MemoryBlockStateTransition(usize id, VideoDecoderMemoryBlockState state) {
		std::lock_guard<std::mutex> guard(memory_block_mutex);
		memory_block_states[id] = state;
	}

	void MemoryBlockStateTransitionLockFree(usize id, VideoDecoderMemoryBlockState state) {
		memory_block_states[id] = state;
	}

	std::optional<std::tuple<std::string, usize>> NextFile() {
		std::lock_guard<std::mutex> guard(file_queue_mutex);
		if (file_queue.size() != 0) {
			auto s(file_queue.front());
			file_queue.pop();
			return { s };
		}
		else {
			return {};
		}
	}

	void EnqueueFile(std::string filepath, usize id) {
		std::lock_guard<std::mutex> guard(file_queue_mutex);
		file_queue.push({ filepath, id });
	}

	void Stop() {
		running = false;
	}
	bool running;
private:
	std::queue<std::tuple<std::string, usize>> file_queue;
	std::mutex file_queue_mutex;
	CUDADeviceMemoryUnique<f32> frames;
	std::vector<VideoDecoderMemoryBlockState> memory_block_states;
	std::mutex memory_block_mutex;
	usize num_buffers;
	usize memory_block_stride;
	usize frame_stride;
};

struct BasicVideoInformation {
	f64 frame_per_second;
	u32 width;
	u32 height;
	u64 duration_us;
	u64 frame_count;
};

struct FrameBatch {
	f32* frames_gpu; // NC(=3)HW bytes in GPU
	usize unique_video_id; // unique video id
	usize number_of_frames; // number of frames returned
	usize frame_offset; // offset of the first frame in this batch relative to video beginning
	bool end_of_video; // true indicates end of video


	/**
	*   @brief  Indicates this region of memory is available for storing other frames now
	*   @param  None
	*   @return void
	*/
	void ConsumeBatch() {
		m_sync_states->MemoryBlockStateTransition(m_memory_block_id, VideoDecoderMemoryBlockState::Ready);
	}
private:
	VideoDecoderSyncState* m_sync_states;
	usize m_memory_block_id;
};

struct VideoDecoder {
	VideoDecoder(CUcontext cuda_context, usize num_buffers, usize frames_per_buffer, u32 resize_width, u32 resize_height) :
		m_cuda_context(cuda_context),
		m_sync_states(num_buffers, frames_per_buffer, resize_width, resize_height)
	{

	}

	~VideoDecoder() {
		try {
			// TODO: remember to release all resources
		}
		catch (...) {

		}
	}

	/**
	*   @brief  Enqueue a video to be processed
	*   @param  video_filename - Filepath to the video
	*   @param  unique_video_id - A unique ID used to identify this video in all subsequent operations
	*   @return void
	*/
	void EnqueueDecode(std::string video_filename, usize unique_video_id) {
		m_sync_states.EnqueueFile(video_filename, unique_video_id);
	}

	/**
	*   @brief  Returns the basic information of a video enqued
	*   @param  None
	*   @return The video's info if the video is being processed
	*/
	std::optional<BasicVideoInformation> GetVideoInformation() {
		return {};
	}

	/**
	*   @brief  Returns the next ready batch if any, this will take ownership of the resultant frames
	*   @param  None
	*   @return WIP
	*/
	std::optional<FrameBatch> PollNextBatch() {
		return {};
	}

	std::optional<
		std::tuple<
			usize, // unique video id
			std::string // hunman readable error message
		>
	> PollErrorState() {
		return {};
	}

private:
	CUcontext m_cuda_context;
	VideoDecoderSyncState m_sync_states;
};
