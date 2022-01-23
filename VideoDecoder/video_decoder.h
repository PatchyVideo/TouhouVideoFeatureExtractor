#pragma once

#include "../common.h"
#include "../wrappers/CUDADeviceMemory.h"
#include "../wrappers/CUDAHostMemory.h"
#include "../wrappers/CUDAThreadContext.h"


struct VideoDecoderSyncState;

namespace video_deocder_details {

	void video_deocder_thread(
		CUcontext cuda_context,
		VideoDecoderSyncState* states
	);

	constexpr usize VIDEO_DECODER_DEFAULT_QUEUE_SIZE = 20;
	constexpr usize VIDEO_DECODER_DEFAULT_THREAD_COUNT = 1; // <-- set to 5 for NVIDIA A100, set to 4 for NVIDIA A30, see https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new#Decoder
}

struct BasicVideoInformation {
	f64 frame_per_second;
	u32 width;
	u32 height;
	u64 duration_us;
	u64 frame_count;
	usize video_id;
};

struct VideoDecoderErrorReport {
	usize video_id;
	std::string message;
};

enum class VideoDecoderMemoryBlockState {
	Ready,
	Filling,
	Filled,
	InUse
};

struct VideoDecoderSyncState;

struct FrameBatch {
	u8* frames_gpu; // NC(=3)HW bytes in GPU
	usize unique_video_id; // unique video id
	usize number_of_frames; // number of frames returned
	usize frame_offset; // offset of the first frame in this batch relative to video beginning
	bool end_of_video; // true indicates end of video

	FrameBatch(VideoDecoderSyncState* sync_state, usize block_id) :
		frames_gpu(nullptr),
		unique_video_id(0),
		number_of_frames(0),
		frame_offset(0),
		end_of_video(false),
		m_sync_states(sync_state),
		m_memory_block_id(block_id)
	{

	}

	/**
	*   @brief  Indicates this region of memory is available for storing other frames now
	*   @param  None
	*   @return void
	*/
	void ConsumeBatch();

	VideoDecoderSyncState* m_sync_states;
	usize m_memory_block_id;
};

struct VideoDecoderSyncState {
	VideoDecoderSyncState(usize num_buffers, usize frames_per_buffer, u32 resize_width, u32 resize_height) {
		running = true;
		output_width = resize_width;
		output_height = resize_height;
		this->num_buffers = num_buffers;
		this->frames_per_buffer = frames_per_buffer;
		frame_stride = 3 * static_cast<usize>(resize_width) * static_cast<usize>(resize_height);
		memory_block_stride = frames_per_buffer * frame_stride;
		frames.reallocate(memory_block_stride * num_buffers);
		memory_block_states.reserve(num_buffers);
		free_memory_blocks.reserve(num_buffers);
		for (usize i{ 0 }; i != num_buffers; ++i) {
			memory_block_states.push_back(VideoDecoderMemoryBlockState::Ready);
			free_memory_blocks.push_back(i);
		}
		video_info.reserve(video_deocder_details::VIDEO_DECODER_DEFAULT_QUEUE_SIZE);
	}

	VideoDecoderSyncState(VideoDecoderSyncState const& a) = delete;
	VideoDecoderSyncState& operator=(VideoDecoderSyncState const& a) = delete;
	VideoDecoderSyncState(VideoDecoderSyncState&& a) = delete;
	VideoDecoderSyncState& operator=(VideoDecoderSyncState&& a) = delete;

	void MemoryBlockStateTransition(usize id, VideoDecoderMemoryBlockState state) {
		std::unique_lock<std::mutex> lock(memory_block_mutex);
		memory_block_states[id] = state;
		if (state == VideoDecoderMemoryBlockState::Ready) {
			free_memory_blocks.push_back(id);
			lock.unlock();
			memory_block_cv.notify_one();
		}
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

	void EnqueueFrameBatch(FrameBatch fb) {
		std::lock_guard<std::mutex> guard(memory_block_mutex);
		memory_block_states[fb.m_memory_block_id] = VideoDecoderMemoryBlockState::Filled;
		filled_memory_blocks.push(fb);
	}

	std::optional<FrameBatch> DequeueFrameBatch() {
		std::lock_guard<std::mutex> guard(memory_block_mutex);
		if (filled_memory_blocks.size()) {
			auto ret(filled_memory_blocks.front());
			filled_memory_blocks.pop();
			memory_block_states[ret.m_memory_block_id] = VideoDecoderMemoryBlockState::InUse;
			return { ret };
		}
		else {
			return {};
		}
	}

	void EnqueueFile(std::string filepath, usize id) {
		std::lock_guard<std::mutex> guard(file_queue_mutex);
		file_queue.push({ filepath, id });
	}

	bool VideoEnqueueRecommended() {
		std::lock_guard<std::mutex> guard(file_queue_mutex);
		return file_queue.size() <= video_deocder_details::VIDEO_DECODER_DEFAULT_QUEUE_SIZE;
	}

	void Stop() {
		running = false;
	}

	usize NextMemoryBlock() {
		std::unique_lock<std::mutex> lock(memory_block_mutex);
		memory_block_cv.wait(lock, [this] { return free_memory_blocks.size() != 0; });
		usize id(free_memory_blocks.back());
		free_memory_blocks.pop_back();
		memory_block_states[id] = VideoDecoderMemoryBlockState::Filling;
		return id;
	}

	void PushVideoInformation(usize id, BasicVideoInformation info) {
		std::lock_guard<std::mutex> guard(video_info_mutex);
		info.video_id = id;
		video_info.push_back(info);
	}

	std::optional<BasicVideoInformation> PopVideoInformation(usize id) {
		std::lock_guard<std::mutex> guard(video_info_mutex);
		for (auto const& info : video_info) {
			if (info.video_id == id)
				return { info };
		}
		return {};
	}

	std::optional<BasicVideoInformation> PopVideoInformation() {
		std::lock_guard<std::mutex> guard(video_info_mutex);
		if (video_info.size()) {
			auto ret(video_info.back());
			video_info.pop_back();
			return { ret };
		}
		return {};
	}

	bool running;
	CUDADeviceMemoryUnique<u8> frames;

	void EnqueueErrorReport(usize vid, std::string report) {
		std::lock_guard<std::mutex> guard(error_reports_mutex);
		VideoDecoderErrorReport r{ vid, report };
		error_reports.push(r);
	}

	std::optional<VideoDecoderErrorReport> PopErrorReport() {
		std::lock_guard<std::mutex> guard(error_reports_mutex);
		if (error_reports.size()) {
			auto ret(error_reports.front());
			error_reports.pop();
			return { ret };
		}
		return {};
	}

	usize num_buffers;
	usize memory_block_stride;
	usize frame_stride;
	usize frames_per_buffer;

	u32 output_width;
	u32 output_height;
private:
	std::queue<std::tuple<std::string, usize>> file_queue;
	std::mutex file_queue_mutex;

	std::vector<VideoDecoderMemoryBlockState> memory_block_states;
	std::vector<usize> free_memory_blocks;
	std::queue<FrameBatch> filled_memory_blocks;
	std::mutex memory_block_mutex;
	std::condition_variable memory_block_cv;

	std::vector<BasicVideoInformation> video_info;
	std::mutex video_info_mutex;

	std::queue<VideoDecoderErrorReport> error_reports;
	std::mutex error_reports_mutex;
};

struct VideoDecoder {
	VideoDecoder(CUcontext cuda_context, usize num_buffers, usize frames_per_buffer, u32 resize_width, u32 resize_height) :
		m_cuda_context(cuda_context),
		m_sync_states(num_buffers, frames_per_buffer, resize_width, resize_height)
	{
		for (usize i(0); i != video_deocder_details::VIDEO_DECODER_DEFAULT_THREAD_COUNT; ++i)
			m_thread[i] = std::thread(video_deocder_details::video_deocder_thread, cuda_context, &m_sync_states);
	}

	VideoDecoder(VideoDecoder const& a) = delete;
	VideoDecoder& operator=(VideoDecoder const& a) = delete;

	~VideoDecoder() noexcept {
		try {
			if (m_sync_states.running) {
				Stop();
				Join();
			}
		}
		catch (...) {

		}
	}

	void Join() {
		for (auto& t : m_thread)
			t.join();
	}

	/**
	*   @brief  Tells caller if the video decoder recommends enqueue a new video
	*   @param  None
	*   @return bool
	*/
	bool VideoEnqueueRecommended() {
		return m_sync_states.VideoEnqueueRecommended();
	}

	/**
	*   @brief  Enqueue a video to be processed
	*   @param  video_filename - Filepath to the video
	*   @param  unique_video_id - A unique ID used to identify this video in all subsequent operations
	*   @return void
	*/
	void EnqueueDecode(std::string video_filename, usize unique_video_id) {
		if (!m_sync_states.running)
			return;
		m_sync_states.EnqueueFile(video_filename, unique_video_id);
	}

	/**
	*   @brief  Returns the basic information of a video enqueued
	*   @param  id - Video ID
	*   @return The video's info if the video is being processed
	*/
	std::optional<BasicVideoInformation> GetVideoInformation(usize id) {
		if (!m_sync_states.running)
			return {};
		return m_sync_states.PopVideoInformation(id);
	}

	/**
	*   @brief  Returns the basic information of a video enqueued
	*   @param  
	*   @return The video's info if the video is being processed
	*/
	std::optional<BasicVideoInformation> GetVideoInformation() {
		if (!m_sync_states.running)
			return {};
		return m_sync_states.PopVideoInformation();
	}

	/**
	*   @brief  Returns the next ready batch if any, this will take ownership of the resultant frames
	*   @param  None
	*   @return WIP
	*/
	std::optional<FrameBatch> PollNextBatch() {
		if (!m_sync_states.running)
			return {};
		return m_sync_states.DequeueFrameBatch();
	}

	std::optional<VideoDecoderErrorReport> PollErrorReport() {
		if (!m_sync_states.running)
			return {};
		return m_sync_states.PopErrorReport();
	}

	void Stop() {
		if (m_sync_states.running) {
			m_sync_states.running = false;
			for (usize i(0); i != video_deocder_details::VIDEO_DECODER_DEFAULT_THREAD_COUNT; ++i)
				m_thread[i].join();
		}
	}

private:
	CUcontext m_cuda_context;
	VideoDecoderSyncState m_sync_states;
	std::thread m_thread[video_deocder_details::VIDEO_DECODER_DEFAULT_THREAD_COUNT];
};
