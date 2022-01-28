
#include "video_decoder.h"
#include "NvCodecs/FFmpegDemuxer.h"
#include "NvCodecs/NvDecoder.h"
#include "../wrappers/CUDAStream.h"
#include "nv12_to_rgb_resize.cuh"

#undef max
#undef min

namespace video_deocder_details {

struct NV12Buffer {
	CUDADeviceMemoryUnique<u8> data;
	CUDAStream stream;
	usize capacity;
	usize count;
	usize element_size;
	bool processing;
	usize block_id;
	usize slot_id;

	NV12Buffer() :
		data(),
		stream(),
		capacity(0),
		count(0),
		element_size(0),
		processing(false),
		block_id(0),
		slot_id(0)
	{

	}
	NV12Buffer(NV12Buffer const& a) = delete;
	NV12Buffer& operator=(NV12Buffer const& a) = delete;
	NV12Buffer(NV12Buffer&& a) = delete;
	NV12Buffer& operator=(NV12Buffer&& a) = delete;

	void Init(usize num_frames, usize nv12_frame_size) {
		if (data.size() < nv12_frame_size * num_frames)
			data.reallocate(nv12_frame_size * num_frames);
		capacity = num_frames;
		count = 0;
		element_size = nv12_frame_size;
		processing = false;
	}

	std::tuple<std::size_t, usize, bool> Copy(std::uint8_t* frames[], std::size_t num_frames)
	{
		std::size_t remaining_spaces(capacity - count);
		std::size_t frames_to_copy(std::min(num_frames, remaining_spaces));
		for (std::size_t i(0); i < frames_to_copy; ++i)
			ck2(cuMemcpyAsync((CUdeviceptr)data.at_offset(element_size, count + i), (CUdeviceptr)frames[i], element_size, stream));
		count += frames_to_copy;
		return { num_frames - frames_to_copy, frames_to_copy, count >= capacity };
	}

	void ProcessNV12(CUdeviceptr out_rgb, usize block_id, usize slot_id, i32 width, i32 height, i32 out_width, i32 out_height) {
		assert(count > 0 && !processing);
		this->block_id = block_id;
		this->slot_id = slot_id;
		processing = true;
		nv12_to_rgb_resize(
			reinterpret_cast<u8*>(out_rgb),
			reinterpret_cast<u8 const* const>(data.ptr),
			count,
			element_size,
			width,
			height,
			out_width,
			out_height,
			stream
		);
	}

	void Clear() {
		count = 0;
	}

	bool Sync() {
		if (processing) {
			ck2(cuStreamSynchronize(stream));
			processing = false;
			return true;
		}
		return false;
	}
};

void video_deocder_thread(CUcontext cuda_context, VideoDecoderSyncState *states) {
	CUDAThreadContext cuda_thread_ctx(cuda_context);

	constexpr usize NUM_NV12_BUFFERS = 4;
	constexpr usize NUM_NV12_BUFFER_SLOT_PER_BUFFER = 9;

	NV12Buffer decoded_frames_buffer[NUM_NV12_BUFFERS];
	assert(states->frames_per_buffer % NUM_NV12_BUFFER_SLOT_PER_BUFFER == 0 && NUM_NV12_BUFFER_SLOT_PER_BUFFER > NUM_NV12_BUFFERS);
	usize frames_per_slot(states->frames_per_buffer / NUM_NV12_BUFFER_SLOT_PER_BUFFER);

	i32 max_frames_returned(0);
	u8** ppFrame{ nullptr };
	usize cur_video_id(0);

	while (states->running) try {
		auto filepath_opt(states->NextFile());
		if (!filepath_opt.has_value()) {
			using namespace std::chrono_literals;
			std::this_thread::sleep_for(10ms);
			continue;
		}
		auto [filepath, video_id] = *filepath_opt;
		cur_video_id = video_id;
		FFmpegDemuxer demuxer(filepath.c_str());

		BasicVideoInformation info;
		info.frame_per_second = demuxer.GetFPS();
		info.duration_us = demuxer.GetDuration();
		info.frame_count = demuxer.GetFrameCount();
		info.height = demuxer.GetHeight();
		info.width = demuxer.GetWidth();
		states->PushVideoInformation(video_id, info);

		i32 nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
		NvDecoder dec(cuda_context, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
		u8* pVideo = NULL, * pFrame;

		bool inited(false);
		usize finished_nv12_buffer_idx(0), scheduled_nv12_buffer_idx(0);
		usize finished_frame(0), decoded_frame(0);
		usize frame_count(0);

		usize active_memory_block(std::numeric_limits<usize>::max());
		usize num_slots_allocated_for_current_memory_block(0);
		usize frame_batch_offset(0);

		std::vector<std::tuple<usize, FrameBatch>> active_frame_batches;
		active_frame_batches.reserve(states->num_buffers);

		auto find_next_memory_block_slot = [
			&active_memory_block,
			&num_slots_allocated_for_current_memory_block,
			&states,
			NUM_NV12_BUFFER_SLOT_PER_BUFFER,
			frames_per_slot,
			&frame_batch_offset,
			NUM_NV12_BUFFERS,
			video_id,
			&active_frame_batches
		]() -> std::tuple<CUdeviceptr, usize, usize> {
			if (active_memory_block == std::numeric_limits<usize>::max()) {
				usize memory_block_id(states->NextMemoryBlock());
				FrameBatch fb(states, memory_block_id);
				fb.frames_gpu = states->frames.at_offset(states->memory_block_stride, memory_block_id);
				fb.frame_offset = frame_batch_offset;
				fb.number_of_frames = 0;
				fb.unique_video_id = video_id;
				active_frame_batches.emplace_back(memory_block_id, fb);
				num_slots_allocated_for_current_memory_block = 0;
				active_memory_block = memory_block_id;
				frame_batch_offset += states->frames_per_buffer;
			}
			usize block_id(active_memory_block);
			usize slot_id(num_slots_allocated_for_current_memory_block);
			++num_slots_allocated_for_current_memory_block;
			if (num_slots_allocated_for_current_memory_block >= NUM_NV12_BUFFER_SLOT_PER_BUFFER) {
				active_memory_block = std::numeric_limits<usize>::max();
			}
			CUdeviceptr slot_ptr((CUdeviceptr)states->frames.at_offset(states->memory_block_stride, block_id) + states->frame_stride * frames_per_slot * slot_id);
			return { slot_ptr, block_id, slot_id };
		};

		usize last_finished_slot(0);
		usize last_finished_block(0);
		bool end_of_video(false);

		auto put_to_memory_block = [&last_finished_block, &last_finished_slot, NUM_NV12_BUFFER_SLOT_PER_BUFFER, &active_frame_batches, &end_of_video, &states](NV12Buffer* buf) -> void {
			usize block_id(buf->block_id);
			usize slot_id(buf->slot_id);
			for (auto& [bid, fb] : active_frame_batches) {
				if (bid == block_id) {
					fb.number_of_frames += buf->count;
				}
			}
			last_finished_slot = slot_id;
			last_finished_block = block_id;
			if (slot_id == NUM_NV12_BUFFER_SLOT_PER_BUFFER - 1) {
				auto it(std::partition(active_frame_batches.begin(), active_frame_batches.end(), [block_id](std::tuple<usize, FrameBatch> const& item) { return std::get<0>(item) != block_id; }));
				FrameBatch fb(std::get<1>(*it));
				fb.end_of_video = end_of_video;
				states->EnqueueFrameBatch(fb);
				active_frame_batches.pop_back(); //active_frame_batches.erase(it, active_frame_batches.end());
			}
		};

		auto notify_end_of_stream = [&last_finished_block, &last_finished_slot, &active_frame_batches, &states]() -> void {
			assert(active_frame_batches.size() == 1 || active_frame_batches.size() == 0);
			if (active_frame_batches.size() == 1) {
				FrameBatch fb(std::get<1>(active_frame_batches.front()));
				fb.end_of_video = true;
				states->EnqueueFrameBatch(fb);
			}
		};

		while (states->running) {
			demuxer.Demux(&pVideo, &nVideoBytes);
			if (nVideoBytes == 0) {
				break;
			}
			nFrameReturned = dec.Decode(pVideo, nVideoBytes);
			if (nFrameReturned > max_frames_returned)
			{
				std::cout << "max_frames_returned:" << nFrameReturned << "\n";
				if (ppFrame)
					ppFrame = static_cast<u8**>(realloc(ppFrame, nFrameReturned * sizeof(u8*)));
				else
					ppFrame = static_cast<u8**>(malloc(nFrameReturned * sizeof(u8*)));
				if (!ppFrame)
					throw std::bad_alloc();
			}
			max_frames_returned = std::max(max_frames_returned, nFrameReturned);
			for (i32 i(0); i != nFrameReturned; ++i)
				ppFrame[i] = dec.GetFrame();
			auto ppFrameMut{ ppFrame };
			while (nFrameReturned > 0 && states->running) {
				if (!inited) {
					inited = true;
					if (dec.GetOutputFormat() != cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12)
						throw std::runtime_error("Only NV12 format is supported now!");
					for (usize i(0); i != NUM_NV12_BUFFERS; ++i) {
						decoded_frames_buffer[i].Init(frames_per_slot, dec.GetFrameSize());
					}
				}
				if (decoded_frames_buffer[scheduled_nv12_buffer_idx].Sync()) {
					finished_frame += decoded_frames_buffer[scheduled_nv12_buffer_idx].count;
					put_to_memory_block(std::addressof(decoded_frames_buffer[scheduled_nv12_buffer_idx]));
					decoded_frames_buffer[scheduled_nv12_buffer_idx].Clear();
				}
				auto [extra_frames, copied_frames, full] = decoded_frames_buffer[scheduled_nv12_buffer_idx].Copy(ppFrameMut, nFrameReturned);
				ppFrameMut += copied_frames;
				nFrameReturned -= copied_frames;
				if (full) {
					auto [ptr, memory_block_id, slot_id] = find_next_memory_block_slot();
					decoded_frames_buffer[scheduled_nv12_buffer_idx].ProcessNV12(ptr, memory_block_id, slot_id, demuxer.GetWidth(), demuxer.GetHeight(), states->output_width, states->output_height);
					scheduled_nv12_buffer_idx = (scheduled_nv12_buffer_idx + 1) % NUM_NV12_BUFFERS;
				}
				decoded_frame += copied_frames;
			}
		}
		// in case no extra buffers when notify_end_of_stream is called
		usize remainder(decoded_frame % states->frames_per_buffer);
		if (decoded_frame % states->frames_per_buffer == 0 || remainder > states->frames_per_buffer - frames_per_slot)
			end_of_video = true;
		// process remaining frames if any
		if (decoded_frames_buffer[scheduled_nv12_buffer_idx].count && !decoded_frames_buffer[scheduled_nv12_buffer_idx].processing) {
			auto [ptr, memory_block_id, slot_id] = find_next_memory_block_slot();
			decoded_frames_buffer[scheduled_nv12_buffer_idx].ProcessNV12(ptr, memory_block_id, slot_id, demuxer.GetWidth(), demuxer.GetHeight(), states->output_width, states->output_height);
			scheduled_nv12_buffer_idx = (scheduled_nv12_buffer_idx + 1) % NUM_NV12_BUFFERS;
		}

		// loop through ring buffer and sync with them all
		for (usize i(0); i != NUM_NV12_BUFFERS; ++i) {
			if (decoded_frames_buffer[scheduled_nv12_buffer_idx].Sync()) {
				finished_frame += decoded_frames_buffer[scheduled_nv12_buffer_idx].count;
				put_to_memory_block(std::addressof(decoded_frames_buffer[scheduled_nv12_buffer_idx]));
				decoded_frames_buffer[scheduled_nv12_buffer_idx].Clear();
			}
			scheduled_nv12_buffer_idx = (scheduled_nv12_buffer_idx + 1) % NUM_NV12_BUFFERS;
		}
		notify_end_of_stream();
	}
	catch (std::exception ex) {
		states->EnqueueErrorReport(cur_video_id, std::string(ex.what()));
	}
}

}

void FrameBatch::ConsumeBatch() {
	m_sync_states->MemoryBlockStateTransition(m_memory_block_id, VideoDecoderMemoryBlockState::Ready);
}
