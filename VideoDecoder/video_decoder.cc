
#include "video_decoder.h"
#include "NvCodecs/FFmpegDemuxer.h"

namespace video_deocder_impl {

void video_deocder_thread(CUcontext cuda_context, VideoDecoderSyncState *states) {
	CUDAThreadContext cuda_thread_ctx(cuda_context);

	while (states->running) {
		try {
			auto filepath_opt(states->NextFile());
			if (!filepath_opt.has_value()) {
				std::this_thread::yield();
			}
			auto [filepath, video_id] = *filepath_opt;
			FFmpegDemuxer demuxer(filepath.c_str());

		}
		catch (std::exception ex) {

		}
	}
}

}

