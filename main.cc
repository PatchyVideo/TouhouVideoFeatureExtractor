
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "common.h"

#include "wrappers/CUDAContext.h"

#include "test_kernel.cuh"
#include "VideoDecoder/NvCodecs/Logger.h"

#include "VideoDecoder/video_decoder.h"
#include "stl_replacements/robin_hood.h"

simplelogger::Logger* nvcodecs_logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int main()
{
    ck2(cuInit(0));
    CUDAContext context(0, 0);

    VideoDecoder decoder(context, 3, 100, 384, 384);
    decoder.EnqueueDecode("..\\test_videos\\1.flv", 123);

    robin_hood::unordered_flat_map<usize, BasicVideoInformation> video_infos;

    usize num_frames(0);
    for (;;) {
        auto info_opt(decoder.GetVideoInformation());
        if (info_opt.has_value()) {
            auto const& info(*info_opt);
            std::cout << "====Video Info====\n";
            std::cout << "Video ID: " << info.video_id << "\n";
            std::cout << "Width: " << info.width << "\n";
            std::cout << "Height: " << info.height << "\n";
            std::cout << "FPS: " << info.frame_per_second << "\n";
            std::cout << "Frame count: " << info.frame_count << "\n";
            std::cout << "Duration: " << info.duration_us << "\n";
            std::cout << "==================\n";
        }
        auto frame_batch_opt(decoder.PollNextBatch());
        if (frame_batch_opt.has_value()) {
            auto& frame_batch(*frame_batch_opt);
            std::cout << "====Frame Batch====\n";
            std::cout << "Offset: " << frame_batch.frame_offset << "\n";
            std::cout << "#Frame: " << frame_batch.number_of_frames << "\n";
            std::cout << "EOS: " << frame_batch.end_of_video << "\n";
            std::cout << "m_memory_block_id: " << frame_batch.m_memory_block_id << "\n";
            std::cout << "===================\n";
            num_frames += frame_batch.number_of_frames;
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(100ms);
            frame_batch.ConsumeBatch();
            if (frame_batch.end_of_video)
                break;
        }
    }
    decoder.Stop();
    std::cout << "End\n";
    std::cout << "frames decoded: " << num_frames << "\n";

    return 0;
}
