
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

#include "Utils/FPSCounter.h"

simplelogger::Logger* nvcodecs_logger = simplelogger::LoggerFactory::CreateConsoleLogger();

struct VideoProvider {
    VideoProvider(char const* filelist) {

    }
};

int main()
{
    ck2(cuInit(0));
    CUDAContext context(0, 0);

    robin_hood::unordered_set<usize> ongoing_videos;

    VideoDecoder decoder(context, 3, 300, 384, 384);
    decoder.EnqueueDecode("..\\test_videos\\mokou_remi.mp4", 123);
    decoder.EnqueueDecode("..\\test_videos\\2.mp4", 124);
    decoder.EnqueueDecode("..\\test_videos\\1.flv", 125);
    usize finished_videos(0);

    robin_hood::unordered_flat_map<usize, BasicVideoInformation> video_infos;

    FPSCounter fps;

    usize num_frames(0);
    for (; finished_videos < 3;) {
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
        auto error_report_opt(decoder.PollErrorReport());
        if (error_report_opt.has_value()) {
            auto& error_report(*error_report_opt);
            std::cout << "Video " << error_report.video_id << " Error: " << error_report.message << "\n";
            ++finished_videos;
        }
        auto frame_batch_opt(decoder.PollNextBatch());
        if (frame_batch_opt.has_value()) {
            auto& frame_batch(*frame_batch_opt);
            //std::cout << "====Frame Batch====\n";
            //std::cout << "Offset: " << frame_batch.frame_offset << "\n";
            //std::cout << "#Frame: " << frame_batch.number_of_frames << "\n";
            //std::cout << "EOS: " << frame_batch.end_of_video << "\n";
            //std::cout << "m_memory_block_id: " << frame_batch.m_memory_block_id << "\n";
            //std::cout << "===================\n";
            num_frames += frame_batch.number_of_frames;
            if (fps.Update(frame_batch.number_of_frames)) {
                std::cout << "FPS: " << fps.GetFPS() << "\n";
            }
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
            frame_batch.ConsumeBatch();
            if (frame_batch.end_of_video) {
                ++finished_videos;
            }
        }
    }
    decoder.Stop();
    std::cout << "End\n";
    std::cout << "frames decoded: " << num_frames << "\n";

    return 0;
}
