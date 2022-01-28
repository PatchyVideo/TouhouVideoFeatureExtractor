
#include <iostream>

#include "fast_io/fast_io.h"
#include "fast_io/fast_io_device.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "common.h"
#include "stl_replacements/robin_hood.h"

#include "wrappers/CUDAContext.h"
#include "wrappers/NvInferRuntime.h"

#include "VideoDecoder/NvCodecs/Logger.h"
#include "VideoDecoder/video_decoder.h"

#include "Utils/FPSCounter.h"
#include "Utils/lodepng.h"

#include "worker.h"

#include <npp.h>

simplelogger::Logger* nvcodecs_logger = simplelogger::LoggerFactory::CreateConsoleLogger();

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        //if (severity != Severity::kINFO)
        std::cout << msg << std::endl;
    }
} g_trtLogger;

struct VideoProvider {
    VideoProvider(char const* filelist) {

    }

    std::optional<std::string> TryFetchNextVideo() {
        return {};
    }
};

struct ReorderBuffer {
    void Reserve(usize pos, usize size) {

    }
    template<
        fast_io::stream handletype,
        fast_io::buffer_mode mde,
        typename decorators,
        std::size_t bfs, ::fast_io::freestanding::random_access_iterator Iter
    > requires (((mde& fast_io::buffer_mode::out) == fast_io::buffer_mode::out) && fast_io::details::allow_iobuf_punning<typename decorators::internal_type, Iter>)
    bool Complete(usize pos, fast_io::basic_io_buffer<handletype, mde, decorators, bfs>& bios, u8 const* const data) {

    }
};

nvinfer1::ICudaEngine* LoadEngineFromFile(std::string_view filename, nvinfer1::IRuntime* runtime)
{
    std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size))
    {
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
        return engine;
    }
    return nullptr;
}

int main()
{
    //char const* hello = "Hello";
    //fast_io::obuf_file obf(fast_io::mnp::os_c_str("1.txt"));
    //fast_io::write(obf, hello, hello + 2);
    //print("Hello\n");
    //return 0;
    ck2(cuInit(0));
    constexpr u32 WIDTH = 288;
    constexpr u32 HEIGHT = 288;
    constexpr usize FRAME_STRIDE = WIDTH * HEIGHT * 3;

    CUDAContext context(0, 0);

    NvInferRuntime trt_runtime(g_trtLogger);
    nvinfer1::ICudaEngine* transnet_engine(LoadEngineFromFile("models/transnet.trt", trt_runtime.runtime));
    nvinfer1::ICudaEngine* clip_engine(LoadEngineFromFile("models/RN50x4.trt", trt_runtime.runtime));

    WorkerManager workers(context, transnet_engine, clip_engine, 1);

    robin_hood::unordered_set<usize> ongoing_videos;

    VideoDecoder decoder(context, 5, 900, WIDTH, HEIGHT);
    //decoder.EnqueueDecode("..\\test_videos\\mokou_remi.mp4", 123);
    //decoder.EnqueueDecode("..\\test_videos\\2.mp4", 124);
    decoder.EnqueueDecode("..\\test_videos\\1.flv", 125);
    usize finished_videos(0);

    robin_hood::unordered_flat_map<usize, BasicVideoInformation> video_infos;

    FPSCounter fps;
    auto start(std::chrono::high_resolution_clock::now());
    bool first(true);

    usize num_frames(0);
    for (; finished_videos < 1;) {
        {
            // prevent starvation
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(10ms);
        }
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
            if (first) {
                first = false;
                usize frame_idx(frame_batch.number_of_frames - 1);
                u8* frame_chw_gpu(frame_batch.frames_gpu + FRAME_STRIDE * frame_idx);
                std::vector<u8> frame_chw(FRAME_STRIDE);
                std::vector<u8> frame_hwc(WIDTH * HEIGHT * 4);
                cuMemcpyDtoH(frame_chw.data(), (CUdeviceptr)frame_chw_gpu, WIDTH * HEIGHT * 3);
                for (usize i(0); i != HEIGHT; ++i) {
                    for (usize j(0); j != WIDTH; ++j) {
                        u8* dst(frame_hwc.data() + (i * WIDTH + j) * 4);
                        u8* src_r(frame_chw.data() + (WIDTH * HEIGHT) * 0 + i * WIDTH + j);
                        u8* src_g(frame_chw.data() + (WIDTH * HEIGHT) * 1 + i * WIDTH + j);
                        u8* src_b(frame_chw.data() + (WIDTH * HEIGHT) * 2 + i * WIDTH + j);
                        dst[0] = *src_r;
                        dst[1] = *src_g;
                        dst[2] = *src_b;
                        dst[3] = 255;
                    }
                }
                lodepng::encode("..\\test_videos\\1.flv.png", frame_hwc, WIDTH, HEIGHT);
                WorkRequest req{ .frames = frame_batch, .custom_data = 0 };
                workers.SubmitWork(req);
                for (;;) {
                    auto r(workers.PollResponse());
                    if (r.has_value()) {
                        auto& r2(*r);
                        std::cout << "n_feats: " << r2.num_results << "\n";
                        r2.ConsumeResponse();
                        break;
                    }
                }
            }
            if (fps.Update(frame_batch.number_of_frames)) {
                std::cout << "FPS: " << fps.GetFPS() << "\n";
            }
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(2000ms);
            frame_batch.ConsumeBatch();
            if (frame_batch.end_of_video) {
                ++finished_videos;
            }
        }
    }
    std::cout << "End\n";
    std::cout << "frames decoded: " << num_frames << "\n";
    auto end(std::chrono::high_resolution_clock::now());
    auto elpased(std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
    std::cout << "Total " << elpased.count() << " ms\n";

    delete transnet_engine;
    delete clip_engine;

    return 0;
}
