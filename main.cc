
#include <iostream>
#include <string>

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
        //    std::cout << msg << std::endl;
    }
} g_trtLogger;

struct VideoProvider {
    fast_io::ibuf_file* file;

    VideoProvider(VideoProvider const& a) = delete;
    VideoProvider& operator=(VideoProvider const& a) = delete;
    VideoProvider(VideoProvider&& a) = delete;
    VideoProvider& operator=(VideoProvider&& a) = delete;

    VideoProvider(std::string_view filelist) : file(nullptr) {
        file = new fast_io::ibuf_file(filelist);
    }

    ~VideoProvider() noexcept {
        delete file;
    }

    std::optional<std::tuple<std::string, std::string>> TryFetchNextVideo() {
        std::string video_id;
        std::string filepath;
        try {
            scan(*file, video_id, filepath);
        }
        catch (...) {
            return {};
        }
        return { { video_id , filepath } };
    }
};

enum class TLVTags: u8 {
    VideoId = 0, //av or BV
    BasicInfo = 1,
    FeatureIndices = 2,
    FeatureContent = 3,
    ErrorInd = 4
};

struct ReorderBuffer {
    usize Reserve(usize size) {

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

int main(int argc, char **argv)
{
    if (argc != 3) {
        perrln("Usage: ", fast_io::mnp::os_c_str(argv[0]), " <saved-feature-file> <video-list-file>");
        return 1;
    }
    ck2(cuInit(0));
    constexpr u32 WIDTH = 288;
    constexpr u32 HEIGHT = 288;
    constexpr usize FRAME_STRIDE = WIDTH * HEIGHT * 3;

    CUDAContext context(0, 0);
    VideoProvider video_provider(argv[2]);
    fast_io::obuf_file obf(fast_io::mnp::os_c_str(argv[1]));
    println(" -- Saving to ", fast_io::mnp::os_c_str(argv[1]));
    println(" -- Sourcing videos from ", fast_io::mnp::os_c_str(argv[2]));

    NvInferRuntime trt_runtime(g_trtLogger);
    println(" -- Loading TensorRT engines");
    fast_io::flush(fast_io::c_stdout());
    nvinfer1::ICudaEngine* transnet_engine(LoadEngineFromFile("models/transnet.trt", trt_runtime.runtime));
    nvinfer1::ICudaEngine* clip_engine(LoadEngineFromFile("models/RN50x4.trt", trt_runtime.runtime));

    WorkerManager workers(context, transnet_engine, clip_engine, 1);

    VideoDecoder decoder(context, 5, 900, WIDTH, HEIGHT);
    usize finished_videos(0);
    usize pending_videos(0);

    usize videoid_ctr(0);

    robin_hood::unordered_flat_map<usize, BasicVideoInformation> video_infos;

    FPSCounter fps;
    auto start(std::chrono::high_resolution_clock::now());
    bool first(true), finished(false);

    std::byte tmp_length[4];
    auto put_length = [&tmp_length](u32 len) -> void {
        tmp_length[0] = static_cast<std::byte>((len >> 24) & 0xFF);
        tmp_length[1] = static_cast<std::byte>((len >> 16) & 0xFF);
        tmp_length[2] = static_cast<std::byte>((len >> 8) & 0xFF);
        tmp_length[3] = static_cast<std::byte>((len >> 0) & 0xFF);
    };
    auto write_tlv = [&obf, &put_length, &tmp_length](TLVTags tag, std::byte const* const data, u32 length) -> void {
        fast_io::put(obf, static_cast<u8>(tag));
        put_length(length);
        fast_io::write(obf, tmp_length, tmp_length + 4);
        fast_io::write(obf, data, data + length);
    };

    usize num_frames(0);
    usize num_features(0);

    println(" -- Processing videos");
    fast_io::flush(fast_io::c_stdout());
    while (finished_videos < pending_videos || first) {
        first = false;
        while (decoder.VideoEnqueueRecommended() && !finished) {
            auto const& next_video(video_provider.TryFetchNextVideo());
            if (next_video.has_value()) {
                auto const& [video_id, video_path] = *next_video;
                decoder.EnqueueDecode(video_path, videoid_ctr++);
                ++pending_videos;
                // put video ID
                write_tlv(TLVTags::VideoId, reinterpret_cast<std::byte const* const>(video_id.data()), video_id.size());
            }
            else {
                finished = true;
            }
        }
        {
            // prevent starvation
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(1ms);
        }
        auto info_opt(decoder.GetVideoInformation());
        if (info_opt.has_value()) {
            auto const& info(*info_opt);
            // put basic info
            write_tlv(TLVTags::BasicInfo, reinterpret_cast<std::byte const* const>(std::addressof(info)), sizeof(BasicVideoInformation));
        }
        auto error_report_opt(decoder.PollErrorReport());
        if (error_report_opt.has_value()) {
            auto& error_report(*error_report_opt);
            std::cerr << "Video " << error_report.video_id << " Error: " << error_report.message << "\n";
            // put error report
            write_tlv(TLVTags::ErrorInd, reinterpret_cast<std::byte const* const>(error_report.message.data()), error_report.message.size());
            ++finished_videos;
        }
        auto frame_batch_opt(decoder.PollNextBatch());
        if (frame_batch_opt.has_value()) {
            auto& frame_batch(*frame_batch_opt);
            num_frames += frame_batch.number_of_frames;
            WorkRequest req{ .frames = frame_batch, .custom_data = 0 };
            workers.SubmitWork(req);
        }
        auto finished_batch_opt(workers.PollResponse());
        if (finished_batch_opt) {
            auto& finished_batch(*finished_batch_opt);
            if (finished_batch.end_of_video) {
                ++finished_videos;
            }
            num_features += finished_batch.num_results;
            if (fps.Update(finished_batch.num_frames) || finished_batch.end_of_video) {
                print("FPS: ", fps.GetFPS(), " #Videos: ", finished_videos, " #Frames: ", num_frames, " #Features: ", num_features, "               \r");
                fast_io::flush(fast_io::c_stdout());
            }
            // put indices
            write_tlv(TLVTags::FeatureIndices, reinterpret_cast<std::byte const* const>(finished_batch.frame_indices), finished_batch.frame_indices_stride * finished_batch.num_results);
            // put features
            write_tlv(TLVTags::FeatureContent, reinterpret_cast<std::byte const* const>(finished_batch.features), finished_batch.features_stride * finished_batch.num_results);
            finished_batch.ConsumeResponse();
        }
    }
    std::cout << std::endl;
    auto end(std::chrono::high_resolution_clock::now());
    auto elpased(std::chrono::duration_cast<std::chrono::seconds>(end - start));
    println(" -- Done processing videos");
    println(" -- Frames decoded: ", num_frames, " features extracted: ", num_features);
    println(" -- Total ", elpased.count(), " seconds");
    fast_io::flush(fast_io::c_stdout());

    return 0;
}
