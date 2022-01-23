#pragma once

#include "common.h"
#include "VideoDecoder/video_decoder.h"

#undef GetJob

struct WorkRequest {
	FrameBatch frames;
};

namespace worker_details {
	struct Worker;
}

struct WorkResponse {
	usize num_results;
	u8 const* const features;
	usize features_stride;
	u8 const* const frame_indices;
	usize frame_indices_stride;

	void ConsumeResponse();
private:
	worker_details::Worker* associated_worker;
};


namespace worker_details {

void worker_thread(Worker* self);

struct Worker {
	Worker() = default;
	Worker(
		usize worker_id,

		std::queue<WorkResponse>* finished_works,
		std::mutex* finished_works_mutex,

		std::vector<usize>* free_workers,
		std::mutex* free_workers_mutex
	) :
		doing_work(false),
		running(false),
		worker_id(worker_id),
		finished_works(finished_works),
		finished_works_mutex(finished_works_mutex),
		free_workers(free_workers),
		free_workers_mutex(free_workers_mutex),
		submitted_job_mutex(nullptr),
		submitted_job_cv(nullptr)
	{
		submitted_job_mutex = new std::mutex;
		submitted_job_cv = new std::condition_variable;
		thread = std::thread(worker_thread, this);
	}

	Worker(Worker const& a) = delete;
	Worker& operator=(Worker const& a) = delete;
	Worker(Worker&& other) noexcept :
		worker_id(other.worker_id),
		finished_works(other.finished_works),
		finished_works_mutex(other.finished_works_mutex),
		free_workers(other.free_workers),
		free_workers_mutex(other.free_workers_mutex),
		submitted_job(std::move(other.submitted_job)),
		submitted_job_mutex(std::move(other.submitted_job_mutex)),
		submitted_job_cv(std::move(other.submitted_job_cv)),
		thread(std::move(other.thread)),
		doing_work(other.doing_work),
		running(other.running)
	{
		other.submitted_job_mutex = nullptr;
		other.submitted_job_cv = nullptr;
		other.doing_work = false;
		other.running = false;
	}
	Worker& operator=(Worker&& other) noexcept
	{
		if (std::addressof(other) != this)
		{
			worker_id = other.worker_id;
			finished_works = other.finished_works;
			finished_works_mutex = other.finished_works_mutex;
			free_workers = other.free_workers;
			free_workers_mutex = other.free_workers_mutex;
			submitted_job = std::move(submitted_job);
			submitted_job_mutex = other.submitted_job_mutex;
			other.submitted_job_mutex = nullptr;
			submitted_job_cv = other.submitted_job_cv;
			other.submitted_job_cv = nullptr;
			thread = std::move(other.thread);
			doing_work = other.doing_work;
			running = other.running;
		}
		return *this;
	}
	~Worker() noexcept {
		try {
			if (submitted_job_mutex) {
				delete submitted_job_mutex;
				submitted_job_mutex = nullptr;
			}
			if (submitted_job_cv) {
				delete submitted_job_cv;
				submitted_job_cv = nullptr;
			}
		}
		catch (...) {

		}
	}

	void Submit(WorkRequest req) {
		assert(!doing_work && !submitted_job.has_value());
		doing_work = true;
		std::unique_lock<std::mutex> lock(*submitted_job_mutex);
		submitted_job.emplace(req);
		lock.unlock();
		submitted_job_cv->notify_one();
	}

	std::optional<WorkRequest> GetJob() {
		while (running) {
			std::unique_lock<std::mutex> lock(*submitted_job_mutex);
			using namespace std::chrono_literals;
			if (submitted_job_cv->wait_for(lock, 100ms, [this] { return submitted_job.has_value(); })) {
				WorkRequest req(*submitted_job);
				submitted_job.reset();
				return { req };
			}
		}
		return {};
	}

	void SubmitResponse(WorkResponse resp) {
		std::unique_lock<std::mutex> lock(*finished_works_mutex);
		finished_works->push(resp);
	}

	void Join() {
		thread.join();
	}

	bool doing_work;
	bool running;

	usize worker_id;

	std::queue<WorkResponse>* finished_works;
	std::mutex* finished_works_mutex;

	std::vector<usize>* free_workers;
	std::mutex* free_workers_mutex;

	std::optional<WorkRequest> submitted_job;
	std::mutex *submitted_job_mutex;
	std::condition_variable *submitted_job_cv;

	std::thread thread;
};

};

struct WorkerManager {
	WorkerManager(usize num_workers) :
		running(true),
		num_workers(num_workers)
	{
		for (usize i(0); i != num_workers; ++i) {
			workers.emplace_back(i, std::addressof(finished_works), std::addressof(finished_works_mutex), std::addressof(free_workers), std::addressof(free_workers_mutex));
		}
	}

	WorkerManager(WorkerManager const& a) = delete;
	WorkerManager& operator=(WorkerManager const& a) = delete;
	WorkerManager(WorkerManager&& a) = delete;
	WorkerManager& operator=(WorkerManager&& a) = delete;

	~WorkerManager() noexcept  {
		try {
			if (running) {
				Stop();
				Join();
			}
		}
		catch (...) {

		}
	}

	void SubmitWork(WorkRequest req) {
		usize worker_id(0);
		while (running) {
			std::lock_guard<std::mutex> guard(free_workers_mutex);
			if (free_workers.size() == 0) {
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(10ms);
				continue;
			}
			worker_id = free_workers.back();
			free_workers.pop_back();
			break;
		}
		workers[worker_id].Submit(req);
	}
	std::optional<WorkResponse> PollResponse() {
		std::lock_guard<std::mutex> guard(finished_works_mutex);
		if (finished_works.size()) {
			auto resp(finished_works.front());
			finished_works.pop();
			return { resp };
		}
		else {
			return {};
		}
	}

	void Stop() {
		running = false;
		for (auto& worker : workers)
			worker.running = false;
	}

	void Join() {
		for (auto& worker : workers)
			worker.Join();
	}

	bool running;

	usize num_workers;
	std::vector<worker_details::Worker> workers;

	std::queue<WorkResponse> finished_works;
	std::mutex finished_works_mutex;

	std::vector<usize> free_workers;
	std::mutex free_workers_mutex;
};
