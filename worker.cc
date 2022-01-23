
#include "worker.h"
#include "wrappers/CUDADeviceMemory.h"

namespace worker_details {

void worker_thread(Worker* self) {

	CUDADeviceMemoryUnique<f32> tmp_f32_1;
	CUDADeviceMemoryUnique<u8> tmp_u8_1;

	while (self->running) {
		auto task_opt(self->GetJob());
		if (!task_opt.has_value())
			continue; // self->running == false
		FrameBatch fb(task_opt->frames);

	}
}

};

void WorkResponse::ConsumeResponse() {
	std::lock_guard<std::mutex> guard(*associated_worker->free_workers_mutex);
	associated_worker->free_workers->push_back(associated_worker->worker_id);
	associated_worker->doing_work = false;
}
