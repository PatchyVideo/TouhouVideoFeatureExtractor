#pragma once

#include <NvInfer.h>
#include <stdexcept>

struct NvInferRuntime
{
	nvinfer1::IRuntime* runtime;
	NvInferRuntime(nvinfer1::ILogger& logger) :runtime(nullptr)
	{
		runtime = nvinfer1::createInferRuntime(logger);
		if (!runtime)
			throw std::runtime_error("Failed to create TensorRT runtime");
	}
		
	~NvInferRuntime() noexcept
	{
		try
		{
			if (runtime) {
				delete runtime;
				runtime = nullptr;
			}
		} catch (...)
		{

		}
		runtime = nullptr;
	}
	NvInferRuntime(NvInferRuntime const& other) = delete;
	NvInferRuntime& operator=(NvInferRuntime const& other) = delete;
		
	NvInferRuntime(NvInferRuntime&& other) noexcept = delete;
	NvInferRuntime& operator=(NvInferRuntime&& other) noexcept = delete;
};

