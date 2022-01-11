
#ifndef _TEST_KERNEL_H_
#define _TEST_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

#endif
