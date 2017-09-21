#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	SyncedMemory::SyncedMemory()
		: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
			//获取gpu当前设备id
			CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
		}

	SyncedMemory::SyncedMemory(size_t size)
		: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
			CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
		}

	SyncedMemory::~SyncedMemory() {
		check_device();
		if (cpu_ptr_ && own_cpu_data_) {//cpu指针有数据，并且是自己申请的内存，才释放
			CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
		}

#ifndef CPU_ONLY
		if (gpu_ptr_ && own_gpu_data_) {//gpu指针有数据，并且是自己申请的内存，才释放
			CUDA_CHECK(cudaFree(gpu_ptr_));
		}
#endif  // CPU_ONLY
	}

	inline void SyncedMemory::to_cpu() {
		check_device();
		switch (head_) {
			case UNINITIALIZED://此时cpu和gpu都没有数据，还在初始化状态
				CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);//申请内存
				caffe_memset(size_, 0, cpu_ptr_);//set为0
				head_ = HEAD_AT_CPU;//标志最新数据在cpu
				own_cpu_data_ = true;//标志cpu数据是自己申请的
				break;
			case HEAD_AT_GPU://此时最新数据在gpu
#ifndef CPU_ONLY
				if (cpu_ptr_ == NULL) {//cpu数据为空
					CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);//申请cpu内存
					own_cpu_data_ = true;//标志cpu数据是自己申请的
				}
				caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);//从gpu复制数据到cpu
				head_ = SYNCED;//改b状态，表示两边数据都是最新的
#else
				NO_GPU;
#endif
				break;
			case HEAD_AT_CPU:/最新数据在cpu，跳过
			case SYNCED://两边都是最新数据，跳过
				break;
		}
	}

	inline void SyncedMemory::to_gpu() {//和to_cpu一样的
		check_device();
#ifndef CPU_ONLY
		switch (head_) {
			case UNINITIALIZED:
				CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
				caffe_gpu_memset(size_, 0, gpu_ptr_);
				head_ = HEAD_AT_GPU;
				own_gpu_data_ = true;
				break;
			case HEAD_AT_CPU:
				if (gpu_ptr_ == NULL) {
					CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
					own_gpu_data_ = true;
				}
				caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
				head_ = SYNCED;
				break;
			case HEAD_AT_GPU:
			case SYNCED:
				break;
		}
#else
		NO_GPU;
#endif
	}

	const void* SyncedMemory::cpu_data() {//拿出cpu数据，这个拿出去是改不了数据内容的
		check_device();
		to_cpu();//拿出前，需要确保cpu数据最新
		return (const void*)cpu_ptr_;
	}

	void SyncedMemory::set_cpu_data(void* data) {//从外部设置cpu数据
		check_device();
		CHECK(data);
		if (own_cpu_data_) {//如果目前有cpu数据，需要先释放
			CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
		}
		cpu_ptr_ = data;
		head_ = HEAD_AT_CPU;//最新数据在cpu
		own_cpu_data_ = false;//标志cpu数据不是自己申请的
	}

	const void* SyncedMemory::gpu_data() {//获取gpu数据
		check_device();
#ifndef CPU_ONLY
		to_gpu();//需要确保此时的gpu数据是最新的
		return (const void*)gpu_ptr_;
#else
		NO_GPU;
		return NULL;
#endif
	}

	void SyncedMemory::set_gpu_data(void* data) {//设置gpu数据
		check_device();
#ifndef CPU_ONLY
		CHECK(data);
		if (own_gpu_data_) {//如果当前有gpu数据，需要先释放
			CUDA_CHECK(cudaFree(gpu_ptr_));
		}
		gpu_ptr_ = data;
		head_ = HEAD_AT_GPU;//标志最新数据在gpu
		own_gpu_data_ = false;//标志gpu数据不是自己申请的
#else
		NO_GPU;
#endif
	}

	void* SyncedMemory::mutable_cpu_data() {//这个指针出去是可以改内存的
		check_device();
		to_cpu();
		head_ = HEAD_AT_CPU;//标志最新数据在cpu
		return cpu_ptr_;
	}

	void* SyncedMemory::mutable_gpu_data() {//这个指针出去是可以改内存的
		check_device();
#ifndef CPU_ONLY
		to_gpu();
		head_ = HEAD_AT_GPU;//表示最新数据在gpu
		return gpu_ptr_;
#else
		NO_GPU;
		return NULL;
#endif
	}

#ifndef CPU_ONLY
	void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {//最新数据在cpu，然后异步流水到gpu
		check_device();
		CHECK(head_ == HEAD_AT_CPU);
		if (gpu_ptr_ == NULL) {//如果gpu指针为空，申请内存
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
			own_gpu_data_ = true;
		}
		const cudaMemcpyKind put = cudaMemcpyHostToDevice;
		CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));//异步流水
		// Assume caller will synchronize on the stream before use
		head_ = SYNCED;//标志两边数据都是最新的。
	}
#endif

	void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
		int device;
		cudaGetDevice(&device);
		CHECK(device == device_);
		if (gpu_ptr_ && own_gpu_data_) {
			cudaPointerAttributes attributes;
			CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));// 判断是否内存地址是否使用了通用虚拟地址技术
			CHECK(attributes.device == device_);
		}
#endif
#endif
	}

}  // namespace caffe

