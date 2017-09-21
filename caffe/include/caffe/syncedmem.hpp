#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
#include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

	// If CUDA is available and in GPU mode, host memory will be allocated pinned,
	// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
	// The improvement in performance seems negligible in the single GPU case,
	// but might be more significant for parallel training. Most importantly,
	// it improved stability for large models on many GPUs.

	//封装好的申请内存函数
	inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
		//gpu模式
		if (Caffe::mode() == Caffe::GPU) {
			CUDA_CHECK(cudaMallocHost(ptr, size));
			*use_cuda = true;
			return;
		}
#endif
		//cpu模式下，可以用mkl
#ifdef USE_MKL
		*ptr = mkl_malloc(size ? size:1, 64);
#else
		*ptr = malloc(size);
#endif
		*use_cuda = false;
		CHECK(*ptr) << "host allocation of size " << size << " failed";
	}

	//释放内存
	inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
		//gpu模式下
		if (use_cuda) {
			CUDA_CHECK(cudaFreeHost(ptr));
			return;
		}
#endif
		//cpu模式下
#ifdef USE_MKL
		mkl_free(ptr);
#else
		free(ptr);
#endif
	}


	/**
	 * @brief Manages memory allocation and synchronization between the host (CPU)
	 *        and device (GPU).
	 *
	 * TODO(dox): more thorough description.
	 */
	//一个存储数据的内存工具，支持在gpu和cpu中互传，保证两边数据一致
	class SyncedMemory {
		public:
			SyncedMemory();
			explicit SyncedMemory(size_t size);//用内存大小，初始化一个内存工具
			~SyncedMemory();
			const void* cpu_data();//保证cpu数据最新的情况下，拿出cpu数据，不能改的
			void set_cpu_data(void* data);//从外部设置cpu数据，然后标志cpu数据是最新的
			const void* gpu_data();//保证gpu数据最新的情况下，拿出gpu数据，不能改的
			void set_gpu_data(void* data);//从外部设置gpu数据，然后gpu数据是最新的
			void* mutable_cpu_data();//保证cpu数据是最新的，拿出cpu数据的指针，可以修改，标志最新数据是cpu
			void* mutable_gpu_data();//保证gpu数据是最新的，拿出gpu数据的指针，可以修改，标志最新数据是gpu
			enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };//其中的五种状态
			SyncedHead head() { return head_; }//返回当前状态
			size_t size() { return size_; }//返回内存的大小

#ifndef CPU_ONLY
			void async_gpu_push(const cudaStream_t& stream);//从cpu异步流水到gpu
#endif

		private:
			void check_device();

			void to_cpu();//保证cpu数据是最新的
			void to_gpu();//保证gpu数据是最新
			void* cpu_ptr_;//存储在cpu的指针
			void* gpu_ptr_;//存储在gpu的指针
			size_t size_;//内存大小
			SyncedHead head_;//内存的状态
			bool own_cpu_data_;//cpu里面的内存是否是自己申请的
			bool cpu_malloc_use_cuda_;//申请内存是否用了cuda，就是gpu模式
			bool own_gpu_data_;//GPU里面的内存是否是自己申请的
			int device_;//设备号

			DISABLE_COPY_AND_ASSIGN(SyncedMemory);
	};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
