#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

	template <typename Dtype>
		void ConvolutionLayer<Dtype>::compute_output_shape() {//计算输出数据的大小
			const int* kernel_shape_data = this->kernel_shape_.cpu_data();
			const int* stride_data = this->stride_.cpu_data();//步长
			const int* pad_data = this->pad_.cpu_data();//添加信息
			const int* dilation_data = this->dilation_.cpu_data();
			this->output_shape_.clear();
			for (int i = 0; i < this->num_spatial_axes_; ++i) {
				// i + 1 to skip channel axis
				const int input_dim = this->input_shape(i + 1);
				const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;//真实卷积核的大小，经过膨胀之后
				const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)//输出的大小
					/ stride_data[i] + 1;
				this->output_shape_.push_back(output_dim);
			}
		}

	//前向传播，转换成矩阵相乘，有偏置项的话，加上偏置项就行了
	template <typename Dtype>
		void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {//前向传播
			const Dtype* weight = this->blobs_[0]->cpu_data();//卷积核
			for (int i = 0; i < bottom.size(); ++i) {//遍历每个输入数据
				const Dtype* bottom_data = bottom[i]->cpu_data();//拿到输入数据
				Dtype* top_data = top[i]->mutable_cpu_data();//拿到输出数据的指针
				for (int n = 0; n < this->num_; ++n) {//batch 中每个都单独处理
					this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
							top_data + n * this->top_dim_);
					if (this->bias_term_) {
						const Dtype* bias = this->blobs_[1]->cpu_data();
						this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
					}
				}
			}
		}

	//其实它的求导完全和全连接层一样
	template <typename Dtype>
		void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {//反向传播
			const Dtype* weight = this->blobs_[0]->cpu_data();
			Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
			for (int i = 0; i < top.size(); ++i) {//遍历每一个输出数据
				const Dtype* top_diff = top[i]->cpu_diff();
				const Dtype* bottom_data = bottom[i]->cpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
				// Bias gradient, if necessary.
				//对偏置项求导 bias_diff = top_diff * (全1) + bias_diff
				if (this->bias_term_ && this->param_propagate_down_[1]) {
					Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
					for (int n = 0; n < this->num_; ++n) {//遍历batch size的每一个
						this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
					}
				}
				if (this->param_propagate_down_[0] || propagate_down[i]) {
					for (int n = 0; n < this->num_; ++n) {//遍历batch size的每一个
						// gradient w.r.t. weight. Note that we will accumulate diffs.
						//对w求导weight_diff = bottom * top_diff + weight, 梯度累积
						if (this->param_propagate_down_[0]) {
							this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
									top_diff + n * this->top_dim_, weight_diff);
						}
						// gradient w.r.t. bottom data, if necessary.
						//处理梯度的传递, bottom_diff = weight * top_diff
						if (propagate_down[i]) {
							this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
									bottom_diff + n * this->bottom_dim_);
						}
					}
				}
			}
		}

#ifdef CPU_ONLY
	STUB_GPU(ConvolutionLayer);
#endif

	INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
