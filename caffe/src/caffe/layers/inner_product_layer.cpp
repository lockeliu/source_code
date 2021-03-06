#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
		void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {//设置训练的参数
			const int num_output = this->layer_param_.inner_product_param().num_output();//输出的参数个数
			bias_term_ = this->layer_param_.inner_product_param().bias_term();//标志是否要加bias这一项
			transpose_ = this->layer_param_.inner_product_param().transpose();//标志是否要转置
			N_ = num_output;//输出的参数个数，因为是一维的，所以就说个数了
			const int axis = bottom[0]->CanonicalAxisIndex(
					this->layer_param_.inner_product_param().axis());//主要是处理负数的情况
			// Dimensions starting from "axis" are "flattened" into a single
			// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
			// and axis == 1, N inner products with dimension CHW are performed.
			K_ = bottom[0]->count(axis);//输入的维度个数
			// Check if we need to set up the weights
			if (this->blobs_.size() > 0) {//如果本身有就直接跳过
				LOG(INFO) << "Skipping parameter initialization";
			} else {
				if (bias_term_) {
					this->blobs_.resize(2);//如果有偏置项，blobs size 为2
				} else {
					this->blobs_.resize(1);//如果没有偏置项，blobs size 1
				}
				// Initialize the weights
				vector<int> weight_shape(2);
				if (transpose_) {//这种是不需要转置
					weight_shape[0] = K_;
					weight_shape[1] = N_;
				} else {
					weight_shape[0] = N_;
					weight_shape[1] = K_;
				}
				this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
				// fill the weights
				shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
							this->layer_param_.inner_product_param().weight_filler()));
				weight_filler->Fill(this->blobs_[0].get());//设置w的权重，各种随机的参数初始化方式
				// If necessary, intiialize and fill the bias term
				if (bias_term_) {
					vector<int> bias_shape(1, N_);
					this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
					shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
								this->layer_param_.inner_product_param().bias_filler()));
					bias_filler->Fill(this->blobs_[1].get());//设置b的权重，各种随机的参数初始化方式
				}
			}  // parameter initialization
			this->param_propagate_down_.resize(this->blobs_.size(), true);//是否反向传播resize
		}

	template <typename Dtype>
		void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {//初始化输出的参数
			// Figure out the dimensions
			const int axis = bottom[0]->CanonicalAxisIndex(
					this->layer_param_.inner_product_param().axis());
			const int new_K = bottom[0]->count(axis);//输出为K
			CHECK_EQ(K_, new_K)
				<< "Input size incompatible with inner product parameters.";
			// The first "axis" dimensions are independent inner products; the total
			// number of these is M_, the product over these dimensions.
			M_ = bottom[0]->count(0, axis);//有M_张图片
			// The top shape will be the bottom shape with the flattened axes dropped,
			// and replaced by a single axis with dimension num_output (N_).
			vector<int> top_shape = bottom[0]->shape();
			top_shape.resize(axis + 1);
			top_shape[axis] = N_;
			top[0]->Reshape(top_shape);
			// Set up the bias multiplier；
			if (bias_term_) {
				vector<int> bias_shape(1, M_);
				bias_multiplier_.Reshape(bias_shape);
				caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
			}
		}

	template <typename Dtype>
		void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {//前向传播
			const Dtype* bottom_data = bottom[0]->cpu_data();//输入矩阵
			Dtype* top_data = top[0]->mutable_cpu_data();//输出矩阵
			const Dtype* weight = this->blobs_[0]->cpu_data();//权重
			caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
					M_, N_, K_,
					(Dtype)1.,bottom_data, weight,
					(Dtype)0., top_data);//top_data = bottom_data * weight
			if (bias_term_) {//其实这个bias_multiplier是 1 * M_，在cblas_sgemm中 1 * M_ 和 M_ * 1 没什么区别
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
						(Dtype)1., bias_multiplier_.cpu_data(),this->blobs_[1]->cpu_data(), 
						(Dtype)1., top_data);//输出矩阵 = 输出矩阵 + 偏置项
			}
		}

	template <typename Dtype>
		void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down,
				const vector<Blob<Dtype>*>& bottom) {//反向传播，y = wx + b
			if (this->param_propagate_down_[0]) {
				const Dtype* top_diff = top[0]->cpu_diff();//上一层的梯度
				const Dtype* bottom_data = bottom[0]->cpu_data();//输出矩阵
				// Gradient with respect to weight
				// 求Weight
				// w_diff = x * y_diff（求导）
				// 实际上是梯度累积 w_diff = w_diff + x * y_diff
				if (transpose_) {
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
							K_, N_, M_,
							(Dtype)1., bottom_data, top_diff,
							(Dtype)1., this->blobs_[0]->mutable_cpu_diff());
				} else {
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
							N_, K_, M_,
							(Dtype)1., top_diff, bottom_data,
							(Dtype)1., this->blobs_[0]->mutable_cpu_diff());
				}
			}
			//求偏置项
			//b_diff = y_diff（求导）
			//实际上是梯度累积 b_diff = b_diff + y_diff
			if (bias_term_ && this->param_propagate_down_[1]) {
				const Dtype* top_diff = top[0]->cpu_diff();
				// Gradient with respect to bias
				caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
						bias_multiplier_.cpu_data(), (Dtype)1.,
						this->blobs_[1]->mutable_cpu_diff());
			}
			//对输入求导
			//x_diff = y_diff * w
			if (propagate_down[0]) {
				const Dtype* top_diff = top[0]->cpu_diff();
				// Gradient with respect to bottom data
				if (transpose_) {
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							M_, K_, N_,
							(Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
							(Dtype)0., bottom[0]->mutable_cpu_diff());
				} else {
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
							M_, K_, N_,
							(Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
							(Dtype)0., bottom[0]->mutable_cpu_diff());
				}
			}
		}

#ifdef CPU_ONLY
	STUB_GPU(InnerProductLayer);
#endif

	INSTANTIATE_CLASS(InnerProductLayer);
	REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
