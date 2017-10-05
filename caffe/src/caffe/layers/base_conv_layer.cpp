#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	//初始化各种训练参数
	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			// Configure the kernel size, padding, stride, and inputs.
			ConvolutionParameter conv_param = this->layer_param_.convolution_param();//卷积的参数配置
			force_nd_im2col_ = conv_param.force_nd_im2col();//是否强制使用n维通用卷积，默认为false
			channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());//返回正确的维度信息，默认是1
			const int first_spatial_axis = channel_axis_ + 1;//真正空间的第一维度，默认是2
			const int num_axes = bottom[0]->num_axes();//输入的维度数，一般是4
			num_spatial_axes_ = num_axes - first_spatial_axis;//去除channel后的维度信息，默认是2
			CHECK_GE(num_spatial_axes_, 0);//这个维度必须大于0
			vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));//卷积核的空间信息
			//设置卷积核大小
			// Setup filter kernel dimensions (kernel_shape_).
			kernel_shape_.Reshape(spatial_dim_blob_shape);//卷积核信息
			int* kernel_shape_data = kernel_shape_.mutable_cpu_data();//拿出卷积核信息的cpu内存指针
			if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
				CHECK_EQ(num_spatial_axes_, 2)//必须等于2
					<< "kernel_h & kernel_w can only be used for 2D convolution.";
				CHECK_EQ(0, conv_param.kernel_size_size())//这个必须等于0
					<< "Either kernel_size or kernel_h/w should be specified; not both.";
				kernel_shape_data[0] = conv_param.kernel_h();//放入卷积核信息存储中
				kernel_shape_data[1] = conv_param.kernel_w();
			} else {
				const int num_kernel_dims = conv_param.kernel_size_size();//卷积核维度
				CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)//需要相等 or 1也行
					<< "kernel_size must be specified once, or once per spatial dimension "
					<< "(kernel_size specified " << num_kernel_dims << " times; "
					<< num_spatial_axes_ << " spatial dims).";
				for (int i = 0; i < num_spatial_axes_; ++i) {
					kernel_shape_data[i] =
						conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);//放入卷积核的信息存储中
				}
			}
			for (int i = 0; i < num_spatial_axes_; ++i) {
				CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";//每个卷积核的维度信息都必须大于0
			}
			//设置卷积核的步长，和上面的设置差不多
			// Setup stride dimensions (stride_).
			stride_.Reshape(spatial_dim_blob_shape);
			int* stride_data = stride_.mutable_cpu_data();
			if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
				CHECK_EQ(num_spatial_axes_, 2)//必须等于2
					<< "stride_h & stride_w can only be used for 2D convolution.";
				CHECK_EQ(0, conv_param.stride_size())//必须等于0
					<< "Either stride or stride_h/w should be specified; not both.";
				stride_data[0] = conv_param.stride_h();
				stride_data[1] = conv_param.stride_w();
			} else {
				const int num_stride_dims = conv_param.stride_size();
				CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
						num_stride_dims == num_spatial_axes_)//必须相等 or 0默认 or 1
					<< "stride must be specified once, or once per spatial dimension "
					<< "(stride specified " << num_stride_dims << " times; "
					<< num_spatial_axes_ << " spatial dims).";
				const int kDefaultStride = 1;
				for (int i = 0; i < num_spatial_axes_; ++i) {
					stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
						conv_param.stride((num_stride_dims == 1) ? 0 : i);
					CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
				}
			}
			//设置填充信息，和上面的设置差不多
			// Setup pad dimensions (pad_).
			pad_.Reshape(spatial_dim_blob_shape);
			int* pad_data = pad_.mutable_cpu_data();
			if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
				CHECK_EQ(num_spatial_axes_, 2)
					<< "pad_h & pad_w can only be used for 2D convolution.";
				CHECK_EQ(0, conv_param.pad_size())
					<< "Either pad or pad_h/w should be specified; not both.";
				pad_data[0] = conv_param.pad_h();
				pad_data[1] = conv_param.pad_w();
			} else {
				const int num_pad_dims = conv_param.pad_size();
				CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
						num_pad_dims == num_spatial_axes_)//相等 or 0 默认 or 1
					<< "pad must be specified once, or once per spatial dimension "
					<< "(pad specified " << num_pad_dims << " times; "
					<< num_spatial_axes_ << " spatial dims).";
				const int kDefaultPad = 0;
				for (int i = 0; i < num_spatial_axes_; ++i) {
					pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
						conv_param.pad((num_pad_dims == 1) ? 0 : i);
				}
			}
			//设置扩展信息
			// Setup dilation dimensions (dilation_).
			dilation_.Reshape(spatial_dim_blob_shape);
			int* dilation_data = dilation_.mutable_cpu_data();
			const int num_dilation_dims = conv_param.dilation_size();
			CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
					num_dilation_dims == num_spatial_axes_)
				<< "dilation must be specified once, or once per spatial dimension "
				<< "(dilation specified " << num_dilation_dims << " times; "
				<< num_spatial_axes_ << " spatial dims).";
			const int kDefaultDilation = 1;
			for (int i = 0; i < num_spatial_axes_; ++i) {
				dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
					conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
			}
			// Special case: im2col is the identity for 1x1 convolution with stride 1
			// and no padding, so flag for skipping the buffer and transformation.
			//判断是否是1×1的卷积核
			is_1x1_ = true;
			for (int i = 0; i < num_spatial_axes_; ++i) {
				is_1x1_ &=
					kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;//卷积核为1×1 and 步长为1 and 添加信息为0
				if (!is_1x1_) { break; }
			}
			// Configure output channels and groups.
			channels_ = bottom[0]->shape(channel_axis_);//返回channel这一维的容量
			num_output_ = this->layer_param_.convolution_param().num_output();//几个输出，就是几个卷积核
			CHECK_GT(num_output_, 0);//卷积核必须大于0
			group_ = this->layer_param_.convolution_param().group();//分组信息
			CHECK_EQ(channels_ % group_, 0);//输入的通道信息必须整除
			CHECK_EQ(num_output_ % group_, 0)//卷积核数必须整除
				<< "Number of output should be multiples of group.";
			//设置输入输出的通道信息
			if (reverse_dimensions()) {
				conv_out_channels_ = channels_;
				conv_in_channels_ = num_output_;
			} else {//一般的卷积是这个
				conv_out_channels_ = num_output_;
				conv_in_channels_ = channels_;
			}
			// Handle the parameters: weights and biases.
			// - blobs_[0] holds the filter weights
			// - blobs_[1] holds the biases (optional)
			//处理权重信息核偏置项
			vector<int> weight_shape(2);
			weight_shape[0] = conv_out_channels_;
			weight_shape[1] = conv_in_channels_ / group_;
			for (int i = 0; i < num_spatial_axes_; ++i) {
				weight_shape.push_back(kernel_shape_data[i]);//放入卷积核维度信息
			}
			bias_term_ = this->layer_param_.convolution_param().bias_term();//偏置项
			vector<int> bias_shape(bias_term_, num_output_);
			//处理训练的参数
			if (this->blobs_.size() > 0) {
				CHECK_EQ(1 + bias_term_, this->blobs_.size())//blobs size需要一样
					<< "Incorrect number of weight blobs.";
				if (weight_shape != this->blobs_[0]->shape()) {//weight 信息需要核blobs 0 相等
					Blob<Dtype> weight_shaped_blob(weight_shape);
					LOG(FATAL) << "Incorrect weight shape: expected shape "
						<< weight_shaped_blob.shape_string() << "; instead, shape was "
						<< this->blobs_[0]->shape_string();
				}
				if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {//偏置项需要核 blobs 1 相等
					Blob<Dtype> bias_shaped_blob(bias_shape);
					LOG(FATAL) << "Incorrect bias shape: expected shape "
						<< bias_shaped_blob.shape_string() << "; instead, shape was "
						<< this->blobs_[1]->shape_string();
				}
				LOG(INFO) << "Skipping parameter initialization";
			} else {
				//初始化
				if (bias_term_) {
					this->blobs_.resize(2);
				} else {
					this->blobs_.resize(1);
				}
				// Initialize and fill the weights:
				// output channels x input channels per-group x kernel height x kernel width
				this->blobs_[0].reset(new Blob<Dtype>(weight_shape));//申请内存
				shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
							this->layer_param_.convolution_param().weight_filler()));
				weight_filler->Fill(this->blobs_[0].get());//初始化，填充
				// If necessary, initialize and fill the biases.
				if (bias_term_) {
					this->blobs_[1].reset(new Blob<Dtype>(bias_shape));//申请内存
					shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
								this->layer_param_.convolution_param().bias_filler()));
					bias_filler->Fill(this->blobs_[1].get());//填充
				}
			}
			kernel_dim_ = this->blobs_[0]->count(1);//一个卷积核的大小信息
			weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;//分组卷积的权重 offset
			// Propagate gradients to the parameters (as directed by backward pass).
			this->param_propagate_down_.resize(this->blobs_.size(), true);//全部设置反向传播
		}

	//处理输出的维度信息，对于输入，上面那层已经处理好了
	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			const int first_spatial_axis = channel_axis_ + 1;//第一个开始的空间信息
			CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
				<< "bottom num_axes may not change.";
			num_ = bottom[0]->count(0, channel_axis_);//就是batch size，批量输入的个数
			CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
				<< "Input size incompatible with convolution kernel.";
			// TODO: generalize to handle inputs of different shapes.
			//卷积的每个输入，需要判断所有的数据都相等
			for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
				CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
					<< "All inputs must have the same shape.";
			}
			// Shape the tops.
			bottom_shape_ = &bottom[0]->shape();//输入数据的维度信息
			compute_output_shape();//计算输出的维度信息
			vector<int> top_shape(bottom[0]->shape().begin(),
					bottom[0]->shape().begin() + channel_axis_);
			top_shape.push_back(num_output_);
			for (int i = 0; i < num_spatial_axes_; ++i) {
				top_shape.push_back(output_shape_[i]);//存储输出的维度信息
			}
			for (int top_id = 0; top_id < top.size(); ++top_id) {
				top[top_id]->Reshape(top_shape);//申请，初始化输出空间
			}
			if (reverse_dimensions()) {
				conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);//输入的空间大小信息
			} else {
				conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);//默认是输出的空间大小信息
			}
			col_offset_ = kernel_dim_ * conv_out_spatial_dim_;//对于col 每个分组的偏移量
			output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;//对于输出数据，每个分组的偏移量
			// Setup input dimensions (conv_input_shape_).
			vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
			conv_input_shape_.Reshape(bottom_dim_blob_shape);
			int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
			for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
				if (reverse_dimensions()) {
					conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
				} else {
					conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
				}
			}
			// The im2col result buffer will only hold one image at a time to avoid
			// overly large memory usage. In the special case of 1x1 convolution
			// it goes lazily unused to save memory.
			//初始化col_buffer，为了im2col用的
			col_buffer_shape_.clear();
			col_buffer_shape_.push_back(kernel_dim_ * group_);
			for (int i = 0; i < num_spatial_axes_; ++i) {
				if (reverse_dimensions()) {
					col_buffer_shape_.push_back(input_shape(i + 1));
				} else {
					col_buffer_shape_.push_back(output_shape_[i]);
				}
			}
			col_buffer_.Reshape(col_buffer_shape_);//申请col_buffer_的内存
			bottom_dim_ = bottom[0]->count(channel_axis_);//一张图片输入的大小
			top_dim_ = top[0]->count(channel_axis_);//一张图片输出的大小
			num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
			num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
			// Set up the all ones "bias multiplier" for adding biases by BLAS
			out_spatial_dim_ = top[0]->count(first_spatial_axis);//一张输出图片的空间信息，没有通道的
			if (bias_term_) {
				//初始化一个全1的偏置项
				vector<int> bias_multiplier_shape(1, out_spatial_dim_);
				bias_multiplier_.Reshape(bias_multiplier_shape);
				caffe_set(bias_multiplier_.count(), Dtype(1),
						bias_multiplier_.mutable_cpu_data());
			}
		}

	//卷积这里是，主要是转换了矩阵，然后相乘
	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
				const Dtype* weights, Dtype* output, bool skip_im2col) {
			const Dtype* col_buff = input;
			if (!is_1x1_) {
				if (!skip_im2col) {
					conv_im2col_cpu(input, ol_buffer_.mutable_cpu_data());
				}
				col_buff = col_buffer_.cpu_data();
			}
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
						group_, conv_out_spatial_dim_, kernel_dim_,
						(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
						(Dtype)0., output + output_offset_ * g);
			}
		}

	//卷积的偏置项就是简单的相加而已
	//给每个矩阵都加上一个数
	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
				const Dtype* bias) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
					out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
					(Dtype)1., output);
		}

	//梯度的传递，就是对x求导 等于 input = weights * output 
	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
				const Dtype* weights, Dtype* input) {
			Dtype* col_buff = col_buffer_.mutable_cpu_data();
			if (is_1x1_) {
				col_buff = input;
			}
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
						conv_out_spatial_dim_, conv_out_channels_ / group_,
						(Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
						(Dtype)0., col_buff + col_offset_ * g);
			}
			if (!is_1x1_) {
				conv_col2im_cpu(col_buff, input);
			}
		}

	//对权重求导，等于 weights = output * input + weights, 梯度累积
	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
				const Dtype* output, Dtype* weights) {
			const Dtype* col_buff = input;
			if (!is_1x1_) {
				conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
				col_buff = col_buffer_.cpu_data();
			}
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
						kernel_dim_, conv_out_spatial_dim_,
						(Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
						(Dtype)1., weights + weight_offset_ * g);
			}
		}
	//对偏置项的求导，bias = input * (全1矩阵) + bias，梯度累积 
	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
				const Dtype* input) {
			caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
					input, bias_multiplier_.cpu_data(), 1., bias);
		}

#ifndef CPU_ONLY

	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
				const Dtype* weights, Dtype* output, bool skip_im2col) {
			const Dtype* col_buff = input;
			if (!is_1x1_) {
				if (!skip_im2col) {
					conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
				}
				col_buff = col_buffer_.gpu_data();
			}
			for (int g = 0; g < group_; ++g) {
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
						group_, conv_out_spatial_dim_, kernel_dim_,
						(Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
						(Dtype)0., output + output_offset_ * g);
			}
		}

	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
				const Dtype* bias) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
					out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
					(Dtype)1., output);
		}

	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
				const Dtype* weights, Dtype* input) {
			Dtype* col_buff = col_buffer_.mutable_gpu_data();
			if (is_1x1_) {
				col_buff = input;
			}
			for (int g = 0; g < group_; ++g) {
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
						conv_out_spatial_dim_, conv_out_channels_ / group_,
						(Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
						(Dtype)0., col_buff + col_offset_ * g);
			}
			if (!is_1x1_) {
				conv_col2im_gpu(col_buff, input);
			}
		}

	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
				const Dtype* output, Dtype* weights) {
			const Dtype* col_buff = input;
			if (!is_1x1_) {
				conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
				col_buff = col_buffer_.gpu_data();
			}
			for (int g = 0; g < group_; ++g) {
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
						kernel_dim_, conv_out_spatial_dim_,
						(Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
						(Dtype)1., weights + weight_offset_ * g);
			}
		}

	template <typename Dtype>
		void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
				const Dtype* input) {
			caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
					input, bias_multiplier_.gpu_data(), 1., bias);
		}

#endif  // !CPU_ONLY

	INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
