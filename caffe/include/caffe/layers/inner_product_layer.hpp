#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	 * @brief Also known as a "fully-connected" layer, computes an inner product
	 *        with a set of learned weights, and (optionally) adds biases.
	 *
	 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
	 */
	template <typename Dtype>
		      class InnerProductLayer : public Layer<Dtype> {
			      public:
				      explicit InnerProductLayer(const LayerParameter& param)//初始化全连接层
					      : Layer<Dtype>(param) {}
				      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
						      const vector<Blob<Dtype>*>& top);//设置全连接层的训练参数
				      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
						      const vector<Blob<Dtype>*>& top);//设置全连接层的输出参数

				      virtual inline const char* type() const { return "InnerProduct"; }//返回该层的类型
				      virtual inline int ExactNumBottomBlobs() const { return 1; }//输入只有一个
				      virtual inline int ExactNumTopBlobs() const { return 1; }//输出只有一个

			      protected:
				      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
						      const vector<Blob<Dtype>*>& top);//cpu前向传播
				      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
						      const vector<Blob<Dtype>*>& top);//gpu前向传播
				      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
						      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);//cpu反向传播
				      virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
						      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);//gpu反向传播

				      int M_;//batch size，一次传播多少个
				      int K_;//输入特征个数
				      int N_;//输出特征个数
				      bool bias_term_;//偏置项
				      Blob<Dtype> bias_multiplier_;//一个全1的矩阵
				      bool transpose_;  ///< if true, assume transposed weights//是否需要转置，true不用转置
		      };

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
