#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;//最大的维度，维度数不能超过这个

namespace caffe {

	/**
	 * @brief A wrapper around SyncedMemory holders serving as the basic
	 *        computational unit through which Layer%s, Net%s, and Solver%s
	 *        interact.
	 *
	 * TODO(dox): more thorough description.
	 */
	template <typename Dtype>
				  class Blob {
					  public:
						  Blob()
							  : data_(), diff_(), count_(0), capacity_(0) {}//初始化一个空的blob

						  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
						  explicit Blob(const int num, const int channels, const int height,
								  const int width);//初始化四个维度的blob，这个是最常用的
						  explicit Blob(const vector<int>& shape);//用vector来初始化blob，可以N个维度

						  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
						  void Reshape(const int num, const int channels, const int height,
								  const int width);//改变维度信息
						  /**
						   * @brief Change the dimensions of the blob, allocating new memory if
						   *        necessary.
						   *
						   * This function can be called both to create an initial allocation
						   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
						   * or Layer::Forward. When changing the size of blob, memory will only be
						   * reallocated if sufficient memory does not already exist, and excess memory
						   * will never be freed.
						   *
						   * Note that reshaping an input blob and immediately calling Net::Backward is
						   * an error; either Net::Forward or Net::Reshape need to be called to
						   * propagate the new input shape to higher layers.
						   */
						  void Reshape(const vector<int>& shape);//改变维度信息
						  void Reshape(const BlobShape& shape);//改变维度信息
						  void ReshapeLike(const Blob& other);//改变维度信息
						  inline string shape_string() const {//返回维度信息
							  ostringstream stream;
							  for (int i = 0; i < shape_.size(); ++i) {
								  stream << shape_[i] << " ";
							  }
							  stream << "(" << count_ << ")";
							  return stream.str();
						  }
						  inline const vector<int>& shape() const { return shape_; }//返回维度信息
						  /**
						   * @brief Returns the dimension of the index-th axis (or the negative index-th
						   *        axis from the end, if index is negative).
						   *
						   * @param index the axis index, which may be negative as it will be
						   *        "canonicalized" using CanonicalAxisIndex.
						   *        Dies on out of range index.
						   */
						  inline int shape(int index) const {//返回某一维的容量
							  return shape_[CanonicalAxisIndex(index)];
						  }
						  inline int num_axes() const { return shape_.size(); }//返回维度数
						  inline int count() const { return count_; }//返回目前存储的数量

						  /**
						   * @brief Compute the volume of a slice; i.e., the product of dimensions
						   *        among a range of axes.
						   *
						   * @param start_axis The first axis to include in the slice.
						   *
						   * @param end_axis The first axis to exclude from the slice.
						   */
						  inline int count(int start_axis, int end_axis) const {//计算从start_axis到end_axis，总的容量
							  CHECK_LE(start_axis, end_axis);
							  CHECK_GE(start_axis, 0);
							  CHECK_GE(end_axis, 0);
							  CHECK_LE(start_axis, num_axes());
							  CHECK_LE(end_axis, num_axes());
							  int count = 1;
							  for (int i = start_axis; i < end_axis; ++i) {
								  count *= shape(i);
							  }
							  return count;
						  }
						  /**
						   * @brief Compute the volume of a slice spanning from a particular first
						   *        axis to the final axis.
						   *
						   * @param start_axis The first axis to include in the slice.
						   */
						  inline int count(int start_axis) const {//计算从start_axis到结尾，总的容量
							  return count(start_axis, num_axes());
						  }

						  /**
						   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
						   *        allowing for negative indexing (e.g., -1 for the last axis).
						   *
						   * @param axis_index the axis index.
						   *        If 0 <= index < num_axes(), return index.
						   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
						   *        e.g., the last axis index (num_axes() - 1) if index == -1,
						   *        the second to last if index == -2, etc.
						   *        Dies on out of range index.
						   */
						  inline int CanonicalAxisIndex(int axis_index) const {//支持-1，这种维度
							  CHECK_GE(axis_index, -num_axes())
								  << "axis " << axis_index << " out of range for " << num_axes()
								  << "-D Blob with shape " << shape_string();
							  CHECK_LT(axis_index, num_axes())
								  << "axis " << axis_index << " out of range for " << num_axes()
								  << "-D Blob with shape " << shape_string();
							  if (axis_index < 0) {
								  return axis_index + num_axes();
							  }
							  return axis_index;
						  }

						  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
						  inline int num() const { return LegacyShape(0); }//返回第一维度的容量
						  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
						  inline int channels() const { return LegacyShape(1); }//返回第二维度的容量
						  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
						  inline int height() const { return LegacyShape(2); }//返回第三维度的容量
						  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
						  inline int width() const { return LegacyShape(3); }//返回第四维度的容量
						  inline int LegacyShape(int index) const {//返回某一维度的容量
							  CHECK_LE(num_axes(), 4)
								  << "Cannot use legacy accessors on Blobs with > 4 axes.";
							  CHECK_LT(index, 4);
							  CHECK_GE(index, -4);
							  if (index >= num_axes() || index < -num_axes()) {
								  // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
								  // indexing) -- this special case simulates the one-padding used to fill
								  // extraneous axes of legacy blobs.
								  return 1;
							  }
							  return shape(index);
						  }

						  inline int offset(const int n, const int c = 0, const int h = 0,
								  const int w = 0) const {//计算数据的偏移量
							  CHECK_GE(n, 0);
							  CHECK_LE(n, num());
							  CHECK_GE(channels(), 0);
							  CHECK_LE(c, channels());
							  CHECK_GE(height(), 0);
							  CHECK_LE(h, height());
							  CHECK_GE(width(), 0);
							  CHECK_LE(w, width());
							  return ((n * channels() + c) * height() + h) * width() + w;
						  }

						  inline int offset(const vector<int>& indices) const {//计算数组的偏移量
							  CHECK_LE(indices.size(), num_axes());
							  int offset = 0;
							  for (int i = 0; i < num_axes(); ++i) {
								  offset *= shape(i);
								  if (indices.size() > i) {
									  CHECK_GE(indices[i], 0);
									  CHECK_LT(indices[i], shape(i));
									  offset += indices[i];
								  }
							  }
							  return offset;
						  }
						  /**
						   * @brief Copy from a source Blob.
						   *
						   * @param source the Blob to copy from
						   * @param copy_diff if false, copy the data; if true, copy the diff
						   * @param reshape if false, require this Blob to be pre-shaped to the shape
						   *        of other (and die otherwise); if true, Reshape this Blob to other's
						   *        shape if necessary
						   */
						  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
								  bool reshape = false);//从别的blob copy 过来 

						  inline Dtype data_at(const int n, const int c, const int h,
								  const int w) const {//返回在数据矩阵某个位置的数据
							  return cpu_data()[offset(n, c, h, w)];
						  }

						  inline Dtype diff_at(const int n, const int c, const int h,
								  const int w) const {//返回在差异矩阵某个位置的数据
							  return cpu_diff()[offset(n, c, h, w)];
						  }

						  inline Dtype data_at(const vector<int>& index) const {//返回在数据矩阵某个位置的数据
							  return cpu_data()[offset(index)];
						  }

						  inline Dtype diff_at(const vector<int>& index) const {//返回在差异矩阵某个位置的数据
							  return cpu_diff()[offset(index)];
						  }

						  inline const shared_ptr<SyncedMemory>& data() const {//返回数据矩阵指针，不能修改
							  CHECK(data_);
							  return data_;
						  }

						  inline const shared_ptr<SyncedMemory>& diff() const {//返回差异矩阵的指针，不能修改
							  CHECK(diff_);
							  return diff_;
						  }

						  const Dtype* cpu_data() const;//返回cpu的数据矩阵
						  void set_cpu_data(Dtype* data);//设置cpu的数据矩阵
						  const int* gpu_shape() const;//返回gpu侧的维度信息
						  const Dtype* gpu_data() const;//返回gpu数据矩阵
						  void set_gpu_data(Dtype* data);//设置gpu数据矩阵
						  const Dtype* cpu_diff() const;//返回cpu差异矩阵
						  const Dtype* gpu_diff() const;//返回gpu差异矩阵
						  Dtype* mutable_cpu_data();//返回cpu数据指针，可以修改
						  Dtype* mutable_gpu_data();//返回gpu数据指针，可以修改
						  Dtype* mutable_cpu_diff();//返回cpu差异矩阵指针，可以修改
						  Dtype* mutable_gpu_diff();//返回ghpu差异矩阵指针，可以修改
						  void Update();//更新数据矩阵，data - diff
						  void FromProto(const BlobProto& proto, bool reshape = true);//从pb导入
						  void ToProto(BlobProto* proto, bool write_diff = false) const;//导出到pb

						  /// @brief Compute the sum of absolute values (L1 norm) of the data.
						  Dtype asum_data() const;//计算数据的L1范数，绝对值求和
						  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
						  Dtype asum_diff() const;//计算差异矩阵的L1范数
						  /// @brief Compute the sum of squares (L2 norm squared) of the data.
						  Dtype sumsq_data() const;//计算数据的L2范数，欧式距离
						  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
						  Dtype sumsq_diff() const;//计算差异矩阵的L2范数

						  /// @brief Scale the blob data by a constant factor.
						  void scale_data(Dtype scale_factor);//对数据矩阵乘以一个数
						  /// @brief Scale the blob diff by a constant factor.
						  void scale_diff(Dtype scale_factor);//对差异矩阵乘以一个数

						  /**
						   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
						   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
						   *        in their Forward pass.
						   *
						   * This deallocates the SyncedMemory holding this Blob's data_, as
						   * shared_ptr calls its destructor when reset with the "=" operator.
						   */
						  void ShareData(const Blob& other);//从其他blob复制data矩阵
						  /**
						   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
						   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
						   *        in their Forward pass.
						   *
						   * This deallocates the SyncedMemory holding this Blob's diff_, as
						   * shared_ptr calls its destructor when reset with the "=" operator.
						   */
						  void ShareDiff(const Blob& other);//从其他blob复制diff矩阵

						  bool ShapeEquals(const BlobProto& other);//判断两个blob的容量是否一样

					  protected:
						  shared_ptr<SyncedMemory> data_;//存储的数据
						  shared_ptr<SyncedMemory> diff_;//存储差异数据
						  shared_ptr<SyncedMemory> shape_data_;//存储维度信息
						  vector<int> shape_;//存储维度信息
						  int count_;//这个blob目前的数据量
						  int capacity_;//这个blob能存储的最大数据量

						  DISABLE_COPY_AND_ASSIGN(Blob);
				  };  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
