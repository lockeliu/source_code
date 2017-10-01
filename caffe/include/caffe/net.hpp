#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_


#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
	 *        specified by a NetParameter.
	 *
	 * TODO(dox): more thorough description.
	 */
	template <typename Dtype>
				  class Net {
					  public:
						  explicit Net(const NetParameter& param);//用net配置初始化网络结构
						  explicit Net(const string& param_file, Phase phase,
								  const int level = 0, const vector<string>* stages = NULL);//用配置文件初始化网络结构
						  virtual ~Net() {}//虚构函数

						  /// @brief Initialize a network with a NetParameter.
						  void Init(const NetParameter& param);//初始化一个网络结构

						  /**
						   * @brief Run Forward and return the result.
						   *
						   */
						  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);//前向传播
						  /// @brief DEPRECATED; use Forward() instead.
						  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
							  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
								  << "will be removed in a future version. Use Forward().";
							  return Forward(loss);
						  }

						  /**
						   * The From and To variants of Forward and Backward operate on the
						   * (topological) ordering by which the net is specified. For general DAG
						   * networks, note that (1) computing from one layer to another might entail
						   * extra computation on unrelated branches, and (2) computation starting in
						   * the middle may be incorrect if all of the layers of a fan-in are not
						   * included.
						   */
						  Dtype ForwardFromTo(int start, int end);//从start开始到end结束，前向传播
						  Dtype ForwardFrom(int start);//从start开始，前向传播
						  Dtype ForwardTo(int end);//前向传播到end结束
						  /// @brief DEPRECATED; set input blobs then use Forward() instead.
						  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
								  Dtype* loss = NULL);//前向传播

						  /**
						   * @brief Zeroes out the diffs of all net parameters.
						   *        Should be run before Backward.
						   */
						  void ClearParamDiffs();//清空所有差异矩阵

						  /**
						   * The network backward should take no input and output, since it solely
						   * computes the gradient w.r.t the parameters, and the data has already been
						   * provided during the forward pass.
						   */
						  void Backward();//反向传播
						  void BackwardFromTo(int start, int end);//从start到end，反向传播
						  void BackwardFrom(int start);//从start开始反向传播
						  void BackwardTo(int end);//反向传播到end

						  /**
						   * @brief Reshape all layers from bottom to top.
						   *
						   * This is useful to propagate changes to layer sizes without running
						   * a forward pass, e.g. to compute output feature size.
						   */
						  void Reshape();//对所有的layer进行reshape操作

						  Dtype ForwardBackward() {//进行一次前向传播，接着一次反向传播
							  Dtype loss;
							  Forward(&loss);//前向传播
							  Backward();//反向传播
							  return loss;
						  }

						  /// @brief Updates the network weights based on the diff values computed.
						  void Update();//把差异矩阵更新到数据矩阵
						  /**
						   * @brief Shares weight data of owner blobs with shared blobs.
						   *
						   * Note: this is called by Net::Init, and thus should normally not be
						   * called manually.
						   */
						  void ShareWeights();//从配置中加载权重

						  /**
						   * @brief For an already initialized net, implicitly copies (i.e., using no
						   *        additional memory) the pre-trained layers from another Net.
						   */
						  void ShareTrainedLayersWith(const Net* other);//从某个网络结构copy权重参数
						  // For an already initialized net, CopyTrainedLayersFrom() copies the already
						  // trained layers from another net parameter instance.
						  /**
						   * @brief For an already initialized net, copies the pre-trained layers from
						   *        another Net.
						   */
						  void CopyTrainedLayersFrom(const NetParameter& param);
						  void CopyTrainedLayersFrom(const string trained_filename);
						  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
						  void CopyTrainedLayersFromHDF5(const string trained_filename);
						  /// @brief Writes the net to a proto.
						  void ToProto(NetParameter* param, bool write_diff = false) const;//序列化成pb
						  /// @brief Writes the net to an HDF5 file.
						  void ToHDF5(const string& filename, bool write_diff = false) const;

						  /// @brief returns the network name.
						  inline const string& name() const { return name_; }//网络名字
						  /// @brief returns the layer names
						  inline const vector<string>& layer_names() const { return layer_names_; }//所有层的名字
						  /// @brief returns the blob names
						  inline const vector<string>& blob_names() const { return blob_names_; }//所有中间输入输出的名字
						  /// @brief returns the blobs
						  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {//所有的中间输入输出
							  return blobs_;
						  }
						  /// @brief returns the layers
						  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {//所有层layer
							  return layers_;
						  }
						  /// @brief returns the phase: TRAIN or TEST
						  inline Phase phase() const { return phase_; }//网络的模式
						  /**
						   * @brief returns the bottom vecs for each layer -- usually you won't
						   *        need this unless you do per-layer checks such as gradients.
						   */
						  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {//所有layers的输入
							  return bottom_vecs_;
						  }
						  /**
						   * @brief returns the top vecs for each layer -- usually you won't
						   *        need this unless you do per-layer checks such as gradients.
						   */
						  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {//所有layers的输出
							  return top_vecs_;
						  }
						  /// @brief returns the ids of the top blobs of layer i
						  inline const vector<int> & top_ids(int i) const {//返回某层的输出id
							  CHECK_GE(i, 0) << "Invalid layer id";
							  CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
							  return top_id_vecs_[i];
						  }
						  /// @brief returns the ids of the bottom blobs of layer i
						  inline const vector<int> & bottom_ids(int i) const {//返回某层的输入id
							  CHECK_GE(i, 0) << "Invalid layer id";
							  CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
							  return bottom_id_vecs_[i];
						  }
						  inline const vector<vector<bool> >& bottom_need_backward() const {//某个输入是否反向传播
							  return bottom_need_backward_;
						  }
						  inline const vector<Dtype>& blob_loss_weights() const {
							  return blob_loss_weights_;
						  }
						  inline const vector<bool>& layer_need_backward() const {//某层是否反向传播
							  return layer_need_backward_;
						  }
						  /// @brief returns the parameters
						  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {//所有参数
							  return params_;
						  }
						  inline const vector<Blob<Dtype>*>& learnable_params() const {//所有的学习参数
							  return learnable_params_;
						  }
						  /// @brief returns the learnable parameter learning rate multipliers
						  inline const vector<float>& params_lr() const { return params_lr_; }//学习率
						  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }//是否有学习率
						  /// @brief returns the learnable parameter decay multipliers
						  inline const vector<float>& params_weight_decay() const {//衰变参数
							  return params_weight_decay_;
						  }
						  inline const vector<bool>& has_params_decay() const {//是否有衰变参数
							  return has_params_decay_;
						  }
						  const map<string, int>& param_names_index() const {
							  return param_names_index_;
						  }
						  inline const vector<int>& param_owners() const { return param_owners_; }
						  inline const vector<string>& param_display_names() const {
							  return param_display_names_;
						  }
						  /// @brief Input and output blob numbers
						  inline int num_inputs() const { return net_input_blobs_.size(); }//输入的个数
						  inline int num_outputs() const { return net_output_blobs_.size(); }//输出的个数
						  inline const vector<Blob<Dtype>*>& input_blobs() const {//返回输入数据
							  return net_input_blobs_;
						  }
						  inline const vector<Blob<Dtype>*>& output_blobs() const {//返回输出数据
							  return net_output_blobs_;
						  }
						  inline const vector<int>& input_blob_indices() const {//返回输入数据的id
							  return net_input_blob_indices_;
						  }
						  inline const vector<int>& output_blob_indices() const {//返回输出数据的id1
							  return net_output_blob_indices_;
						  }
						  bool has_blob(const string& blob_name) const;//判断是否有这个blobs
						  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;//用blob名字查找是否有这个blob数据
						  bool has_layer(const string& layer_name) const;//用layers名字判断是否有该层
						  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;//用layer名查找是否有该layer

						  void set_debug_info(const bool value) { debug_info_ = value; }

						  // Helpers for Init.
						  /**
						   * @brief Remove layers that the user specified should be excluded given the current
						   *        phase, level, and stage.
						   */
						  static void FilterNet(const NetParameter& param,
								  NetParameter* param_filtered);//根据规则筛选网络结构
						  /// @brief return whether NetState state meets NetStateRule rule
						  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
								  const string& layer_name);//判断规则是否符合或者不符合

						  //以下都是callback回调函数
						  // Invoked at specific points during an iteration
						  class Callback {
							  protected:
								  virtual void run(int layer) = 0;

								  template <typename T>
									  friend class Net;
						  };
						  const vector<Callback*>& before_forward() const { return before_forward_; }
						  void add_before_forward(Callback* value) {
							  before_forward_.push_back(value);
						  }
						  const vector<Callback*>& after_forward() const { return after_forward_; }
						  void add_after_forward(Callback* value) {
							  after_forward_.push_back(value);
						  }
						  const vector<Callback*>& before_backward() const { return before_backward_; }
						  void add_before_backward(Callback* value) {
							  before_backward_.push_back(value);
						  }
						  const vector<Callback*>& after_backward() const { return after_backward_; }
						  void add_after_backward(Callback* value) {
							  after_backward_.push_back(value);
						  }

					  protected:
						  // Helpers for Init.
						  /// @brief Append a new top blob to the net.
						  void AppendTop(const NetParameter& param, const int layer_id,
								  const int top_id, set<string>* available_blobs,
								  map<string, int>* blob_name_to_idx);
						  /// @brief Append a new bottom blob to the net.
						  int AppendBottom(const NetParameter& param, const int layer_id,
								  const int bottom_id, set<string>* available_blobs,
								  map<string, int>* blob_name_to_idx);
						  /// @brief Append a new parameter blob to the net.
						  void AppendParam(const NetParameter& param, const int layer_id,
								  const int param_id);

						  /// @brief Helper for displaying debug info in Forward.
						  void ForwardDebugInfo(const int layer_id);
						  /// @brief Helper for displaying debug info in Backward.
						  void BackwardDebugInfo(const int layer_id);
						  /// @brief Helper for displaying debug info in Update.
						  void UpdateDebugInfo(const int param_id);

						  /// @brief The network name
						  string name_;//网络的名字
						  /// @brief The phase: TRAIN or TEST
						  Phase phase_;//网络的模式，训练或者测试
						  /// @brief Individual layers in the net
						  vector<shared_ptr<Layer<Dtype> > > layers_;//存储了网络结构的每一层
						  vector<string> layer_names_;//用vector存储了每一层的名字
						  map<string, int> layer_names_index_;//层名字和层id的对应关系
						  vector<bool> layer_need_backward_;//存储每一层是否需要反向传播
						  /// @brief the blobs storing intermediate results between the layer.
						  vector<shared_ptr<Blob<Dtype> > > blobs_;
						  vector<string> blob_names_;
						  map<string, int> blob_names_index_;
						  vector<bool> blob_need_backward_;
						  /// bottom_vecs stores the vectors containing the input for each layer.
						  /// They don't actually host the blobs (blobs_ does), so we simply store
						  /// pointers.
						  vector<vector<Blob<Dtype>*> > bottom_vecs_;//存储每一层的输入数据
						  vector<vector<int> > bottom_id_vecs_;//存储了每一层的输入层id
						  vector<vector<bool> > bottom_need_backward_;//存储了每一层的是否需要反向传播
						  /// top_vecs stores the vectors containing the output for each layer
						  vector<vector<Blob<Dtype>*> > top_vecs_;//存储了每一层的输出数据
						  vector<vector<int> > top_id_vecs_;//存储了每一层输出id
						  /// Vector of weight in the loss (or objective) function of each net blob,
						  /// indexed by blob_id.
						  vector<Dtype> blob_loss_weights_;//每个blobs的loss weight
						  vector<vector<int> > param_id_vecs_;
						  vector<int> param_owners_;
						  vector<string> param_display_names_;
						  vector<pair<int, int> > param_layer_indices_;
						  map<string, int> param_names_index_;
						  /// blob indices for the input and the output of the net
						  vector<int> net_input_blob_indices_;
						  vector<int> net_output_blob_indices_;
						  vector<Blob<Dtype>*> net_input_blobs_;
						  vector<Blob<Dtype>*> net_output_blobs_;
						  /// The parameters in the network.
						  vector<shared_ptr<Blob<Dtype> > > params_;
						  vector<Blob<Dtype>*> learnable_params_;
						  /**
						   * The mapping from params_ -> learnable_params_: we have
						   * learnable_param_ids_.size() == params_.size(),
						   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
						   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
						   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
						   */
						  vector<int> learnable_param_ids_;
						  /// the learning rate multipliers for learnable_params_
						  vector<float> params_lr_;
						  vector<bool> has_params_lr_;
						  /// the weight decay multipliers for learnable_params_
						  vector<float> params_weight_decay_;
						  vector<bool> has_params_decay_;
						  /// The bytes of memory used by this net
						  size_t memory_used_;
						  /// Whether to compute and display debug info for the net.
						  bool debug_info_;//是否开debug信息
						  // Callbacks
						  //一些callback调用
						  vector<Callback*> before_forward_;
						  vector<Callback*> after_forward_;
						  vector<Callback*> before_backward_;
						  vector<Callback*> after_backward_;

						  DISABLE_COPY_AND_ASSIGN(Net);
				  };


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
