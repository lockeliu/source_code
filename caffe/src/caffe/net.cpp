#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

	template <typename Dtype>
		Net<Dtype>::Net(const NetParameter& param) {//用net配置初始化一个网络结构
			Init(param);
		}

	template <typename Dtype>
		Net<Dtype>::Net(const string& param_file, Phase phase,
				const int level, const vector<string>* stages) {//用配置文件初始化一个网络结构
			NetParameter param;
			ReadNetParamsFromTextFileOrDie(param_file, &param);//读入一个配置文件
			// Set phase, stages and level
			param.mutable_state()->set_phase(phase);//设置网络结构的模式
			if (stages != NULL) {
				for (int i = 0; i < stages->size(); i++) {
					param.mutable_state()->add_stage((*stages)[i]);
				}
			}
			param.mutable_state()->set_level(level);//设置网络的级别
			Init(param);
		}

	template <typename Dtype>
		void Net<Dtype>::Init(const NetParameter& in_param) {
			// Set phase from the state.
			phase_ = in_param.state().phase();//网络结构的模式，train or test
			// Filter layers based on their include/exclude rules and
			// the current NetState.
			NetParameter filtered_param;
			FilterNet(in_param, &filtered_param);//根据一些规则过滤网络结构
			LOG_IF(INFO, Caffe::root_solver())
				<< "Initializing net from parameters: " << std::endl
				<< filtered_param.DebugString();
			// Create a copy of filtered_param with splits added where necessary.
			NetParameter param;
			InsertSplits(filtered_param, &param);//增加split层，共享数据

			// Basically, build all the layers and set up their connections.
			name_ = param.name();//网络的名字
			map<string, int> blob_name_to_idx;//参数和索引的对应关系
			set<string> available_blobs;//是否是可用的参数
			memory_used_ = 0;

			// For each layer, set up its input and output
			bottom_vecs_.resize(param.layer_size());//bottom 的vector 
			top_vecs_.resize(param.layer_size());// top的bector
			bottom_id_vecs_.resize(param.layer_size());//bottom 和索引的对应关系
			param_id_vecs_.resize(param.layer_size());//参数和索引的对应关系 
			top_id_vecs_.resize(param.layer_size());//top和索引的对应关系 
			bottom_need_backward_.resize(param.layer_size());//是否需要反向传播

			for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {//循环每一层
				// Inherit phase from net if unset.
				if (!param.layer(layer_id).has_phase()) {
					param.mutable_layer(layer_id)->set_phase(phase_);//如果这一层没有设置模式参数，把网络的模式参数set进去
				}
				// Setup layer.
				const LayerParameter& layer_param = param.layer(layer_id);//从网络配置拿到这一层的层配置
				if (layer_param.propagate_down_size() > 0) {
					CHECK_EQ(layer_param.propagate_down_size(),//如果有控制反向传播的参数，个数需要和输入参数的个数一致
							layer_param.bottom_size())
						<< "propagate_down param must be specified "
						<< "either 0 or bottom_size times ";
				}

				layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));//工厂初始化一个layer，放在vector里面
				layer_names_.push_back(layer_param.name());//存该层的名字

				LOG_IF(INFO, Caffe::root_solver())
					<< "Creating Layer " << layer_param.name();
				bool need_backward = false;//默认不需要反向传播

				// Figure out this layer's input and output
				// 处理输入层
				for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
						++bottom_id) {
					const int blob_id = AppendBottom(param, layer_id, bottom_id,
							&available_blobs, &blob_name_to_idx);//增加一个输入
					// If a blob needs backward, this layer should provide it.
					need_backward |= blob_need_backward_[blob_id];//反向传播
				}
				//处理输出层
				int num_top = layer_param.top_size();
				for (int top_id = 0; top_id < num_top; ++top_id) {
					AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);//增加一个输出数据
					// Collect Input layer tops as Net inputs.
					if (layer_param.type() == "Input") {
						const int blob_id = blobs_.size() - 1;
						net_input_blob_indices_.push_back(blob_id);
						net_input_blobs_.push_back(blobs_[blob_id].get());//网络的输入数据
					}
				}
				// If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
				// specified fewer than the required number (as specified by
				// ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
				Layer<Dtype>* layer = layers_[layer_id].get();
				if (layer->AutoTopBlobs()) {
					const int needed_num_top =
						std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
					for (; num_top < needed_num_top; ++num_top) {
						// Add "anonymous" top blobs -- do not modify available_blobs or
						// blob_name_to_idx as we don't want these blobs to be usable as input
						// to other layers.
						AppendTop(param, layer_id, num_top, NULL, NULL);
					}
				}

				// After this layer is connected, set it up.
				layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);//处理完该层的输入输出之后，初始化该层
				LOG_IF(INFO, Caffe::root_solver())
					<< "Setting up " << layer_names_[layer_id];
				for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
					if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
						blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));//如果内存不够，重新申请内存
					}
					blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);//设置loss_weight
					LOG_IF(INFO, Caffe::root_solver())
						<< "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
					if (layer->loss(top_id)) {
						LOG_IF(INFO, Caffe::root_solver())
							<< "    with loss weight " << layer->loss(top_id);
					}
					memory_used_ += top_vecs_[layer_id][top_id]->count();//使用的内存大小
				}
				LOG_IF(INFO, Caffe::root_solver())
					<< "Memory required for data: " << memory_used_ * sizeof(Dtype);
				const int param_size = layer_param.param_size();//该层里面的参数个数
				const int num_param_blobs = layers_[layer_id]->blobs().size();//参数的blob 数量
				CHECK_LE(param_size, num_param_blobs)
					<< "Too many params specified for layer " << layer_param.name();
				ParamSpec default_param_spec;
				for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
					const ParamSpec* param_spec = (param_id < param_size) ?
						&layer_param.param(param_id) : &default_param_spec;
					const bool param_need_backward = param_spec->lr_mult() != 0;
					need_backward |= param_need_backward;
					layers_[layer_id]->set_param_propagate_down(param_id,
							param_need_backward);//设置反向传播参数
				}
				for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
					AppendParam(param, layer_id, param_id);//设置每一层权值的一些参数，学习率，正则率，参数id等，其实也是把权重共享给net，感觉没啥必要
				}
				// Finally, set the backward flag
				layer_need_backward_.push_back(need_backward);
				if (need_backward) {
					for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
						blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;//是否反向传播
					}
				}
			}//每一层的循环这里结束

			// Go through the net backwards to determine which blobs contribute to the
			// loss.  We can skip backward computation for blobs that don't contribute
			// to the loss.
			// Also checks if all bottom blobs don't need backward computation (possible
			// because the skip_propagate_down param) and so we can skip bacward
			// computation for the entire layer
			set<string> blobs_under_loss;
			set<string> blobs_skip_backp;
			//一旦某一层禁止反向传播的话，前面的全部禁止反向传播了
			for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
				bool layer_contributes_loss = false;//该层贡献损失
				bool layer_skip_propagate_down = true;//该层是否反向传播
				//处理输出数据
				for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {//处理每个输出
					const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
					if (layers_[layer_id]->loss(top_id) ||
							(blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {//有损失函数 or 
						layer_contributes_loss = true;
					}
					if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
						layer_skip_propagate_down = false;
					}
					if (layer_contributes_loss && !layer_skip_propagate_down)
						break;
				}
				// If this layer can skip backward computation, also all his bottom blobs
				// don't need backpropagation
				if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {//是否跳过反向传播
					layer_need_backward_[layer_id] = false;
					for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();//需要禁止改成所有数据的反向传播
							++bottom_id) {
						bottom_need_backward_[layer_id][bottom_id] = false;
					}
				}
				if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }//如果不贡献损失值，禁止反向传播
				if (Caffe::root_solver()) {
					if (layer_need_backward_[layer_id]) {
						LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
					} else {/
						LOG(INFO) << layer_names_[layer_id]
							<< " does not need backward computation.";
					}
				}
				//处理输入数据
				for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
						++bottom_id) {//遍历每个输入数据
					if (layer_contributes_loss) {//是否贡献损失
						const string& blob_name =
							blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
						blobs_under_loss.insert(blob_name);//贡献损失，才放这里
					} else {
						bottom_need_backward_[layer_id][bottom_id] = false;//反则不需要反向传播
					}
					if (!bottom_need_backward_[layer_id][bottom_id]) {//如果不需要反向传播，需要跳过
						const string& blob_name =
							blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
						blobs_skip_backp.insert(blob_name);//如果需要跳过反向传播
					}
				}
			}
			// Handle force_backward if needed.
			if (param.force_backward()) {//是否强制进行反向传播
				for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {//遍历每一层
					layer_need_backward_[layer_id] = true;//强制进行反向传播
					for (int bottom_id = 0;
							bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {//处理每个输入数据
						bottom_need_backward_[layer_id][bottom_id] =
							bottom_need_backward_[layer_id][bottom_id] ||
							layers_[layer_id]->AllowForceBackward(bottom_id);//判断是否允许强制反向传播
						blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
							blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
							bottom_need_backward_[layer_id][bottom_id];//该层是否反向传播
					}
					for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
							++param_id) {//处理每个参数数据
						layers_[layer_id]->set_param_propagate_down(param_id, true);//强制进行反向传播
					}
				}
			}
			//剩下的就是output了
			// In the end, all remaining blobs are considered output blobs.
			for (set<string>::iterator it = available_blobs.begin();
					it != available_blobs.end(); ++it) {
				LOG_IF(INFO, Caffe::root_solver())
					<< "This network produces output " << *it;
				net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
				net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
			}
			for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
				blob_names_index_[blob_names_[blob_id]] = blob_id;//blob 名字和索引id的对应关系
			}
			for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
				layer_names_index_[layer_names_[layer_id]] = layer_id;//layer名字和索引id的对应关系
			}
			ShareWeights();//处理共享参数权重
			debug_info_ = param.debug_info();//debug信息
			LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
		}

	template <typename Dtype>
		void Net<Dtype>::FilterNet(const NetParameter& param,
				NetParameter* param_filtered) {//根据网络规则过滤一些层
			NetState net_state(param.state());
			param_filtered->CopyFrom(param);//copy 原本的net配置
			param_filtered->clear_layer();//清空所有的层
			for (int i = 0; i < param.layer_size(); ++i) {//遍历所有的层
				const LayerParameter& layer_param = param.layer(i);
				const string& layer_name = layer_param.name();//该层名字
				CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
					<< "Specify either include rules or exclude rules; not both.";
				// If no include rules are specified, the layer is included by default and
				// only excluded if it meets one of the exclude rules.
				bool layer_included = (layer_param.include_size() == 0);//如果等于0，直接通过选用
				for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {//查看排除条件
					if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {//如果符合，直接pass
						layer_included = false;
					}
				}
				for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {//查看包含条件
					if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {//如果符合，直接选用
						layer_included = true;
					}
				}
				if (layer_included) {//通过过滤条件
					param_filtered->add_layer()->CopyFrom(layer_param);//添加入新网络结构
				}
			}
		}

	template <typename Dtype>
		bool Net<Dtype>::StateMeetsRule(const NetState& state,
				const NetStateRule& rule, const string& layer_name) {
			//只要relu是空，就全部是通过了
			// Check whether the rule is broken due to phase.
			if (rule.has_phase()) {//如果rule有phase，才判断
				if (rule.phase() != state.phase()) {//如果不相等直接返回false
					LOG_IF(INFO, Caffe::root_solver())
						<< "The NetState phase (" << state.phase()
						<< ") differed from the phase (" << rule.phase()
						<< ") specified by a rule in layer " << layer_name;
					return false;
				}
			}
			// Check whether the rule is broken due to min level.
			if (rule.has_min_level()) {//如果有min_level，才判断
				if (state.level() < rule.min_level()) {
					LOG_IF(INFO, Caffe::root_solver())
						<< "The NetState level (" << state.level()
						<< ") is above the min_level (" << rule.min_level()
						<< ") specified by a rule in layer " << layer_name;
					return false;
				}
			}
			// Check whether the rule is broken due to max level.
			if (rule.has_max_level()) {//如果有max_level，才判断
				if (state.level() > rule.max_level()) {
					LOG_IF(INFO, Caffe::root_solver())
						<< "The NetState level (" << state.level()
						<< ") is above the max_level (" << rule.max_level()
						<< ") specified by a rule in layer " << layer_name;
					return false;
				}
			}
			// Check whether the rule is broken due to stage. The NetState must
			// contain ALL of the rule's stages to meet it.
			for (int i = 0; i < rule.stage_size(); ++i) {
				// Check that the NetState contains the rule's ith stage.
				bool has_stage = false;
				for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
					if (rule.stage(i) == state.stage(j)) { has_stage = true; }//如果有一个条件符合，就true
				}
				if (!has_stage) {
					LOG_IF(INFO, Caffe::root_solver())
						<< "The NetState did not contain stage '" << rule.stage(i)
						<< "' specified by a rule in layer " << layer_name;
					return false;
				}
			}
			// Check whether the rule is broken due to not_stage. The NetState must
			// contain NONE of the rule's not_stages to meet it.
			for (int i = 0; i < rule.not_stage_size(); ++i) {
				// Check that the NetState contains the rule's ith not_stage.
				bool has_stage = false;
				for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
					if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }//如果有一个条件不符合就false
				}
				if (has_stage) {
					LOG_IF(INFO, Caffe::root_solver())
						<< "The NetState contained a not_stage '" << rule.not_stage(i)
						<< "' specified by a rule in layer " << layer_name;
					return false;
				}
			}
			//最终才true
			return true;
		}

	// Helper for Net::Init: add a new top blob to the net.
	template <typename Dtype>
		void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
				const int top_id, set<string>* available_blobs,
				map<string, int>* blob_name_to_idx) {
			shared_ptr<LayerParameter> layer_param(
					new LayerParameter(param.layer(layer_id)));
			const string& blob_name = (layer_param->top_size() > top_id) ?
				layer_param->top(top_id) : "(automatic)";
			// Check if we are doing in-place computation
			if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
					blob_name == layer_param->bottom(top_id)) {//同层共享输入输出数据了
				// In-place computation
				LOG_IF(INFO, Caffe::root_solver())
					<< layer_param->name() << " -> " << blob_name << " (in-place)";
				top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
				top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
			} else if (blob_name_to_idx &&
					blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {//输入重名
				// If we are not doing in-place computation but have duplicated blobs,
				// raise an error.
				LOG(FATAL) << "Top blob '" << blob_name
					<< "' produced by multiple sources.";
			} else {//正常的走这里
				// Normal output.
				if (Caffe::root_solver()) {
					LOG(INFO) << layer_param->name() << " -> " << blob_name;
				}
				shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());//申请一个数据块
				const int blob_id = blobs_.size();//数据块的id
				blobs_.push_back(blob_pointer);//存所有的输入输出
				blob_names_.push_back(blob_name);//存这个数据块的名字
				blob_need_backward_.push_back(false);//设置不需要反向传播
				if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }//数据块名字和id的映射关系
				top_id_vecs_[layer_id].push_back(blob_id);//放到输出vector里面
				top_vecs_[layer_id].push_back(blob_pointer.get());//放到输出vector里面
			}
			if (available_blobs) { available_blobs->insert(blob_name); }//增加可用的层
		}

	// Helper for Net::Init: add a new bottom blob to the net.
	//处理输入数据，blob存了所有中间数据，available存了出现过的blob
	template <typename Dtype>
		int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
				const int bottom_id, set<string>* available_blobs,
				map<string, int>* blob_name_to_idx) {//处理bottom_vecs_,bottom_id_vecs_,available_blobs,bottom_need_backward_
			const LayerParameter& layer_param = param.layer(layer_id);
			const string& blob_name = layer_param.bottom(bottom_id);
			if (available_blobs->find(blob_name) == available_blobs->end()) {//输入层的名字需要在输出层中已经出现过才行的
				LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
					<< layer_param.name() << "', bottom index " << bottom_id << ")";
			}
			const int blob_id = (*blob_name_to_idx)[blob_name];//层的名字 索引 到层的id
			LOG_IF(INFO, Caffe::root_solver())
				<< layer_names_[layer_id] << " <- " << blob_name;
			bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());//vector套vector存每一层的输入
			bottom_id_vecs_[layer_id].push_back(blob_id);//vector套vector套每一层的输入的索引id
			available_blobs->erase(blob_name);//删除这个层的名字
			bool need_backward = blob_need_backward_[blob_id];//是否需要反向传播
			// Check if the backpropagation on bottom_id should be skipped
			if (layer_param.propagate_down_size() > 0) {
				need_backward = layer_param.propagate_down(bottom_id);
			}
			bottom_need_backward_[layer_id].push_back(need_backward);//是否需要方向传播
			return blob_id;
		}

	template <typename Dtype>
		void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
				const int param_id) {
			const LayerParameter& layer_param = layers_[layer_id]->layer_param();
			const int param_size = layer_param.param_size();//该层参数的个数
			string param_name =
				(param_size > param_id) ? layer_param.param(param_id).name() : "";//该层的名字
			if (param_name.size()) {//如果该层参数有名字
				param_display_names_.push_back(param_name);
			} else {//如果该层没填参数名字
				ostringstream param_display_name;
				param_display_name << param_id;//这个是layer的参数id
				param_display_names_.push_back(param_display_name.str());//直接用参数id当作名字
			}
			const int net_param_id = params_.size();//网络参数id
			params_.push_back(layers_[layer_id]->blobs()[param_id]);//放在params vector里面
			param_id_vecs_[layer_id].push_back(net_param_id);//参数和参数id的对应关系
			param_layer_indices_.push_back(make_pair(layer_id, param_id));//layer id 和 参数 id的对应关系
			ParamSpec default_param_spec;
			const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
				&layer_param.param(param_id) : &default_param_spec;//拿到当前的参数，copy了一份出来
			if (!param_size || !param_name.size() || (param_name.size() &&
						param_names_index_.find(param_name) == param_names_index_.end())) {//参数为空   or 参数名为空 or 以前不存在过这个参数名，那么这个参数是自己独享的
				// This layer "owns" this parameter blob -- it is either anonymous
				// (i.e., not given a param_name) or explicitly given a name that we
				// haven't already seen.
				param_owners_.push_back(-1);//-1 表示独享这个参数
				if (param_name.size()) {
					param_names_index_[param_name] = net_param_id;//参数名字核索引的对应关系
				}
				const int learnable_param_id = learnable_params_.size();//可学习参数的id
				learnable_params_.push_back(params_[net_param_id].get());//放入可学习参数vector中
				learnable_param_ids_.push_back(learnable_param_id);//放入可学习参数idvector中
				has_params_lr_.push_back(param_spec->has_lr_mult());//该参数是否有学习率
				has_params_decay_.push_back(param_spec->has_decay_mult());//该参数是否有衰变率
				params_lr_.push_back(param_spec->lr_mult());//放入学习率
				params_weight_decay_.push_back(param_spec->decay_mult());//放入学习衰变率
			} else {//以下是共享学习参数
				// Named param blob with name we've seen before: share params
				const int owner_net_param_id = param_names_index_[param_name];//拿出以前的参数id
				param_owners_.push_back(owner_net_param_id);//把参数id放进去
				const pair<int, int>& owner_index =
					param_layer_indices_[owner_net_param_id];//拿出layer id 和参数id
				const int owner_layer_id = owner_index.first;//拿出来共享的层id
				const int owner_param_id = owner_index.second;//拿出共享的层参数id
				LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
					<< "' owned by "
					<< "layer '" << layer_names_[owner_layer_id] << "', param "
					<< "index " << owner_param_id;
				Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();//拿到该层的数据
				Blob<Dtype>* owner_blob =
					layers_[owner_layer_id]->blobs()[owner_param_id].get();//共享参数数据
				const int param_size = layer_param.param_size();//参数个数
				if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
							ParamSpec_DimCheckMode_PERMISSIVE)) {//一种模式吧
					// Permissive dimension checking -- only check counts are the same.
					CHECK_EQ(this_blob->count(), owner_blob->count())//共享数据间大小要一样
						<< "Cannot share param '" << param_name << "' owned by layer '"
						<< layer_names_[owner_layer_id] << "' with layer '"
						<< layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
						<< "shape is " << owner_blob->shape_string() << "; sharing layer "
						<< "shape is " << this_blob->shape_string();
				} else {
					// Strict dimension checking -- all dims must be the same.
					CHECK(this_blob->shape() == owner_blob->shape())//共享数据间，维度信息要一致
						<< "Cannot share param '" << param_name << "' owned by layer '"
						<< layer_names_[owner_layer_id] << "' with layer '"
						<< layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
						<< "shape is " << owner_blob->shape_string() << "; sharing layer "
						<< "expects shape " << this_blob->shape_string();
				}
				const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
				learnable_param_ids_.push_back(learnable_param_id);//参数id，共享了参数id
				//处理学习率
				if (param_spec->has_lr_mult()) {
					if (has_params_lr_[learnable_param_id]) {//如果本身有学习率，，需要相等
						CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
							<< "Shared param '" << param_name << "' has mismatched lr_mult.";
					} else {//如果没有，重新设置
						has_params_lr_[learnable_param_id] = true;
						params_lr_[learnable_param_id] = param_spec->lr_mult();
					}
				}
				//处理衰变率
				if (param_spec->has_decay_mult()) {
					if (has_params_decay_[learnable_param_id]) {//如果本身有衰变率，需要相等
						CHECK_EQ(param_spec->decay_mult(),
								params_weight_decay_[learnable_param_id])
							<< "Shared param '" << param_name << "' has mismatched decay_mult.";
					} else {//如果没有，重新设置
						has_params_decay_[learnable_param_id] = true;
						params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
					}
				}
			}
		}

	//上面的是初始化	
	template <typename Dtype>
		Dtype Net<Dtype>::ForwardFromTo(int start, int end) {//从start到end，前向传播
			CHECK_GE(start, 0);
			CHECK_LT(end, layers_.size());
			Dtype loss = 0;
			for (int i = start; i <= end; ++i) {
				for (int c = 0; c < before_forward_.size(); ++c) {
					before_forward_[c]->run(i);//一些callback，可以不管
				}
				Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);//对每一层进行前向传播
				loss += layer_loss;
				if (debug_info_) { ForwardDebugInfo(i); }//打印debug信息而已
				for (int c = 0; c < after_forward_.size(); ++c) {
					after_forward_[c]->run(i);//一些callback，可以不管
				}
			}
			return loss;
		}

	template <typename Dtype>
		Dtype Net<Dtype>::ForwardFrom(int start) {
			return ForwardFromTo(start, layers_.size() - 1);
		}

	template <typename Dtype>
		Dtype Net<Dtype>::ForwardTo(int end) {
			return ForwardFromTo(0, end);
		}

	template <typename Dtype>
		const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {//前向传播
			if (loss != NULL) {
				*loss = ForwardFromTo(0, layers_.size() - 1);
			} else {
				ForwardFromTo(0, layers_.size() - 1);
			}
			return net_output_blobs_;
		}

	template <typename Dtype>
		const vector<Blob<Dtype>*>& Net<Dtype>::Forward(//指定输入，开始前向传播
				const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
			LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
				<< "will be removed in a future version. Use Forward(loss).";
			// Copy bottom to net bottoms
			for (int i = 0; i < bottom.size(); ++i) {
				net_input_blobs_[i]->CopyFrom(*bottom[i]);
			}
			return Forward(loss);
		}

	template <typename Dtype>
		void Net<Dtype>::BackwardFromTo(int start, int end) {//从start到end，反向传播
			CHECK_GE(end, 0);
			CHECK_LT(start, layers_.size());
			for (int i = start; i >= end; --i) {
				for (int c = 0; c < before_backward_.size(); ++c) {
					before_backward_[c]->run(i);//callback
				}
				if (layer_need_backward_[i]) {
					layers_[i]->Backward(
							top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);//反向传播
					if (debug_info_) { BackwardDebugInfo(i); }//打印debug信息而已
				}
				for (int c = 0; c < after_backward_.size(); ++c) {
					after_backward_[c]->run(i);//callback
				}
			}
		}

	template <typename Dtype>
		void Net<Dtype>::ForwardDebugInfo(const int layer_id) {//打印前向传播的debug信息
			for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
				const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
				const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
				const Dtype data_abs_val_mean = blob.asum_data() / blob.count();//L1范数
				LOG_IF(INFO, Caffe::root_solver())
					<< "    [Forward] "
					<< "Layer " << layer_names_[layer_id]
					<< ", top blob " << blob_name
					<< " data: " << data_abs_val_mean;//L1范数
			}
			for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
					++param_id) {
				const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
				const int net_param_id = param_id_vecs_[layer_id][param_id];
				const string& blob_name = param_display_names_[net_param_id];
				const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
				LOG_IF(INFO, Caffe::root_solver())
					<< "    [Forward] "
					<< "Layer " << layer_names_[layer_id]
					<< ", param blob " << blob_name
					<< " data: " << data_abs_val_mean;//L1范数
			}
		}

	template <typename Dtype>
		void Net<Dtype>::BackwardDebugInfo(const int layer_id) {//反向传播的debug信息
			const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
			for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
				if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
				const Blob<Dtype>& blob = *bottom_vec[bottom_id];
				const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
				const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
				LOG_IF(INFO, Caffe::root_solver())
					<< "    [Backward] "
					<< "Layer " << layer_names_[layer_id]
					<< ", bottom blob " << blob_name
					<< " diff: " << diff_abs_val_mean;//L1范数
			}
			for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
					++param_id) {
				if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
				const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
				const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
				LOG_IF(INFO, Caffe::root_solver())
					<< "    [Backward] "
					<< "Layer " << layer_names_[layer_id]
					<< ", param blob " << param_id
					<< " diff: " << diff_abs_val_mean;
			}
		}

	template <typename Dtype>
		void Net<Dtype>::UpdateDebugInfo(const int param_id) {//输出debug信息
			const Blob<Dtype>& blob = *params_[param_id];
			const int param_owner = param_owners_[param_id];
			const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
			const string& param_display_name = param_display_names_[param_id];
			const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
			if (param_owner < 0) {
				const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
				LOG_IF(INFO, Caffe::root_solver())
					<< "    [Update] Layer " << layer_name
					<< ", param " << param_display_name
					<< " data: " << data_abs_val_mean
					<< "; diff: " << diff_abs_val_mean;
			} else {
				const string& owner_layer_name =
					layer_names_[param_layer_indices_[param_owner].first];
				LOG_IF(INFO, Caffe::root_solver())
					<< "    [Update] Layer " << layer_name
					<< ", param blob " << param_display_name
					<< " (owned by layer " << owner_layer_name << ", " << "param "
					<< param_display_names_[param_owners_[param_id]] << ")"
					<< " diff: " << diff_abs_val_mean;
			}
		}

	template <typename Dtype>
		void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
			int num_source_layers = other->layers().size();//other 网络结构的层数
			for (int i = 0; i < num_source_layers; ++i) {
				Layer<Dtype>* source_layer = other->layers()[i].get();//拿出某一层
				const string& source_layer_name = other->layer_names()[i];//这一层的名字
				int target_layer_id = 0;
				while (target_layer_id != layer_names_.size() &&
						layer_names_[target_layer_id] != source_layer_name) {//在本网络结构中找名字相同的那一层
					++target_layer_id;
				}
				if (target_layer_id == layer_names_.size()) {//找不到，跳过
					LOG(INFO) << "Ignoring source layer " << source_layer_name;
					continue;
				}
				DLOG(INFO) << "Copying source layer " << source_layer_name;
				vector<shared_ptr<Blob<Dtype> > >& target_blobs =
					layers_[target_layer_id]->blobs();//拿出找到这一层的所有训练参数
				CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
					<< "Incompatible number of blobs for layer " << source_layer_name;//两个的大小应该一致
				for (int j = 0; j < target_blobs.size(); ++j) {
					Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
					CHECK(target_blobs[j]->shape() == source_blob->shape())//该参数的大小要一致
						<< "Cannot share param " << j << " weights from layer '"
						<< source_layer_name << "'; shape mismatch.  Source param shape is "
						<< source_blob->shape_string() << "; target param shape is "
						<< target_blobs[j]->shape_string();
					target_blobs[j]->ShareData(*source_blob);//共享参数
				}
			}
		}

	template <typename Dtype>
		void Net<Dtype>::BackwardFrom(int start) {//从start反向传播到0
			BackwardFromTo(start, 0);
		}

	template <typename Dtype>
		void Net<Dtype>::BackwardTo(int end) {//末尾反向传播end
			BackwardFromTo(layers_.size() - 1, end);
		}

	template <typename Dtype>
		void Net<Dtype>::Backward() {//从末尾开始反向传播到开头
			BackwardFromTo(layers_.size() - 1, 0);
			if (debug_info_) {
				Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
				for (int i = 0; i < learnable_params_.size(); ++i) {
					asum_data += learnable_params_[i]->asum_data();
					asum_diff += learnable_params_[i]->asum_diff();
					sumsq_data += learnable_params_[i]->sumsq_data();
					sumsq_diff += learnable_params_[i]->sumsq_diff();
				}
				const Dtype l2norm_data = std::sqrt(sumsq_data);
				const Dtype l2norm_diff = std::sqrt(sumsq_diff);
				LOG(ERROR) << "    [Backward] All net params (data, diff): "
					<< "L1 norm = (" << asum_data << ", " << asum_diff << "); "
					<< "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
			}
		}

	template <typename Dtype>
		void Net<Dtype>::Reshape() {//遍历每一层，逐步reshape
			for (int i = 0; i < layers_.size(); ++i) {
				layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
			}
		}

	template <typename Dtype>
		void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {//和ShareTrainedLayersWith基本一致
			int num_source_layers = param.layer_size();
			for (int i = 0; i < num_source_layers; ++i) {//遍历每一层
				const LayerParameter& source_layer = param.layer(i);
				const string& source_layer_name = source_layer.name();
				int target_layer_id = 0;
				while (target_layer_id != layer_names_.size() &&
						layer_names_[target_layer_id] != source_layer_name) {//用名字搜索那一层
					++target_layer_id;
				}
				if (target_layer_id == layer_names_.size()) {//找不到跳过
					LOG(INFO) << "Ignoring source layer " << source_layer_name;
					continue;
				}
				DLOG(INFO) << "Copying source layer " << source_layer_name;
				vector<shared_ptr<Blob<Dtype> > >& target_blobs =
					layers_[target_layer_id]->blobs();
				CHECK_EQ(target_blobs.size(), source_layer.blobs_size())//参数个数要一致
					<< "Incompatible number of blobs for layer " << source_layer_name;
				for (int j = 0; j < target_blobs.size(); ++j) {
					if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {//判断参数是否相等
						Blob<Dtype> source_blob;
						const bool kReshape = true;
						source_blob.FromProto(source_layer.blobs(j), kReshape);
						LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
							<< source_layer_name << "'; shape mismatch.  Source param shape is "
							<< source_blob.shape_string() << "; target param shape is "
							<< target_blobs[j]->shape_string() << ". "
							<< "To learn this layer's parameters from scratch rather than "
							<< "copying from a saved net, rename the layer.";
					}
					const bool kReshape = false;
					target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);//共享参数
				}
			}
		}

	template <typename Dtype>
		void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {//从文件中导入网络参数
			if (H5Fis_hdf5(trained_filename.c_str())) {
				CopyTrainedLayersFromHDF5(trained_filename);
			} else {
				CopyTrainedLayersFromBinaryProto(trained_filename);
			}
		}

	template <typename Dtype>
		void Net<Dtype>::CopyTrainedLayersFromBinaryProto(//从pb中导入网络参数
				const string trained_filename) {
			NetParameter param;
			ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
			CopyTrainedLayersFrom(param);
		}

	template <typename Dtype>
		void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {//共享权重文件
			hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
					H5P_DEFAULT);
			CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
			hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
			CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
			int num_layers = hdf5_get_num_links(data_hid);
			for (int i = 0; i < num_layers; ++i) {
				string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
				if (!layer_names_index_.count(source_layer_name)) {
					LOG(INFO) << "Ignoring source layer " << source_layer_name;
					continue;
				}
				int target_layer_id = layer_names_index_[source_layer_name];
				DLOG(INFO) << "Copying source layer " << source_layer_name;
				vector<shared_ptr<Blob<Dtype> > >& target_blobs =
					layers_[target_layer_id]->blobs();
				hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
						H5P_DEFAULT);
				CHECK_GE(layer_hid, 0)
					<< "Error reading weights from " << trained_filename;
				// Check that source layer doesn't have more params than target layer
				int num_source_params = hdf5_get_num_links(layer_hid);
				CHECK_LE(num_source_params, target_blobs.size())
					<< "Incompatible number of blobs for layer " << source_layer_name;
				for (int j = 0; j < target_blobs.size(); ++j) {
					ostringstream oss;
					oss << j;
					string dataset_name = oss.str();
					int target_net_param_id = param_id_vecs_[target_layer_id][j];
					if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
						// Target param doesn't exist in source weights...
						if (param_owners_[target_net_param_id] != -1) {
							// ...but it's weight-shared in target, so that's fine.
							continue;
						} else {
							LOG(FATAL) << "Incompatible number of blobs for layer "
								<< source_layer_name;
						}
					}
					hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
							target_blobs[j].get());
				}
				H5Gclose(layer_hid);
			}
			H5Gclose(data_hid);
			H5Fclose(file_hid);
		}

	template <typename Dtype>
		void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {//序列化成pb
			param->Clear();
			param->set_name(name_);
			// Add bottom and top
			DLOG(INFO) << "Serializing " << layers_.size() << " layers";
			for (int i = 0; i < layers_.size(); ++i) {
				LayerParameter* layer_param = param->add_layer();
				layers_[i]->ToProto(layer_param, write_diff);
			}
		}

	template <typename Dtype>
		void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {//转成hdf5
			hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
					H5P_DEFAULT);
			CHECK_GE(file_hid, 0)
				<< "Couldn't open " << filename << " to save weights.";
			hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
					H5P_DEFAULT);
			CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
			hid_t diff_hid = -1;
			if (write_diff) {
				diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
						H5P_DEFAULT);
				CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
			}
			for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
				const LayerParameter& layer_param = layers_[layer_id]->layer_param();
				string layer_name = layer_param.name();
				hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
						H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
				CHECK_GE(layer_data_hid, 0)
					<< "Error saving weights to " << filename << ".";
				hid_t layer_diff_hid = -1;
				if (write_diff) {
					layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
							H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
					CHECK_GE(layer_diff_hid, 0)
						<< "Error saving weights to " << filename << ".";
				}
				int num_params = layers_[layer_id]->blobs().size();
				for (int param_id = 0; param_id < num_params; ++param_id) {
					ostringstream dataset_name;
					dataset_name << param_id;
					const int net_param_id = param_id_vecs_[layer_id][param_id];
					if (param_owners_[net_param_id] == -1) {
						// Only save params that own themselves
						hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
								*params_[net_param_id]);
					}
					if (write_diff) {
						// Write diffs regardless of weight-sharing
						hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
								*params_[net_param_id], true);
					}
				}
				H5Gclose(layer_data_hid);
				if (write_diff) {
					H5Gclose(layer_diff_hid);
				}
			}
			H5Gclose(data_hid);
			if (write_diff) {
				H5Gclose(diff_hid);
			}
			H5Fclose(file_hid);
		}

	template <typename Dtype>
		void Net<Dtype>::Update() {//用差异矩阵更新每一层的权重
			for (int i = 0; i < learnable_params_.size(); ++i) {
				learnable_params_[i]->Update();
			}
		}

	template <typename Dtype>
		void Net<Dtype>::ClearParamDiffs() {//清空所有的层差异矩阵
			for (int i = 0; i < learnable_params_.size(); ++i) {
				Blob<Dtype>* blob = learnable_params_[i];
				switch (Caffe::mode()) {
					case Caffe::CPU:
						caffe_set(blob->count(), static_cast<Dtype>(0),
								blob->mutable_cpu_diff());
						break;
					case Caffe::GPU:
#ifndef CPU_ONLY
						caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
								blob->mutable_gpu_diff());
#else
						NO_GPU;
#endif
						break;
				}
			}
		}

	template <typename Dtype>
		void Net<Dtype>::ShareWeights() {//处理那种共享权重参数
			for (int i = 0; i < params_.size(); ++i) {
				if (param_owners_[i] < 0) { continue; }
				params_[i]->ShareData(*params_[param_owners_[i]]);
				params_[i]->ShareDiff(*params_[param_owners_[i]]);
			}
		}

	template <typename Dtype>
		bool Net<Dtype>::has_blob(const string& blob_name) const {//用blobs名字查找是否有这一层
			return blob_names_index_.find(blob_name) != blob_names_index_.end();
		}

	template <typename Dtype>
		const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(//用blob名字找这一层
				const string& blob_name) const {
			shared_ptr<Blob<Dtype> > blob_ptr;
			if (has_blob(blob_name)) {
				blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
			} else {
				blob_ptr.reset((Blob<Dtype>*)(NULL));
				LOG(WARNING) << "Unknown blob name " << blob_name;
			}
			return blob_ptr;
		}

	template <typename Dtype>
		bool Net<Dtype>::has_layer(const string& layer_name) const {//用layer名字查找是否有这一层
			return layer_names_index_.find(layer_name) != layer_names_index_.end();
		}

	template <typename Dtype>
		const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
				const string& layer_name) const {//用layer名字查找这一层
			shared_ptr<Layer<Dtype> > layer_ptr;
			if (has_layer(layer_name)) {
				layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
			} else {
				layer_ptr.reset((Layer<Dtype>*)(NULL));
				LOG(WARNING) << "Unknown layer name " << layer_name;
			}
			return layer_ptr;
		}

	INSTANTIATE_CLASS(Net);

}  // namespace caffe
