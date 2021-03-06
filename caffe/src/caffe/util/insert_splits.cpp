#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

namespace caffe {//这里主要是为了增加split层

	void InsertSplits(const NetParameter& param, NetParameter* param_split) {
		// Initialize by copying from the input NetParameter.
		param_split->CopyFrom(param);//copy一份配置
		param_split->clear_layer();//清空所有的层
		map<string, pair<int, int> > blob_name_to_last_top_idx;
		map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;
		map<pair<int, int>, int> top_idx_to_bottom_count;
		map<pair<int, int>, float> top_idx_to_loss_weight;
		map<pair<int, int>, int> top_idx_to_bottom_split_idx;
		map<int, string> layer_idx_to_layer_name;//层id和层名字的对应关系
		for (int i = 0; i < param.layer_size(); ++i) {//遍历每一层
			const LayerParameter& layer_param = param.layer(i);//拿到每一层的层结构
			layer_idx_to_layer_name[i] = layer_param.name();//设置层id和层名字的对应关系
			for (int j = 0; j < layer_param.bottom_size(); ++j) {//遍历该层的输入数据
				const string& blob_name = layer_param.bottom(j);//该层的输入的名字
				if (blob_name_to_last_top_idx.find(blob_name) ==
						blob_name_to_last_top_idx.end()) {//每个输入必须有一个输出对应关系
					LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
						<< layer_param.name() << "', bottom index " << j << ")";//如果没有对应关系返回错误
				}
				const pair<int, int>& bottom_idx = make_pair(i, j);//组合成输入的index
				const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];//拿到输出的index
				bottom_idx_to_source_top_idx[bottom_idx] = top_idx;//输入和输出的一个对应关系
				++top_idx_to_bottom_count[top_idx];//标志该输出层链接几个输入
			}
			for (int j = 0; j < layer_param.top_size(); ++j) {//遍历该层的输出数据
				const string& blob_name = layer_param.top(j);//该层的输出名字
				blob_name_to_last_top_idx[blob_name] = make_pair(i, j);//输出的名字和层id 输出id的对应关系，输出的index
			}
			// A use of a top blob as a loss should be handled similarly to the use of
			// a top blob as a bottom blob to another layer.
			const int last_loss =
				std::min(layer_param.loss_weight_size(), layer_param.top_size());//该层输出的数量和lossweight的数量最小值
			for (int j = 0; j < last_loss; ++j) {
				const string& blob_name = layer_param.top(j);//该层输出的名字
				const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];//该层输出的index
				top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);//映射到该loss_weight上面
				if (top_idx_to_loss_weight[top_idx]) {
					++top_idx_to_bottom_count[top_idx];//如果这是一个单纯的loss_weight，就是，不会增加split层的
				}
			}
		}
		for (int i = 0; i < param.layer_size(); ++i) {//遍历每一层
			LayerParameter* layer_param = param_split->add_layer();//添加那一层
			layer_param->CopyFrom(param.layer(i));//copy那一层的数据
			// Replace any shared bottom blobs with split layer outputs.
			for (int j = 0; j < layer_param->bottom_size(); ++j) {//处理它的输入
				const pair<int, int>& top_idx =
					bottom_idx_to_source_top_idx[make_pair(i, j)];
				const int split_count = top_idx_to_bottom_count[top_idx];
				if (split_count > 1) {
					const string& layer_name = layer_idx_to_layer_name[top_idx.first];
					const string& blob_name = layer_param->bottom(j);
					layer_param->set_bottom(j, SplitBlobName(layer_name,
								blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));
				}
			}
			// Create split layer for any top blobs used by other layer as bottom
			// blobs more than once.
			for (int j = 0; j < layer_param->top_size(); ++j) {//处理它的输出
				const pair<int, int>& top_idx = make_pair(i, j);//组合成该层输出的index
				const int split_count = top_idx_to_bottom_count[top_idx];//分割的部分数
				if (split_count > 1) {//输出用作两部分才处理
					const string& layer_name = layer_idx_to_layer_name[i];//该层的名字
					const string& blob_name = layer_param->top(j);//该输出的名字
					LayerParameter* split_layer_param = param_split->add_layer();//添加一层
					const float loss_weight = top_idx_to_loss_weight[top_idx];
					ConfigureSplitLayer(layer_name, blob_name, j, split_count,
							loss_weight, split_layer_param);
					if (loss_weight) {
						layer_param->clear_loss_weight();
						top_idx_to_bottom_split_idx[top_idx]++;
					}
				}
			}
		}
	}

	void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
			const int blob_idx, const int split_count, const float loss_weight,
			LayerParameter* split_layer_param) {//增加split层，一个输入，k个输出，k个输出是一样的，共享数据
		split_layer_param->Clear();
		split_layer_param->add_bottom(blob_name);
		split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
		split_layer_param->set_type("Split");
		for (int k = 0; k < split_count; ++k) {
			split_layer_param->add_top(
					SplitBlobName(layer_name, blob_name, blob_idx, k));
			if (loss_weight) {
				if (k == 0) {
					split_layer_param->add_loss_weight(loss_weight);
				} else {
					split_layer_param->add_loss_weight(0);
				}
			}
		}
	}

	string SplitLayerName(const string& layer_name, const string& blob_name,
			const int blob_idx) {//把各个字段拼接起来而已，层的名字
		ostringstream split_layer_name;
		split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
			<< "_split";
		return split_layer_name.str();
	}

	string SplitBlobName(const string& layer_name, const string& blob_name,
			const int blob_idx, const int split_idx) {//把各个字段拼接起来而已，数据的名字
		ostringstream split_blob_name;
		split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
			<< "_split_" << split_idx;
		return split_blob_name.str();
	}

}  // namespace caffe
