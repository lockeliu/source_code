#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

	template<typename Dtype>
		void Solver<Dtype>::SetActionFunction(ActionCallback func) {//callback
			action_request_function_ = func;
		}

	template<typename Dtype>
		SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
			if (action_request_function_) {
				// If the external request function has been set, call it.
				return action_request_function_();
			}
			return SolverAction::NONE;
		}

	template <typename Dtype>
		Solver<Dtype>::Solver(const SolverParameter& param)
		: net_(), callbacks_(), requested_early_exit_(false) {//用solver的pb初始化
			Init(param);
		}

	template <typename Dtype>
		Solver<Dtype>::Solver(const string& param_file)//用solver的配置文件初始化
		: net_(), callbacks_(), requested_early_exit_(false) {
			SolverParameter param;
			ReadSolverParamsFromTextFileOrDie(param_file, &param);
			Init(param);
		}

	template <typename Dtype>
		void Solver<Dtype>::Init(const SolverParameter& param) {//真正的solver初始化
			LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
				<< std::endl << param.DebugString();
			param_ = param;//赋值配置
			CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
			CheckSnapshotWritePermissions();//检查模型是否具有写入权限，防止训练完没有写权限保存模型就悲剧了
			if (param_.random_seed() >= 0) {
				Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());
			}
			// Scaffolding code
			InitTrainNet();//初始化训练solver
			InitTestNets();//初始化测试solver
			if (Caffe::root_solver()) {
				LOG(INFO) << "Solver scaffolding done.";
			}
			iter_ = 0;//迭代次数
			current_step_ = 0;//没用的参数
		}

	template <typename Dtype>
		void Solver<Dtype>::InitTrainNet() {
			const int num_train_nets = param_.has_net() + param_.has_net_param() +
				param_.has_train_net() + param_.has_train_net_param();//网络个数？
			const string& field_names = "net, net_param, train_net, train_net_param";
			CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
				<< "using one of these fields: " << field_names;
			CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
				<< "one of these fields specifying a train_net: " << field_names;
			NetParameter net_param;//复制net配置到net_param
			if (param_.has_train_net_param()) {//pb
				LOG_IF(INFO, Caffe::root_solver())
					<< "Creating training net specified in train_net_param.";
				net_param.CopyFrom(param_.train_net_param());
			} else if (param_.has_train_net()) {//文件
				LOG_IF(INFO, Caffe::root_solver())
					<< "Creating training net from train_net file: " << param_.train_net();
				ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
			}
			if (param_.has_net_param()) {//pb
				LOG_IF(INFO, Caffe::root_solver())
					<< "Creating training net specified in net_param.";
				net_param.CopyFrom(param_.net_param());
			}
			if (param_.has_net()) {//文件
				LOG_IF(INFO, Caffe::root_solver())
					<< "Creating training net from net file: " << param_.net();
				ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
			}
			// Set the correct NetState.  We start with the solver defaults (lowest
			// precedence); then, merge in any NetState specified by the net_param itself;
			// finally, merge in any NetState specified by the train_state (highest
			// precedence).
			NetState net_state;//设置网络的状态
			net_state.set_phase(TRAIN);//设置网络的模式
			net_state.MergeFrom(net_param.state());
			net_state.MergeFrom(param_.train_state());//网络规则
			net_param.mutable_state()->CopyFrom(net_state);//网络规则
			net_.reset(new Net<Dtype>(net_param));//用net配置初始化网络结构
		}

	template <typename Dtype>
		void Solver<Dtype>::InitTestNets() {
			const bool has_net_param = param_.has_net_param();
			const bool has_net_file = param_.has_net();
			const int num_generic_nets = has_net_param + has_net_file;
			CHECK_LE(num_generic_nets, 1)
				<< "Both net_param and net_file may not be specified.";
			const int num_test_net_params = param_.test_net_param_size();
			const int num_test_net_files = param_.test_net_size();
			const int num_test_nets = num_test_net_params + num_test_net_files;
			if (num_generic_nets) {
				CHECK_GE(param_.test_iter_size(), num_test_nets)
					<< "test_iter must be specified for each test network.";
			} else {
				CHECK_EQ(param_.test_iter_size(), num_test_nets)//测试迭代测试的个数必须和测试网络结构个数相等
					<< "test_iter must be specified for each test network.";
			}
			// If we have a generic net (specified by net or net_param, rather than
			// test_net or test_net_param), we may have an unlimited number of actual
			// test networks -- the actual number is given by the number of remaining
			// test_iters after any test nets specified by test_net_param and/or test_net
			// are evaluated.
			const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
			const int num_test_net_instances = num_test_nets + num_generic_net_instances;
			if (param_.test_state_size()) {
				CHECK_EQ(param_.test_state_size(), num_test_net_instances)//和测试网络个数需要相等
					<< "test_state must be unspecified or specified once per test net.";
			}
			if (num_test_net_instances) {
				CHECK_GT(param_.test_interval(), 0);//只要有测试网络就必须有这个
			}
			int test_net_id = 0;
			vector<string> sources(num_test_net_instances);//存放网络名字，debug用而已
			vector<NetParameter> net_params(num_test_net_instances);//存放网络结构配置
			for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {//从pb copy网络结构到pb
				sources[test_net_id] = "test_net_param";
				net_params[test_net_id].CopyFrom(param_.test_net_param(i));
			}
			for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {//从文件copy网络结构到pb
				sources[test_net_id] = "test_net file: " + param_.test_net(i);
				ReadNetParamsFromTextFileOrDie(param_.test_net(i),
						&net_params[test_net_id]);
			}
			const int remaining_test_nets = param_.test_iter_size() - test_net_id;//还剩下多少个网络结构
			//下面两个是互斥的，一个出现，另外就不会出现
			if (has_net_param) {
				for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
					sources[test_net_id] = "net_param";//网络名字
					net_params[test_net_id].CopyFrom(param_.net_param());
				}
			}
			if (has_net_file) {
				for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
					sources[test_net_id] = "net file: " + param_.net();
					ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
				}
			}
			test_nets_.resize(num_test_net_instances);
			for (int i = 0; i < num_test_net_instances; ++i) {//遍历每一个测试网络结构
				// Set the correct NetState.  We start with the solver defaults (lowest
				// precedence); then, merge in any NetState specified by the net_param
				// itself; finally, merge in any NetState specified by the test_state
				// (highest precedence).
				NetState net_state;//网络结构的规则
				net_state.set_phase(TEST);//测试模式
				net_state.MergeFrom(net_params[i].state());
				if (param_.test_state_size()) {
					net_state.MergeFrom(param_.test_state(i));
				}
				net_params[i].mutable_state()->CopyFrom(net_state);//复制网络规则
				LOG(INFO)
					<< "Creating test net (#" << i << ") specified by " << sources[i];
				test_nets_[i].reset(new Net<Dtype>(net_params[i]));//用网络配置初始化网络结构
				test_nets_[i]->set_debug_info(param_.debug_info());//设置debug信息
			}
		}

	template <typename Dtype>
		void Solver<Dtype>::Step(int iters) {
			const int start_iter = iter_;
			const int stop_iter = iter_ + iters;
			int average_loss = this->param_.average_loss();
			losses_.clear();
			smoothed_loss_ = 0;
			iteration_timer_.Start();

			while (iter_ < stop_iter) {//训练迭代
				// zero-init the params
				net_->ClearParamDiffs();//清空所有的差异矩阵
				if (param_.test_interval() && iter_ % param_.test_interval() == 0
						&& (iter_ > 0 || param_.test_initialization())) {//判断是否需要到测试的时候了
					if (Caffe::root_solver()) {
						TestAll();
					}
					if (requested_early_exit_) {
						// Break out of the while loop because stop was requested while testing.
						break;
					}
				}
				//调用callback函数
				for (int i = 0; i < callbacks_.size(); ++i) {
					callbacks_[i]->on_start();
				}
				const bool display = param_.display() && iter_ % param_.display() == 0;//是否展示
				net_->set_debug_info(display && param_.debug_info());
				// accumulate the loss and gradient
				Dtype loss = 0;
				for (int i = 0; i < param_.iter_size(); ++i) {
					loss += net_->ForwardBackward();//进行iter_size次前向传播反向传播
				}
				loss /= param_.iter_size();//对loss求平均值
				// average the loss across iterations for smoothed reporting
				UpdateSmoothedLoss(loss, start_iter, average_loss);
				if (display) {//展示
					float lapse = iteration_timer_.Seconds();
					float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
					LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
						<< " (" << per_s << " iter/s, " << lapse << "s/"
						<< param_.display() << " iters), loss = " << smoothed_loss_;//耗时，loss等信息
					iteration_timer_.Start();
					iterations_last_ = iter_;
					const vector<Blob<Dtype>*>& result = net_->output_blobs();
					int score_index = 0;
					for (int j = 0; j < result.size(); ++j) {//展示acc和loss里面的值
						const Dtype* result_vec = result[j]->cpu_data();
						const string& output_name =
							net_->blob_names()[net_->output_blob_indices()[j]];
						const Dtype loss_weight =
							net_->blob_loss_weights()[net_->output_blob_indices()[j]];
						for (int k = 0; k < result[j]->count(); ++k) {
							ostringstream loss_msg_stream;
							if (loss_weight) {
								loss_msg_stream << " (* " << loss_weight
									<< " = " << loss_weight * result_vec[k] << " loss)";
							}
							LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
								<< score_index++ << ": " << output_name << " = "
								<< result_vec[k] << loss_msg_stream.str();
						}
					}
				}
				//处理完之后的一个callback
				for (int i = 0; i < callbacks_.size(); ++i) {
					callbacks_[i]->on_gradients_ready();
				}
				ApplyUpdate();//进行一次update

				// Increment the internal iter_ counter -- its value should always indicate
				// the number of times the weights have been updated.
				++iter_;

				SolverAction::Enum request = GetRequestedAction();

				// Save a snapshot if needed.
				if ((param_.snapshot()
							&& iter_ % param_.snapshot() == 0
							&& Caffe::root_solver()) ||
						(request == SolverAction::SNAPSHOT)) {//是否到需要保存快照的次数
					Snapshot();
				}
				if (SolverAction::STOP == request) {
					requested_early_exit_ = true;
					// Break out of training loop.
					break;
				}
			}
		}

	template <typename Dtype>
		void Solver<Dtype>::Solve(const char* resume_file) {//训练的函数
			CHECK(Caffe::root_solver());
			LOG(INFO) << "Solving " << net_->name();
			LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

			// Initialize to false every time we start solving.
			requested_early_exit_ = false;

			if (resume_file) {
				LOG(INFO) << "Restoring previous solver status from " << resume_file;
				Restore(resume_file);//加载配置文件
			}

			// For a network that is trained by the solver, no bottom or top vecs
			// should be given, and we will just provide dummy vecs.
			int start_iter = iter_;//当前的迭代次数，支持断点续传，只要保存了中间文件
			Step(param_.max_iter() - iter_);//迭代这么多次
			// If we haven't already, save a snapshot after optimization, unless
			// overridden by setting snapshot_after_train := false
			if (param_.snapshot_after_train()//训练完成后最后一次保存模型问津
					&& (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
				Snapshot();
			}
			if (requested_early_exit_) {
				LOG(INFO) << "Optimization stopped early.";
				return;
			}
			// After the optimization is done, run an additional train and test pass to
			// display the train and test loss/outputs if appropriate (based on the
			// display and test_interval settings, respectively).  Unlike in the rest of
			// training, for the train net we only run a forward pass as we've already
			// updated the parameters "max_iter" times -- this final pass is only done to
			// display the loss, which is computed in the forward pass.
			if (param_.display() && iter_ % param_.display() == 0) {//计算损失函数
				int average_loss = this->param_.average_loss();
				Dtype loss;
				net_->Forward(&loss);//前向传播

				UpdateSmoothedLoss(loss, start_iter, average_loss);

				LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
			}
			if (param_.test_interval() && iter_ % param_.test_interval() == 0) {//如果需要测试，就测试所有的测试网络
				TestAll();
			}
			LOG(INFO) << "Optimization Done.";
		}

	template <typename Dtype>
		void Solver<Dtype>::TestAll() {
			for (int test_net_id = 0;
					test_net_id < test_nets_.size() && !requested_early_exit_;
					++test_net_id) {//测试所有的测试网络结构
				Test(test_net_id);
			}
		}

	template <typename Dtype>
		void Solver<Dtype>::Test(const int test_net_id) {
			CHECK(Caffe::root_solver());
			LOG(INFO) << "Iteration " << iter_
				<< ", Testing net (#" << test_net_id << ")";
			CHECK_NOTNULL(test_nets_[test_net_id].get())->
				ShareTrainedLayersWith(net_.get());//先把权重共享过来，再进行测试
			vector<Dtype> test_score;
			vector<int> test_score_output_id;
			const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];//拿出测试网络
			Dtype loss = 0;
			for (int i = 0; i < param_.test_iter(test_net_id); ++i) {//测试的时候迭代这么多次
				SolverAction::Enum request = GetRequestedAction();
				// Check to see if stoppage of testing/training has been requested.
				while (request != SolverAction::NONE) {
					if (SolverAction::SNAPSHOT == request) {
						Snapshot();//测试的时候也保存快照
					} else if (SolverAction::STOP == request) {
						requested_early_exit_ = true;
					}
					request = GetRequestedAction();
				}
				if (requested_early_exit_) {
					// break out of test loop.
					break;
				}

				Dtype iter_loss;
				const vector<Blob<Dtype>*>& result =
					test_net->Forward(&iter_loss);//前向传播,返回了acc和loss等等
				if (param_.test_compute_loss()) {//计算loss
					loss += iter_loss;
				}
				if (i == 0) {
					for (int j = 0; j < result.size(); ++j) {
						const Dtype* result_vec = result[j]->cpu_data();
						for (int k = 0; k < result[j]->count(); ++k) {
							test_score.push_back(result_vec[k]);
							test_score_output_id.push_back(j);
						}
					}
				} else {
					int idx = 0;
					for (int j = 0; j < result.size(); ++j) {
						const Dtype* result_vec = result[j]->cpu_data();
						for (int k = 0; k < result[j]->count(); ++k) {
							test_score[idx++] += result_vec[k];//已经有对应关系了，上面
						}
					}
				}
			}
			if (requested_early_exit_) {
				LOG(INFO)     << "Test interrupted.";
				return;
			}
			if (param_.test_compute_loss()) {
				loss /= param_.test_iter(test_net_id);
				LOG(INFO) << "Test loss: " << loss;
			}
			for (int i = 0; i < test_score.size(); ++i) {//统计loss结果，输出
				const int output_blob_index =
					test_net->output_blob_indices()[test_score_output_id[i]];
				const string& output_name = test_net->blob_names()[output_blob_index];
				const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
				ostringstream loss_msg_stream;
				const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
				if (loss_weight) {
					loss_msg_stream << " (* " << loss_weight
						<< " = " << loss_weight * mean_score << " loss)";
				}
				LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
					<< mean_score << loss_msg_stream.str();
			}
		}

	template <typename Dtype>
		void Solver<Dtype>::Snapshot() {//保存一次快照
			CHECK(Caffe::root_solver());
			string model_filename;
			switch (param_.snapshot_format()) {
				case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
					model_filename = SnapshotToBinaryProto();
					break;
				case caffe::SolverParameter_SnapshotFormat_HDF5:
					model_filename = SnapshotToHDF5();
					break;
				default:
					LOG(FATAL) << "Unsupported snapshot format.";
			}

			SnapshotSolverState(model_filename);
		}

	template <typename Dtype>
		void Solver<Dtype>::CheckSnapshotWritePermissions() {//测试下写快照文件，是否有权限
			if (Caffe::root_solver() && param_.snapshot()) {
				CHECK(param_.has_snapshot_prefix())
					<< "In solver params, snapshot is specified but snapshot_prefix is not";
				string probe_filename = SnapshotFilename(".tempfile");
				std::ofstream probe_ofs(probe_filename.c_str());
				if (probe_ofs.good()) {
					probe_ofs.close();
					std::remove(probe_filename.c_str());
				} else {
					LOG(FATAL) << "Cannot write to snapshot prefix '"
						<< param_.snapshot_prefix() << "'.  Make sure "
						<< "that the directory exists and is writeable.";
				}
			}
		}

	template <typename Dtype>
		string Solver<Dtype>::SnapshotFilename(const string extension) {//返回快照的文件
			return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
				+ extension;
		}

	template <typename Dtype>
		string Solver<Dtype>::SnapshotToBinaryProto() {//快照转成pb
			string model_filename = SnapshotFilename(".caffemodel");
			LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
			NetParameter net_param;
			net_->ToProto(&net_param, param_.snapshot_diff());
			WriteProtoToBinaryFile(net_param, model_filename);
			return model_filename;
		}

	template <typename Dtype>
		string Solver<Dtype>::SnapshotToHDF5() {//快照转成hdf5
			string model_filename = SnapshotFilename(".caffemodel.h5");
			LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
			net_->ToHDF5(model_filename, param_.snapshot_diff());
			return model_filename;
		}

	template <typename Dtype>
		void Solver<Dtype>::Restore(const char* state_file) {//加载配置文件
			string state_filename(state_file);
			if (state_filename.size() >= 3 &&
					state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
				RestoreSolverStateFromHDF5(state_filename);
			} else {
				RestoreSolverStateFromBinaryProto(state_filename);
			}
		}

	template <typename Dtype>
		void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
				int average_loss) {//计算总体的loss函数
			if (losses_.size() < average_loss) {
				losses_.push_back(loss);
				int size = losses_.size();
				smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
			} else {
				int idx = (iter_ - start_iter) % average_loss;
				smoothed_loss_ += (loss - losses_[idx]) / average_loss;
				losses_[idx] = loss;
			}
		}

	INSTANTIATE_CLASS(Solver);

}  // namespace caffe
