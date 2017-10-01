#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

	/**
	 * @brief Enumeration of actions that a client of the Solver may request by
	 * implementing the Solver's action request function, which a
	 * client may optionally provide in order to request early termination
	 * or saving a snapshot without exiting. In the executable caffe, this
	 * mechanism is used to allow the snapshot to be saved when stopping
	 * execution with a SIGINT (Ctrl-C).
	 */
	namespace SolverAction {
		enum Enum {
			NONE = 0,  // Take no special action.//没有特殊支持快照
			STOP = 1,  // Stop training. snapshot_after_train controls whether a//保存一次快照就直接退出
			// snapshot is created.
			SNAPSHOT = 2  // Take a snapshot, and keep training.//支持快照保存模型的方式
		};
	}

	/**
	 * @brief Type of a function that returns a Solver Action enumeration.
	 */
	typedef boost::function<SolverAction::Enum()> ActionCallback;

	/**
	 * @brief An interface for classes that perform optimization on Net%s.
	 *
	 * Requires implementation of ApplyUpdate to compute a parameter update
	 * given the current state of the Net parameters.
	 */
	template <typename Dtype>
		class Solver {
			public:
				explicit Solver(const SolverParameter& param);//用solver配置初始化一个solver
				explicit Solver(const string& param_file);//用solver配置文件初始化一个solver
				void Init(const SolverParameter& param);
				void InitTrainNet();//初始化训练的solver
				void InitTestNets();//初始化测试的solver

				// Client of the Solver optionally may call this in order to set the function
				// that the solver uses to see what action it should take (e.g. snapshot or
				// exit training early).
				void SetActionFunction(ActionCallback func);
				SolverAction::Enum GetRequestedAction();
				// The main entry of the solver function. In default, iter will be zero. Pass
				// in a non-zero iter number to resume training for a pre-trained net.
				virtual void Solve(const char* resume_file = NULL);//训练某个网络结构
				inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }//训练某个网络结构
				void Step(int iters);//训练iters次
				// The Restore method simply dispatches to one of the
				// RestoreSolverStateFrom___ protected methods. You should implement these
				// methods to restore the state from the appropriate snapshot type.
				void Restore(const char* resume_file);
				// The Solver::Snapshot function implements the basic snapshotting utility
				// that stores the learned net. You should implement the SnapshotSolverState()
				// function that produces a SolverState protocol buffer that needs to be
				// written to disk together with the learned net.
				void Snapshot();
				virtual ~Solver() {}
				inline const SolverParameter& param() const { return param_; }
				inline shared_ptr<Net<Dtype> > net() { return net_; }
				inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
					return test_nets_;
				}
				int iter() const { return iter_; }

				// Invoked at specific points during an iteration
				class Callback {
					protected:
						virtual void on_start() = 0;
						virtual void on_gradients_ready() = 0;

						template <typename T>
							friend class Solver;
				};
				const vector<Callback*>& callbacks() const { return callbacks_; }
				void add_callback(Callback* value) {
					callbacks_.push_back(value);
				}

				void CheckSnapshotWritePermissions();//check，模型目录是否有写权限
				/**
				 * @brief Returns the solver type.
				 */
				virtual inline const char* type() const { return ""; }

			protected:
				// Make and apply the update value for the current iteration.
				virtual void ApplyUpdate() = 0;//更新矩阵
				string SnapshotFilename(const string extension);
				string SnapshotToBinaryProto();
				string SnapshotToHDF5();
				// The test routine
				void TestAll();//验证所有验证网络结构
				void Test(const int test_net_id = 0);//验证某个网络
				virtual void SnapshotSolverState(const string& model_filename) = 0;
				virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
				virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
				void DisplayOutputBlobs(const int net_id);//输出某个验证网络的输出，loss or acc
				void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);//更行损失函数

				SolverParameter param_;//solver的网络配置
				int iter_;//迭代次数
				int current_step_;//没用的参数
				shared_ptr<Net<Dtype> > net_;//训练网络结构
				vector<shared_ptr<Net<Dtype> > > test_nets_;//多个验证网络结构
				vector<Callback*> callbacks_;//callback回调函数
				vector<Dtype> losses_;//辅助计算损失函数
				Dtype smoothed_loss_;//计算总的损失函数

				// A function that can be set by a client of the Solver to provide indication
				// that it wants a snapshot saved and/or to exit early.
				ActionCallback action_request_function_;//一种callback

				// True iff a request to stop early was received.
				bool requested_early_exit_;//支持从外部停止调用停止训练

				// Timing information, handy to tune e.g. nbr of GPUs
				Timer iteration_timer_;//计时器
				float iterations_last_;//记录上一次展示的迭代次数

				DISABLE_COPY_AND_ASSIGN(Solver);
		};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
