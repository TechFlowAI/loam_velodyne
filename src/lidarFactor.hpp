#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>


struct LidarEdgeFactor
{
     LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, 
                         Eigen::Vector3d last_point_b_, double s_)
          : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}
     
     template<typename T>
     bool operator() (const T* q, const T* t, T* residual) const
     {
          // 将double数组转成eigen数据结构，注意这里一定要写成模板
          Eigen::Matrix<T,3,1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
          Eigen::Matrix<T,3,1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
          Eigen::Matrix<T,3,1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

          Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
          Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
          // 计算的是上一帧到当前帧的位姿变换，因此根据匀速模型，计算该点对应的位姿
          q_last_curr = q_identity.slerp(T(s), q_last_curr);
          Eigen::Matrix<T,3,1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

          Eigen::Matrix<T, 3, 1>  lp;
          // 把当前点根据当前计算的帧间位姿变换到上一帧
          lp = q_last_curr * cp + t_last_curr;

          Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
          Eigen::Matrix<T, 3, 1> de = lpa - lpb;
          // 残差的模是该点到底边的垂线长度
          residual[0] = nu.x() / de.norm();
          residual[1] = nu.y() / de.norm();
          residual[2] = nu.z() / de.norm();
          return true;
     }

     static ceres::CostFunction* Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
                                        const Eigen::Vector3d last_point_b_, const double s_)
     {
          // arg1:残差的维度3 arg2:参数q的维度 arg3:参数t的维度
          return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3>(
               new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
     }


     Eigen::Vector3d curr_point, last_point_a, last_point_b;
     double s;
};


struct LidarPlaneFactor
{
     LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_, Eigen::Vector3d last_point_l_,
                         Eigen::Vector3d last_point_m_, double s_)
          : curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_), 
              last_point_m(last_point_m_), s(s_)
     {
          ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
          ljm_norm.normalize();
     }

     template<typename T>
     bool operator() (const T* q, const T* t, T* residual) const
     {
          Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
          Eigen::Matrix<T, 3, 1> lpj(T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z()));
          Eigen::Matrix<T, 3, 1> ljm(T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z()));

          Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
          Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
          q_last_curr = q_identity.slerp(T(s), q_last_curr);
          Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};
          
          Eigen::Matrix<T, 3, 1> lp;
          lp = q_last_curr * cp + t_last_curr;

          residual[0] = (lp - lpj).dot(ljm_norm);
          return true;

     }

     static ceres::CostFunction* Crete(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_, const Eigen::Vector3d last_point_l_,
                         const Eigen::Vector3d last_point_m_, const double s_)
     {
          return (new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(
               new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
     }
     Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
     double s;
     Eigen::Vector3d ljm_norm;
};

struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);	// 求解点到平面的距离
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};
