#include <math.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "loam_velodyne/common.h"
#include "loam_velodyne/tic_toc.h"

/*
     地图保存为栅格地图，每个栅格长50m，栅格的长宽高分别为21*21*11，也就是说保存的地图大小为1050m*1050m*550m
*/

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

int frameCount = 0;

// 起始偏移量，后面会一直变化
int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;

// 地图栅格的大小
constexpr int laserCloudWidth = 21;
constexpr int laserCloudHeight = 21;
constexpr int laserCloudDepth = 11;
constexpr int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; // 4851

int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

Eigen::Quaterniond q_wmap_wodom(1, 0, 0 ,0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);


pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

// points in every cube
// 分别保存角点和面点地图 每个地图用数组表示，数组的大小就是所有cube的总和
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

// ouput: all visualble cube points
// 存储当前帧附近的所有点云
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
nav_msgs::Path laserAfterMappedPath;

// set initial guess
void transformAssociateToMap()
{
     q_w_curr = q_wmap_wodom * q_wodom_curr;
     t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

// 根据odomtry提供的初始位姿将点转到地图坐标系下
void pointAssociateToMap(PointType const* const pi, PointType* const po)
{
     Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
     Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
     po->x = point_w.x();
     po->y = point_w.y();
     po->z = point_w.z();
     po->intensity = pi->intensity;
}

// 更新odom到map之间的位姿变换
void transformUpdate()
{
     q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
     t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudCornerLast2)
{
     mBuf.lock();
     cornerLastBuf.push(laserCloudCornerLast2);
     mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish 以里程计的频率向外发送位姿
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

// 主处理线程
void process()
{
     while(1)
     {
          while(!cornerLastBuf.empty() && surfLastBuf.empty() && 
                    !fullResBuf.empty() && !odometryBuf.empty())
          {
               mBuf.lock();
               // 以cornerLastBuf作为基准，把时间戳小于其的全部pop出去
               /*
			这样做的必要性(好处):
				1.保证实时性。因为前端处理速度快，可能1s发送一个数据给后端，但是后端的处理速度可能是10s处理一个数据。
                    如果遇到这种情况，做法就是将处理一次后的cornerLastBuf清空(187-191行)，比如当前cornerLastBuf里面
                    有十个数据，是前端从第1s到第10s发送过来的数据，此时由于后端处理太慢，这时才开始处理第1s数据，把第1s数据
                    拿到之后，直接清空，则下一次在处理的数据就是第11s的，如果不这么做，会出现，前端都发送到20s数据，结果才处理
                    第2s的，做这个操作之后处理的是11s的，提高了算法的实时性
				2.防止内存爆了。因为后端处理很慢，如果不及时清理那么四个Buf的内存都会爆炸。因此，在处理
                    cornerLastBuf第一个数据的时候，把其他三个Buf在cornerLastBuf之前的数据全部清空
			*/
               while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                    odometryBuf.pop();
               if (odometryBuf.empty())
               {
                    mBuf.unlock();
                    break;
               }
               
               while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                    surfLastBuf.pop();
               if (surfLastBuf.empty())
               {
                    mBuf.unlock();
                    break;
               }

               while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
                    fullResBuf.pop();
               if (fullResBuf.empty())
               {
                    mBuf.unlock();
                    break;
               }
               timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
               timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
               timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
               timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
               // 原则上取出来的时间戳都是一样的，如果不一样说明有问题
               if (timeLaserCloudCornerLast != timeLaserOdometry || 
                   timeLaserCloudSurfLast != timeLaserOdometry ||
                   timeLaserCloudFullRes != timeLaserOdometry)
               {
                    ROS_WARN("unsync message!");
                    mBuf.unlock();
                    break;
               }
               // 点云全部转成pcl的数据格式
               laserCloudCornerLast->clear();
               pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
               cornerLastBuf.pop();

               laserCloudSurfLast->clear();
               pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
               surfLastBuf.pop();

               laserCloudFullRes->clear();
               pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
               fullResBuf.pop();
               
               // lidar odom的结果转成eigen数据格式
               q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
               q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
               q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
               q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.w;
               t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
               t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
               t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
               odometryBuf.pop();
               // 考虑到实时性，就把队列里其他的都pop出去，不然可能出现处理延时的情况
               while (!cornerLastBuf.empty())
               {
                    cornerLastBuf.pop();
                    std::cout << "drop lidar frame in mapping for real time performance\n";
               }
               mBuf.unlock();
               
               TicToc t_whole;
               // 根据前端结果，得到后端的一个初始估计值
               transformAssociateToMap();
               
               TicToc t_shift;
               // 根据里程计提供的初始值计算当前位姿在地图中的索引，一个cube为边长为50m的立方体
               // 后端的地图本质上一个以当前点为中心，一个栅格地图
               // 加25的目的是四舍五入
               int centerCubeI = int((t_w_curr.x() + 25) / 50.0) + laserCloudCenWidth;
               int centerCubeJ = int((t_w_curr.y() + 25) / 50.0) + laserCloudCenHeight;
               int centerCubeK = int((t_w_curr.z() + 25) / 50.0) + laserCloudCenDepth;

               // 这里是为了消除由于c++去整导致的一些与实际不符合的情况 c++是向0取整 
               // -1.6 --> -1，这种情况应该为-2更加合理
               if (t_w_curr.x() + 25 < 0) centerCubeI--;
               if (t_w_curr.y() + 25 < 0) centerCubeJ--;
               if (t_w_curr.z() + 25 < 0) centerCubeK--;
               // 如果当前栅格索引小于3,就说明当前点快接近地图边界，需要进行调整，相当于地图整体往x正方形移动
               while (centerCubeI < 3)
               {
                    for (int j = 0; j < laserCloudCenHeight; ++j)
                    {
                         for (int k = 0; k < laserCloudCenDepth; ++k)
                         {
                              int i = laserCloudCenWidth - 1;
                              // 从x最大值开始
                              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer = 
                                   laserCloudCornerArray[i + laserCloudCenWidth * j + laserCloudCenWidth * laserCloudCenHeight * k];
                              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer = 
                                   laserCloudSurfArray[i + laserCloudCenWidth * j + laserCloudCenWidth * laserCloudCenHeight * k];
                              // 整体右移
                              for (; i >= 1; --i)
                              {
                                   laserCloudCornerArray[i + laserCloudCenWidth * j + laserCloudCenWidth * laserCloudCenHeight * k] =
                                        laserCloudCornerArray[i - 1 + laserCloudCenWidth * j + laserCloudCenWidth * laserCloudCenHeight * k];
                                   laserCloudSurfArray[i + laserCloudCenWidth * j + laserCloudCenWidth * laserCloudCenHeight * k] = 
                                        laserCloudSurfArray[i - 1 + laserCloudCenWidth * j + laserCloudCenWidth * laserCloudCenHeight * k];
                              }
                              // 此时 i = 0, 也就是最左边的格子赋值了之前最右边的格子
                              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
                              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
                              laserCloudCubeCornerPointer->clear();
                              laserCloudCubeSurfPointer->clear();
                         }
                    }
                    centerCubeI++;
                    laserCloudCenWidth++;
               }
               // 同理x抵达有边界，就整体左移
               while (centerCubeI >= laserCloudCenWidth - 3)
               {
                    for (int j = 0; j < laserCloudCenHeight; ++j)
                    {
                         for (int k = 0; k < laserCloudCenDepth; ++k)
                         {
                              int i = 0;
                              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer = 
                                   laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                              // 整体左移
                              for (; i < laserCloudCenWidth - 1; ++i)
                              {
                                   laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                              }
                              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
                              laserCloudCubeCornerPointer->clear();
                              laserCloudCubeSurfPointer->clear();
                         }
                    }
                    centerCubeI--;
                    laserCloudCenWidth--;
               }
               // y和z方向同理
               while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}
               while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}


			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}
               // 以上操作相当于维护了一个局部地图，保证当前帧不在局部地图的边缘，这样才可以从地图中获取足够的约束
               int laserCloudVaildNum = 0;
               int laserCloudSurroundNum = 0;
               // 从当前格子为中心，选出一定范围的点云
               for (int i = centerCubeI - 2; i <= centerCubeI + 2; ++i)
               {
                    for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; ++j)
                    {
                         for (int k = centerCubeK - 2; k <= centerCubeK + 2; ++k)
                         {
                              if (i >= 0 && i < laserCloudWidth && 
                                  j >= 0 && j < laserCloudHeight &&
                                  k >= 0 && k < laserCloudDepth)
                              {
                                   // 把各自的索引记录下来
                                   laserCloudValidInd[laserCloudVaildNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                                   laserCloudVaildNum++;
                                   laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
                              }
                         }
                    }
               }
               laserCloudCornerFromMap->clear();
               laserCloudSurfFromMap->clear();
               // 开始构建用来这一帧优化的小的局部地图
               for (int i = 0; i < laserCloudVaildNum; ++i)
               {
                    *laserCloudCornerFromMap = *laserCloudCornerArray[laserCloudValidInd[i]];
                    *laserCloudSurfFromMap = *laserCloudSurfArray[laserCloudValidInd[i]];
               }
               int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
               int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

               // 为了减少计算量，使用体素滤波对点云(当前帧)进行下采样
               pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
               downSizeFilterCorner.setInputCloud(laserCloudSurfLast);
               downSizeFilterCorner.filter(*laserCloudCornerStack);
               int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

               pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			downSizeFilterSurf.filter(*laserCloudSurfStack);
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

               std::cout << "map prepare time: " << t_shift.toc() << std::endl;
               std::cout << "map corner nums: " << laserCloudCornerFromMapNum << std::endl;
               std::cout << "map surf nums: " << laserCloudSurfFromMapNum << std::endl;
               
               // 最终的有效点云数目进行判断
               if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 10)
               {
                    TicToc t_opt;
                    TicToc t_tree;
                    // 送入kdtree便于最近邻搜索
                    kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
                    kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
                    std::cout << "build tree time: " << t_tree.toc() << "ms" << std::endl;
                    // 建立对应关系的迭代次数不超过2次
                    for (int iterCount = 0; iterCount < 2; ++iterCount)
                    {
                         // 建立ceres问题
                         ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
                         ceres::LocalParameterization* q_parameterization = new ceres::EigenQuaternionParameterization();
                         ceres::Problem::Options problem_options;
                         ceres::Problem problem(problem_options);
                         problem.AddParameterBlock(parameters, 4, q_parameterization);
                         problem.AddParameterBlock(parameters + 4, 3);

                         TicToc t_data;
                         int corner_num = 0;
                         // 构建角点相关的约束
                         for (int i = 0; i < laserCloudCornerStackNum; ++i)
                         {
                              pointOri = laserCloudCornerStack->points[i];
                              // 把当前点根据初始值投到地图坐标系下
                              pointAssociateToMap(&pointOri, &pointSel);
                              // 地图中寻找和该点最近的5个点
                              kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                              // 判断最远的点距离不能超过1m，否则就是无效点
                              if (pointSearchSqDis[4] < 1.0)
                              {
                                   std::vector<Eigen::Vector3d> nearCorners;
                                   Eigen::Vector3d center(0, 0 ,0);
                                   for (int j = 0; j < 5; ++j)
                                   {
                                        Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                                                            laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                                                            laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                                        center += tmp;
                                        nearCorners.push_back(tmp);
                                   }
                                   // 计算这5个点的均值
                                   center = center / 5.0;
                                   
                                   Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                                   // 构建协方差矩阵
                                   for (int j = 0; j < 5; ++j)
                                   {
                                        Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                        covMat += tmpZeroMean * tmpZeroMean.transpose();
                                   }
                                   // 进行特征值分解
                                   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
                                   // 根据特征值分解判断是否为线特征：满足特征值两小一大
                                   // 最大特征值对应的特征向量为线特征方向
                                   Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                                   Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                                   // 判断方式：最大特征值大于次大特征值的3倍
                                   if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                                   {
                                        Eigen::Vector3d point_on_line = center;
                                        Eigen::Vector3d point_a, point_b;
                                        // 根据拟合出来的线特征方向，以均值点为中心构建两个虚拟点
                                        point_a = 0.1 * unit_direction + point_on_line;
                                        point_b = -0.1 * unit_direction + point_on_line;
                                        // 构建约束，和lidar odometry约束一致
                                        // 这里s=1的原因是：传入的q为q_w_curr
                                        ceres::CostFunction* cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                                        problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                                        corner_num++;
                                   }
                              }
                         }

                         int surf_num = 0;
                         // 构建面点约束
                         for (int i = 0; i < laserCloudSurfStackNum; ++i)
                         {
                              pointOri = laserCloudSurfStack->points[i];
                              pointAssociateToMap(&pointOri, &pointSel);
                              kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                              Eigen::Matrix<double, 5, 3> matA0;
                              Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                              // 构建平面方程Ax + By +Cz + 1 = 0
						// 通过构建一个超定方程来求解这个平面方程
                              if (pointSearchSqDis[4] < 1.0)
                              {
                                   for (int j = 0; j < 5; ++j)
                                   {
                                        matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchSqDis[j]].x;
                                        matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchSqDis[j]].y;
                                        matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchSqDis[j]].z;
                                   }
                                   // 调用eigen接口求解该方程，解就是这个平面的法向量
                                   Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                                   double negative_OA_dot_norm = 1 / norm.norm();
                                   // 法向量归一化
                                   norm.normalize();

                                   bool planeValid = true;
                                   for (int j = 0; j < 5; ++j)
                                   {
                                        // 这里相当于求解点到平面的距离
								if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
										 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
										 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
								{
									planeValid = false;	// 点如果距离平面太远，就认为这是一个拟合的不好的平面
									break;
								}
                                   }
                                   Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                                   // 如果平面有效就构建平面约束
                                   if (planeValid)
                                   {
                                        ceres::CostFunction* cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                                        problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                                        surf_num++;
                                   }
                              }
                         }
                         std::cout << "mapping data assosiation time: " << t_data.toc() << std::endl;
                         // 调用ceres求解
                         TicToc t_solver;
                         ceres::Solver::Options options;
                         options.linear_solver_type = ceres::DENSE_QR;
                         options.max_num_iterations = 4;
                         options.minimizer_progress_to_stdout = false;
                         options.check_gradients = false;
                         options.gradient_check_relative_precision = 1e-4;
                         ceres::Solver::Summary summary;
                         ceres::Solve(options, &problem, &summary);
                         std::cout << "mapping solver time: " << t_solver.toc() << std::endl;
                    }
                    std::cout << "mapping optimization time: " << t_opt.toc() << std::endl;
               }
               else
               {
                    ROS_WARN("map corner and surf num are not enough");
               }     
               // 更新最新scan to map得出来的位姿    
               transformUpdate();

               TicToc t_add;
               // 将优化后的当前帧角点加到局部地图中去
               for (int i = 0; i < laserCloudCornerStackNum; ++i)
               {
                    // 该点根据最新算出来的位姿投到地图坐标系
                    pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
                    // 算出这个点所在的cube的索引
                    int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;
				// 同样四舍五入一下
				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;
                    // 如果超过边界的话就算了
				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
                    {
                         // 根据xyz的索引计算出在存储地图数组中的位置
                         int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                         laserCloudCornerArray[cubeInd]->push_back(pointSel);
                    }
               }
               // 面点也做同样的处理
			for (int i = 0; i < laserCloudSurfStackNum; i++)
			{
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
               std::cout << "add points time: " << t_add.toc() << std::endl;

               TicToc t_filter;
               // 把当前涉及到的局部地图的栅格做一个下采样
               for (int i = 0; i < laserCloudVaildNum; ++i)
               {
                    int ind = laserCloudValidInd[i];
                    pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
                    downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
                    downSizeFilterCorner.filter(*tmpCorner);
                    laserCloudCornerArray[ind] = tmpCorner;

                    pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
               }
               std::cout << "filter time: " << t_filter.toc() << std::endl;

               //publish surround map for every 5 frame
			// 每隔5帧对外发布一下
               if (frameCount % 5 == 0)
               {
                    laserCloudSurround->clear();
				// 把该当前帧相关的局部地图发布出去
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
               }
               // 每隔20帧发布全量的局部地图
			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				// 21 × 21 × 11 = 4851
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

               int laserCloudFullResNum = laserCloudFullRes->points.size();
			// 把当前帧发布出去
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

               sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

               std::cout << "whole mapping time: " << t_whole.toc() << std::endl;
               // 发布当前的位姿
               nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);
			// 发布当前轨迹
			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);
               // 发布tf
               static tf::TransformBroadcaster br;
               tf::Transform transform;
               tf::Quaternion q;
               transform.setOrigin(tf::Vector3(t_w_curr(0), t_w_curr(1), t_w_curr(2)));
               q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
               br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "aft_mapped"));

               frameCount++;
          }
          std::chrono::milliseconds dura(2);
          std::this_thread::sleep_for(dura);
     }
}

int main(int argc, char* argv[])
{
     ros::init(argc, argv, "laserMapping");
     ros::NodeHandle nh;
     float lineRes = 0;
     float planeRes = 0;
     nh.param<float>("mapping_line_resolution", lineRes, 0.4);
     nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
     std::cout << "line resolution is " << lineRes << ", plane resolution is " << planeRes << std::endl;
     downSizeFilterCorner.setLeafSize(lineRes, lineRes, lineRes);
     downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);

     // 订阅点云于起始位恣
     ros::Subscriber subLaserCloudCornerLast = nh.subscribe("laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

     ros::Subscriber subLaserCloudSurfLast = nh.subscribe("laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

     ros::Subscriber subLaserCloudFullRes = nh.subscribe("velodyne_cloud_3", 100, laserCloudFullResHandler);

     ros::Subscriber subLaserOdometry = nh.subscribe("/laser_odom_to_init", 100, laserOdometryHandler);

     pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
     
     pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
     
     pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

     for (int i = 0; i < laserCloudNum; ++i)
     {
          laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
          laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
     }
     std::thread mapping_process(process);
     ros::spin();
     return 0;
}