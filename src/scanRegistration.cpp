#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include "loam_velodyne/tic_toc.h"
#include "loam_velodyne/common.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>


int N_SCANS = 0;
double MINIMUN_RANGE = 0.1;
bool systemInited = false;
int systemInitCount = 0;
constexpr int systemDelay = 0;
constexpr double scanPeriod = 0.1; //0.1s == 100ms == 10hz
int cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i, int j) { return cloudCurvature[i] < cloudCurvature[j]; }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;

template <typename PointT>
void removeClosePointCloud(const pcl::PointCloud<PointT>& cloud_in,
                              pcl::PointCloud<PointT>& cloud_out, float thres)
{
     if (&cloud_in != &cloud_out)
     {
          cloud_out.header = cloud_in.header;
          cloud_out.points.resize(cloud_in.points.size());
     }
     size_t j = 0;
     for (size_t i = 0; i < cloud_in.size(); ++i)
     {
          if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
               continue;
          cloud_out.points[j] = cloud_in.points[i];
          j++;
     }
     if (j != cloud_in.points.size()) cloud_out.points.resize(j);
     cloud_out.height = 1;
     cloud_out.width = static_cast<uint32_t>(j);
     cloud_out.is_dense = true;
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
     //如果系统没有初始化，就等几帧
     if (!systemInited)
     {
          systemInitCount++;
          if (systemInitCount >= systemDelay) systemInited = true;
          else return;
     }

     TicToc t_whole;
     TicToc t_prepare;
     std::vector<int> scanStartInd(N_SCANS, 0);
     std::vector<int> scanEndInd(N_SCANS, 0);

     pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
     // 把点云从ros格式转到pcl格式
     pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
     std::vector<int> indices;
     // 去除点云中的nan点
     pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
     // 去除距离小于阈值的点
     removeClosePointCloud(laserCloudIn, laserCloudIn, MINIMUN_RANGE);
     
     // 计算起始点和结束点的角度，由于激光雷达是顺时针旋转，这里取反相当于转成逆时针
     int cloudSize = laserCloudIn.points.size();
     float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
     // atan2的范围是（-PI，PI]，这里加上2PI是为了保证起始和结束相差2PI 符合实际
     float endOri = -atan2(laserCloudIn.points[cloudSize-1].y, 
                              laserCloudIn.points[cloudSize-1].x) + 2 * M_PI;
     
     // 总有一些例外，比如这里大于3PI，和小于PI，就需要做一些调整到合理范围
     if (endOri - startOri > 3 * M_PI) endOri -= 2 * M_PI;
     else if (endOri - startOri < M_PI) endOri += 2 * M_PI;

     bool halfPassed = false;
     int count = cloudSize;
     PointType point;
     std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
     // 计算每一个点
     for (int i = 0; i < cloudSize; ++i)
     {
          point.x = laserCloudIn.points[i].x;
          point.y = laserCloudIn.points[i].y;
          point.z = laserCloudIn.points[i].z;
          // 计算俯仰角
          float angle = atan(point.z / sqrt(point.x*point.x + point.y*point.y));
          int scanID = 0;
          // 计算是第几根线
          if (N_SCANS == 16)
          {
               scanID = int((angle + 15)/2 + 0.5); // 0.5是为了防止四舍五入 15线垂直视野-15-15度，每两根线的分辨率是2度
               if (scanID > (N_SCANS - 1) || scanID < 0)
               {
                    count--;
                    continue;
               }
          }
          else if (N_SCANS == 32)
          {
               scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
               if (scanID > (N_SCANS - 1) || scanID < 0)
               {
                    count--;
                    continue;
               }  
          }
          else if (N_SCANS == 64)
          {   
               if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
               else
                    scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

               // use [0 50]  > 50 remove outlies 
               if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
               {
                    count--;
                    continue;
               }
          }
          else
          {
               printf("wrong scan number\n");
               ROS_BREAK();
          }

          // 计算水平角
          float ori = -atan2(point.y, point.x);
          if (!halfPassed)
          {
               // 确保-PI / 2 < ori - startOri < 3 / 2 * PI
               if (ori < startOri - M_PI / 2) ori += 2 * M_PI;
               else if (ori > startOri + M_PI * 3 / 2) ori -= 2 * M_PI;
               if (ori - startOri > M_PI) halfPassed = true;
          }
          else
          {
               // 确保-PI * 3 / 2 < ori - endOri < PI / 2
               ori += 2 * M_PI;
               if (ori < endOri - M_PI * 3 / 2) ori += 2 * M_PI;
               else if (ori > endOri + M_PI / 2) ori -= 2 * M_PI;
          }
          // 角度的计算是为了计算相对的起始时刻的时间
          float relTime = (ori - startOri) / (endOri - startOri);
          point.intensity = scanID + relTime * scanPeriod;
          laserCloudScans[scanID].emplace_back(point);
     }
     // cloudsSize是最终有效点云的数目
     cloudSize = count;
     std::cout << "points size: " << cloudSize << std::endl;

     pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
     // 全部集合到一个点云里面，但是使用两个数据标记每根线的起始和终止，+5,-6为了方便计算曲率
     for (int i = 0; i < N_SCANS; ++i)
     {
          scanStartInd[i] = laserCloud->size() + 5;
          *laserCloud += laserCloudScans[i];
          scanEndInd[i] = laserCloud->size() - 6;
     }
     std::cout << "prepare time: " << t_prepare.toc() << std::endl;
     
     // 开始计算曲率
     for (int i = 5; i < cloudSize - 5; ++i)
     {
          float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
          float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
          float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
          // 存储曲率，索引
          cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
          cloudSortInd[i] = i;
          cloudNeighborPicked[i] = 0;
          cloudLabel[i] = 0;
     }
     TicToc t_pts;
     pcl::PointCloud<PointType> cornerPointsSharp;
     pcl::PointCloud<PointType> cornerPointsLessSharp;
     pcl::PointCloud<PointType> surfPointsFlat;
     pcl::PointCloud<PointType> surfPointsLessFlat;


     // 剔除异常点
     for (int i = 5; i < cloudSize - 6; ++i)
     {
          float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + laserCloud->points[i].y * laserCloud->points[i].y + laserCloud->points[i].z * laserCloud->points[i].z);
          float depth2 = sqrt(laserCloud->points[i+1].x * laserCloud->points[i+1].x + laserCloud->points[i+1].y * laserCloud->points[i+1].y + laserCloud->points[i+1].z * laserCloud->points[i+1].z);
          float diffX = laserCloud->points[i+1].x - laserCloud->points[i].x;
          float diffY = laserCloud->points[i+1].y - laserCloud->points[i].y;
          float diffZ = laserCloud->points[i+1].z - laserCloud->points[i].z;
          if (diffX*diffX + diffY*diffY + diffZ*diffZ < 0.05)
          {
               // 遮挡的情况
               if (depth1 - depth2 > 0.3)
               {
                    cloudNeighborPicked[i-1] = 1;
                    cloudNeighborPicked[i-2] = 1;
                    cloudNeighborPicked[i-3] = 1;
                    cloudNeighborPicked[i-4] = 1;
                    cloudNeighborPicked[i-5] = 1;
                    cloudNeighborPicked[i] = 1;
               }
               else if (depth2 - depth1 > 0.3)
               {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
               }

               // 平行的情况
               float depth3 = sqrt(laserCloud->points[i-1].x * laserCloud->points[i-1].x + laserCloud->points[i-1].y * laserCloud->points[i-1].y + laserCloud->points[i-1].z * laserCloud->points[i-1].z);
               float diff1 = std::abs(depth1 - depth2);
               float diff2 = std::abs(depth1 - depth3);
               if (diff1 > 0.02 * depth1 && diff2 > 0.02 * depth1) cloudNeighborPicked[i] = 1;
          }
     }


     float t_q_sort = 0;
     // 遍历每一个scan
     for (int i = 0; i < N_SCANS; ++i)
     {
          // 如果当前scan的点少于6个点，视为没有有效点
          if (scanEndInd[i] - scanStartInd[i] < 6) continue;
          // 用来存储不太平整的点
          pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>());
          // 将每个scan分为6等分
          for (int j = 0; j < 6; ++j)
          {
               // 每个等分的起始和结束点
               int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
               int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

               TicToc t_tmp;
               // 对点云按照曲率进行排序，小的在前，大的在后
               std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);
               t_q_sort += t_tmp.toc();

               int largestPickedNum = 0;
               // 挑选曲率比较大的点
               for (int k = ep; k >= sp; --k)
               {
                    // 排序后顺序就乱了，这个时候索引的作用就得到提现了，能够找到原来所对应的点
                    int ind = cloudSortInd[k];

                    // 看看这个点是否是有效点，同时曲率是否大雨阈值
                    // cloudNeighborPicked用来记录这个点是否为有效点，如果为1代表这个点或者
                    // 这个点周围的某个点已经被选为特征点，即无效，这是为了使特征点均匀化，同时，
                    // 阈值的目的是为了这些点满足为边缘特征的基本条件，排除一种极端情况：即使曲率
                    // 排序后最大，但是仍然是平面点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1)
                    {
                         largestPickedNum++;
                         // 每段选两个曲率大的点
                         if (largestPickedNum <= 2)
                         {
                              // label=2表示曲率大的点
                              cloudLabel[ind] = 2;
                              cornerPointsSharp.push_back(laserCloud->points[ind]);
                              cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                         }
                         // 并选取曲率稍微大的标记
                         else if (largestPickedNum <= 20)
                         {
                              // label = 1 表示曲率稍微大
                              cloudLabel[ind] = 1;
                              cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                         }
                         else
                         {
                              break;
                         }
                         // 这些点被选中后，pick标志位置变为1
                         cloudNeighborPicked[ind] = 1;
                         // 为了保证特征点不过于集中，将选中的点周围5个点都标志位1,避免后续被选中
                         for (int l = 1; l <= 5; ++l)
                         {
                              // 查看相邻点距离是否差异过大，如果差异过大说明点云在此处不连续（在此之前已经做过
                              // 去除nan点操作，从而可能会发生两个相邻的点中间剔除了一些点导致距离很大），
                              // 是特征边缘，就会是新的特征（虽然两个点相邻，但是属于特征边缘），因此就不置位了
                              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l -1].x;
                              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l -1].y;
                              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l -1].z;
                              if (diffX*diffX + diffY*diffY + diffZ*diffZ > 0.05) break;
                              cloudNeighborPicked[ind + l] = 1;
                         }
                         for (int l = -1; l >= -5; --l)
                         {
                              // 查看相邻点距离是否差异过大，如果差异过大说明点云在此处不连续（在此之前已经做过
                              // 去除nan点操作，从而可能会发生两个相邻的点中间剔除了一些点导致距离很大），
                              // 是特征边缘，就会是新的特征（虽然两个点相邻，但是属于特征边缘），因此就不置位了
                              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                              if (diffX*diffX + diffY*diffY + diffZ*diffZ > 0.05) break;
                              cloudNeighborPicked[ind + l] = 1;
                         }
                    }
               }
               // 下面挑选面点
               int smallestPickedNum = 0;
               for (int k = sp; k <= ep; ++k)
               {
                    int ind = cloudSortInd[k];
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1)
                    {
                         smallestPickedNum++;
                         // label = -1表示平坦的点
                         cloudLabel[ind] = -1;
                         surfPointsFlat.push_back(laserCloud->points[ind]);
                         // 这里不需要区分平坦和比较平坦，因为剩下的label=0都表示比较平坦的点
                         if (smallestPickedNum >= 4) break;
                         cloudNeighborPicked[ind] = 1;
                         for (int l = 1; l <= 5; ++l)
                         {
                              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l -1].x;
                              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l -1].y;
                              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l -1].z;
                              if (diffX*diffX + diffY*diffY + diffZ*diffZ > 0.05) break;
                              cloudNeighborPicked[ind + l] = 1;
                         }
                         for (int l = -1; l >= -5; --l)
                         {
                              float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                              float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                              float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                              if (diffX*diffX + diffY*diffY + diffZ*diffZ > 0.05) break;
                              cloudNeighborPicked[ind + l] = 1;
                         }
                    }
               }
               for (int k = sp; k <= ep; ++k)
               {
                    if (cloudLabel[k] <= 0)
                    {
                         surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                    }
               }
          }
          pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
          pcl::VoxelGrid<PointType> downSizeFilter;
          // 一般平坦的点很多，做一个体素滤波降采样
          downSizeFilter.setInputCloud(surfPointsLessFlatScan);
          downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
          downSizeFilter.filter(surfPointsLessFlatScanDS);

          surfPointsLessFlat += surfPointsLessFlatScanDS;
     }
     std::cout << "sort q time: " << t_q_sort << std::endl;
     std::cout << "seperate points time: " << t_pts.toc() << std::endl;

     // 分别将当前点云，四种特征的点云发布出去
     sensor_msgs::PointCloud2 laserCloudOutMsg;
     pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
     laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
     laserCloudOutMsg.header.frame_id = "/camera_init";
     pubLaserCloud.publish(laserCloudOutMsg);

     sensor_msgs::PointCloud2 cornerPointsSharpMsg;
     pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
     cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
     cornerPointsSharpMsg.header.frame_id = "/camera_init";
     pubCornerPointsSharp.publish(cornerPointsSharpMsg);

     sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
     pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
     cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
     cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
     pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

     sensor_msgs::PointCloud2 surfPointsFlat2;
     pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
     surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
     surfPointsFlat2.header.frame_id = "/camera_init";
     pubSurfPointsFlat.publish(surfPointsFlat2);

     sensor_msgs::PointCloud2 surfPointsLessFlat2;
     pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
     surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
     surfPointsLessFlat2.header.frame_id = "/camera_init";
     pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

     std::cout << "scan registration time: " << t_whole.toc() << " ms" << std::endl;
     if (t_whole.toc() > 100)
     {
          ROS_WARN("scan registration process over 100ms");
     }

}

int main(int argc, char* argv[])
{
     ros::init(argc, argv, "scanRegistration");
     ros::NodeHandle nh;
     // 从配置文件中获取多少线的激光雷达
     nh.param<int>("sacn_line", N_SCANS, 16); // 16为默认值
     // 最小有效距离
     nh.param<double>("minimum_range", MINIMUN_RANGE, 0.1);
     
     std::cout << "scan line number: " << N_SCANS << std::endl;
     
     // 只有线束是12  32  64的才可以继续
     if (N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
     {
          std::cout << "only support velodyne with 16 32 64 scan line!" << std::endl;
          return 0;
     }

     ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

     pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

     // 发布曲率很大的点：每个scan的每部分选2个，每个scan总共2*6
     pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
     // 发布曲率较大的点：每个scan的每部分选20个，其中也包括曲率很大的点，每个scan总共20*6
     pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
     // 发布非常平坦的点：每个scan的每部分选4，每个scan总共4*6
     pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_flat", 100);
     // 发布比较平坦的点，剩余的点都是的
     pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_less_flat", 100);

     ros::spin();

     return 0;
}