#include <iostream>
#include <ros/ros.h> 
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/visualization/cloud_viewer.h>
    
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/PCLPointCloud2.h>
#include <iostream>
#include <string> 

using namespace std;
using namespace pcl; 
using namespace pcl::io; 
//draw pointcloud  and visualize in real time 

 
/*

pcl::visualization::CloudViewer viewer("viewer"); 
void callback(const sensor_msgs::PointCloud2::ConstPtr & scana)  
{     
  cout << "working ........ 11"<< endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr newcloud(new pcl::PointCloud<pcl::PointXYZ>); 
  pcl::fromROSMsg (*scana, *newcloud);
  pcl::visualization::CloudViewer viewer("viewer"); 
  viewer.showCloud(newcloud);    
}


int main(int argc, char** argv)  
{  
  ros::init(argc, argv, "PointCloud_pcl"); 
  ros::NodeHandle nh_; 
  ros::Subscriber sub = nh_.subscribe("/cloud_index", 10, callback);  
  ros::spin();
  return 0;  
}
 
*/



  
//txt to pcd file 
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool readTxtFile(const string &fileName, const char tag, const PointCloudT::Ptr &pointCloud)
{
  cout << "reading file start..... " << endl;
  ifstream fin(fileName);
  string linestr;
  vector<PointT> myPoint;
  while (getline(fin, linestr))
  {
    vector<string> strvec;
    string s;
    stringstream ss(linestr);
    while (getline(ss, s, tag))
    {
      strvec.push_back(s);
    }
    if (strvec.size() < 3){
      cout << "error" << endl;
      return false;
    }
    PointT p;
    p.x = stod(strvec[0]);
    p.y = stod(strvec[1]);
    p.z = stod(strvec[2]);
    myPoint.push_back(p);
  }
  fin.close();
  pointCloud->width = (int)myPoint.size();
  pointCloud->height = 1;
  pointCloud->is_dense = false;
  pointCloud->points.resize(pointCloud->width * pointCloud->height);
  for (int i = 0; i < myPoint.size(); i++)
  {
    pointCloud->points[i].x = myPoint[i].x;
    pointCloud->points[i].y = myPoint[i].y;
    pointCloud->points[i].z = myPoint[i].z;
  }
  cout << "reading file finished! " << endl;
  cout << "There are " << pointCloud->points.size() << " points!" << endl;
  return true;
}

int PCDtoPLYconvertor(string & input_filename ,string& output_filename)
{
pcl::PCLPointCloud2 cloud;
if (loadPCDFile(input_filename , cloud) < 0)
{
cout << "Error: cannot load the PCD file!!!"<< endl;
return -1;
}
PLYWriter writer;
writer.write(output_filename, cloud, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(),true,true);
return 0;

}


int main()
{
  /*PointCloudT::Ptr cloud(new PointCloudT);
  for (int i =1; i++; i<+12)
  {
    string path ="/media/ruixuan/Volume/ruixuan/Documents/database/us_image/auto_cal/recon/manual" + to_string(i) + ".txt";
  
    readTxtFile(path,',',cloud);
 
  
  pcl::visualization::PCLVisualizer viewer("cloud viewer");
  viewer.addPointCloud<PointT>(cloud, "sample");
  while (!viewer.wasStopped())
  {
    viewer.spinOnce();
  }
 
  
 string path_pcd= "/media/ruixuan/Volume/ruixuan/Documents/database/us_image/auto_cal/recon/manual" + to_string(i) + ".pcd";
  
   pcl::PCDWriter writer;
   writer.writeASCII<PointT>( path_pcd, *cloud); 
  

   string path_ply = "/media/ruixuan/Volume/ruixuan/Documents/database/us_image/auto_cal/recon/manual" + to_string(i) + ".ply";
   PCDtoPLYconvertor(path_pcd , path_ply);
  
}*/

    PointCloudT::Ptr cloud(new PointCloudT);
    string path ="/media/ruixuan/Volume/ruixuan/Pictures/auto_cali3/result1_cv_test1.txt";
    readTxtFile(path,',',cloud); 
    string path_pcd= "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali3/result1_cv_test1.pcd";
    pcl::PCDWriter writer;
    writer.writeASCII<PointT>( path_pcd, *cloud); 
    string path_ply = "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali3/result1_cv_test1.ply";
    PCDtoPLYconvertor(path_pcd , path_ply);
   /*for (int i =1; i<=4 ; i++)
   {
    string path ="/media/ruixuan/Volume/ruixuan/Pictures/auto_cali2/result" + to_string(i) + ".txt";
    readTxtFile(path,',',cloud); 
   
 
 string path_pcd= "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali2/result" + to_string(i) + ".pcd";
   pcl::PCDWriter writer;
   writer.writeASCII<PointT>( path_pcd, *cloud); 
   string path_ply = "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali2/result" + to_string(i) + ".ply";
   PCDtoPLYconvertor(path_pcd , path_ply);
}*/
  system("pause");
  return 0;
}
