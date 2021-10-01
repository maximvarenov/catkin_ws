#include <ros/ros.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
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

using namespace pcl;
using namespace pcl::io;
using namespace std;

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
string input_filename = "/media/ruixuan/Volume/ruixuan/Documents/database/us_image/25-11/reconstruction2.pcd";
string output_filename = "/media/ruixuan/Volume/ruixuan/Documents/database/us_image/25-11/reconstruction2.ply";
PCDtoPLYconvertor(input_filename , output_filename);
return 0;
}


