#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h> 
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <math.h>
#include <sensor_msgs/point_field_conversion.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include "opencv2/objdetect/objdetect.hpp" 
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h> 
#include <message_filters/synchronizer.h> 
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
/*#include <ndi_aurora_msgs/AuroraData.h>
#include <ndi_aurora_msgs/AuroraDataVector.h>*/
#include <geometry_msgs/PoseStamped.h>
#include <unistd.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;  
int ICount=1;
//data from 20-10 cal group2 for fusion track system
double tsi[16]={0.99885,  -0.056172,  -0.0187474, 197.9938, -0.0568648, -0.99892054, -0.063445, 27.9413, -0.018004, 0.07481137, -0.999252, -54.97850716,0,0,0,1};

double homo1[16]={0.0};
sensor_msgs::PointCloud cloud;  
sensor_msgs::PointCloud cloud_pub;

void MultMatrix(int row_1, int col_1, double *temp1, int row_2, int col_2, double *temp2, double *result)
{
	int times = col_1;
	int row_result, col_result;
	int num = 0;
	for (row_result = 1; row_result <= row_1; row_result++)
	{
		for (col_result = 1; col_result <= col_2; col_result++)
		{
			result[num] = 0;
			for (int i=0; i <= times-1; i++)
			{
				result[num] += temp1[(row_result - 1)*col_1 + i] * temp2[i*col_2 + col_result -1 ];
			}
			num++;
		}
	}
}

void AddMatrix(double  *temp1,  double *temp2, double *result)
{
	int num = 0;
	for (int row_result = 0; row_result <= 15; row_result++)
	{
		result[num] += temp1[row_result ] + temp2[row_result ];
		num++;
	}
}

double det(int n, double *Mat)
{
	if (n == 1)
		return Mat[0];
	double *subMat = new double[(n - 1)*(n - 1)];
	int mov = 0; 
	double sum = 0.0;  
	for (int Matrow = 0; Matrow<n; Matrow++)  
	{
		for (int subMatrow = 0; subMatrow<n - 1; subMatrow++)
		{
			mov = Matrow > subMatrow ? 0 : 1;
			for (int j = 0; j<n - 1; j++)   
			{
				subMat[subMatrow*(n - 1) + j] = Mat[(subMatrow + mov)*n + j + 1];
			}
		}
		int flag = (Matrow % 2 == 0 ? 1 : -1);
		sum += flag* Mat[Matrow*n] * det(n - 1, subMat);
	}
	delete[]subMat;
	return sum;
}

Mat& MyGammaCorrection(Mat& src, float fGamma)   
  {  
    CV_Assert(src.data);  
    CV_Assert(src.depth() != sizeof(uchar));   
    unsigned char lut[256];  
    for( int i = 0; i < 256; i++ )  
    {  
        lut[i] = pow((float)(i/255.0), fGamma) * 255.0;  
    }  
    const int channels = src.channels();  
    switch(channels)  
    {  
        case 1:  
            {  
                MatIterator_<uchar> it, end;  
                for( it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++ )  
                    //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;  
                    *it = lut[(*it)];  
                break;  
            }  
        case 3:   
            {  
                MatIterator_<Vec3b> it, end;  
                for( it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; it++ )  
                {  
                    (*it)[0] = lut[((*it)[0])];  
                    (*it)[1] = lut[((*it)[1])];  
                    (*it)[2] = lut[((*it)[2])];  
                }  
                break;  
            }  
    }  
    return src;     
  }  


 
typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped,sensor_msgs::Image> MySyncPolicy;
ros::Publisher array_pub; 
ros::Publisher pc_pub; 
void imageCallback(const geometry_msgs::PoseStamped::ConstPtr & scana, const sensor_msgs::Image::ConstPtr & msg)  
{  
	void MultMatrix(int row_1, int col_1, double *temp1, int row_2, int col_2, double *temp2, double *result);
	void AddMatrix(double  *temp1,  double *temp2, double *result);
	geometry_msgs::PoseStamped scan;
	scan =*scana;
	double ox = scan.pose.position.x;
	double oy = scan.pose.position.y;
	double oz = scan.pose.position.z;
	double qx = scan.pose.orientation.x;
	double qy = scan.pose.orientation.y;
	double qz = scan.pose.orientation.z;
	double qw = scan.pose.orientation.w;
	double homo1[16]={0.0};
	for(int num=0;num<16;num++)
	{
		homo1[num]=0.0;
	}
	double ro[16]={2*qw*qw+2*qx*qx-1,2*qx*qy-2*qz*qw,2*qx*qz+2*qy*qw,0.0,2*qx*qy+2*qz*qw,2*qw*qw+2*qy*qy-1,2*qy*qz-2*qx*qw,0.0,2*qz*qx-2*qy*qw,2*qy*qz+2*qx*qw,2*qw*qw+2*qz*qz-1,0.0,0.0,0.0,0.0,0.0};
	double trans[16]={0.0,0.0,0.0,ox,0.0,0.0,0.0,oy,0.0,0.0,0.0,oz,0.0,0.0,0.0,1.0};
	AddMatrix(ro, trans, homo1); 


	cv_bridge::CvImageConstPtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);  
	}
	catch (cv_bridge::Exception& ex)
	{
		ROS_ERROR("cv_bridge exception in rgbcallback: %s", ex.what());
		exit(-1);
	}  

	Mat cameraFeed;
	Mat gray ;
	Mat thresh;
	vector<vector<Point> > conpoints;
	vector<Vec4i> hierarchy;
	int num=0;
	cameraFeed = cv_ptr->image; 
	 

/*
    ofstream fout;
    fout.open("/media/ruixuan/Volume/ruixuan/Pictures/auto_cali/pose3.txt", ios::app);
    fout << ox<<" "<<oy<<" "<<oz<<" "<<qx<<" "<<qy<<" "<<qz<<" "<<qw<<endl;
	cv::String path = "/media/ruixuan/Volume/ruixuan/Pictures/auto_cali/raw_3/";
    cv::String filename = path + to_string(ICount) + ".png";
    imwrite(filename, cameraFeed);
    cout << to_string(ICount)<<endl; 
*/


    cvtColor(cameraFeed, gray, CV_RGB2GRAY); 
    ofstream fout2;
    fout2.open("/media/ruixuan/Volume/ruixuan/Pictures/auto_cali2/result4.txt", ios::app); 

	threshold(gray,thresh, 180, 255, THRESH_BINARY); 
	findContours(thresh, conpoints, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
	Mat imageContours=Mat::zeros(thresh.size(),CV_8UC1);
	Mat Contours=Mat::zeros(thresh.size(),CV_8UC1);   
	for(int i=0;i<conpoints.size();i++)
	{
		for(int j=0;j<conpoints[i].size();j++) 
		{
			Point P=Point(conpoints[i][j].x,conpoints[i][j].y);
			Contours.at<uchar>(P)=255;
		}
		char ch[256];
		sprintf(ch,"%d",i);
		string str=ch;
		drawContours(imageContours,conpoints,i,Scalar(255),1,8,hierarchy);
	} 
 
	IplImage* img = new IplImage(thresh);
	uchar *data = (uchar *)img->imageData;
	int step = img->widthStep / sizeof(uchar);
	int channels = img->nChannels;
	int B;
	int raw[200][200]={0}; 
    float scale_u = 0.159;
    float scale_v = 0.156;

	for(int i =0; i <200 ; i++)  
	{
		for(int j = 0; j <200; j++) //640  
		{
			B = (int)data[i * step + j * channels + 0];
			raw[i][j]=B;
		} 
	}     

	for(int j = 0; j < 200; j++)   // 640
	{ 
		bool detect=true;
		for(int i = 0; i < 200  && detect; i++)
		{
            if( raw[i][j]-raw[i-1][j]==255) 
            { 
                double i_s=scale_u*i;
                double j_s=scale_v*j;
                double res[4]={0.0}; 
                double res1[4]={0.0};
                double uvw[4]={i_s,j_s,0,1};
                MultMatrix(4, 4, tsi, 4, 1, uvw, res1);
                MultMatrix(4, 4, homo1, 4, 1, res1, res);
                /*cloud.points[ICount*250+num].x = res[0] ;
                cloud.points[ICount*250+num].y = res[1] ;
                cloud.points[ICount*250+num].z = res[2] ;  
                cloud_pub.points[num].x = res[0] ;
                cloud_pub.points[num].y = res[1] -300;
                cloud_pub.points[num].z = res[2] -1800;   */
                // cout << res[0]<<","<<res[1]<<","<<res[2]<< endl;
                fout2 << res[0]<<","<<res[1]<<","<<res[2]<< endl; 
                num++;
                detect=false;
             } 
        } 
    }  


    /*sensor_msgs::PointCloud2 cloud2;
    cloud2.header.frame_id = "odom";
    sensor_msgs::convertPointCloudToPointCloud2(cloud,cloud2);
    array_pub.publish(cloud2);*/
    pc_pub.publish(cloud_pub); 
    fout2.close(); 
    ICount++;  
}



int main(int argc, char** argv)  
 {  
   ros::init(argc, argv, "pcl_recon"); 
   ros::NodeHandle nh_; 
   image_transport::ImageTransport it_(nh_);  
   message_filters::Subscriber<geometry_msgs::PoseStamped>* aurora_sub= new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh_, "/Fusion_track", 10);   
   image_transport::SubscriberFilter * image_sub  = new image_transport::SubscriberFilter(it_, "/IVUSimg", 10);
   message_filters::Synchronizer<MySyncPolicy> * sync= new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *aurora_sub, *image_sub);
   array_pub = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_index2", 5); 
   pc_pub = nh_.advertise<sensor_msgs::PointCloud>("/cloud_index", 5); 
   cloud.header.frame_id="sensor_frame";
   cloud.points.resize(500*500);
   cloud_pub.header.frame_id="sensor_frame_fusion";
   cloud_pub.points.resize(20*20);
   sync->registerCallback(boost::bind(imageCallback,_1,_2));
   ros::spin();
   return 0;  
 } 