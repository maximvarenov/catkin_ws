#include "hw3_0410817/hw3_node.h"
#include <geometry_msgs/Quaternion.h>
#include <string>
#include <vector>
#include <sstream>
#include <math.h> 
#include <fstream>

#define PI 3.14159265

ImuIntegrator::ImuIntegrator(const ros::Publisher &pub)
{
    Eigen::Vector3d zero(0, 0, 0);
    pose.pos = zero;
    pose.orien = Eigen::Matrix3d::Identity();
    velocity = zero;
    line_pub = pub;
    firstT = true;

    // Line strip is blue
    path.color.b = 1.0;
    path.color.a = 1.0;
    path.type = visualization_msgs::Marker::CUBE;
    path.header.frame_id = "/world";
    path.ns = "points_and_lines";
    path.action = visualization_msgs::Marker::ADD;
    path.pose.orientation.w = 1.0;
    path.scale.x = 0.2;
    geometry_msgs::Point p;
    p.x = 0;
    p.y = 0;
    p.z = 0;
    path.points.push_back(p);
}

void ImuIntegrator::ImuCallback(const sensor_msgs::Imu &msg)
{
    if (firstT) {
        time = msg.header.stamp;
        deltaT = 0;
        firstT = false;
    }
    else {
        deltaT = (msg.header.stamp - time).toSec();
        time = msg.header.stamp;
        calcOrientation1(msg.orientation);
        calcPosition(msg.linear_acceleration);
        updatePath(pose.pos);
        publishMessage();
    }
    //std::cout << pose.pos << std::endl;
}



void ImuIntegrator::updatePath(const Eigen::Vector3d &msg)
{
    geometry_msgs::Point p;
    p.x = msg[0];
    p.y = msg[1];
    p.z = msg[2];
    path.points.push_back(p);
}

void ImuIntegrator::publishMessage()
{
    line_pub.publish(path);
}

void ImuIntegrator::calcOrientation1(const geometry_msgs::Quaternion &msg)
{
    Eigen::Quaterniond quaternion4(msg.w,msg.x,msg.y,msg.z);
    pose.orien =  quaternion4.toRotationMatrix();
}

void ImuIntegrator::calcPosition(const geometry_msgs::Vector3 &msg)
{
    Eigen::Vector3d acc(msg.x -0, msg.y - 0, msg.z - 0);
    pose.pos = pose.pos + deltaT * velocity + 0.5 * deltaT * deltaT * acc;   
    velocity = velocity + deltaT * acc;
    std::cout<<time <<" "<<acc[0]<<" " <<acc[1]<< " "<<acc[2]<<" "<<velocity[0] << " "<<velocity[1]<< " "<<velocity[2]<<" "<<pose.pos[0] << " "<<pose.pos[1]<< " "<<pose.pos[2]<< std::endl;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "hw3_node");
    ros::NodeHandle nh;
    ros::Publisher line = nh.advertise<visualization_msgs::Marker>("Imu_path", 1000);
    ImuIntegrator *imu_integrator = new ImuIntegrator(line);
    ros::Subscriber Imu_message = nh.subscribe("/imu_data", 1000, &ImuIntegrator::ImuCallback, imu_integrator);
    ros::spin();
}



/*
class ImuOdom{
private:
    //ROS essentials
    ros::NodeHandle n;
    ros::Subscriber sub_imu;
    ros::Publisher pub_marker;
    uint16_t marker_id;
    std::vector<geometry_msgs::Point> points;
    //
    //R body to global (3x3)
    Eigen::Matrix3d C;
    Eigen::Vector3d gravity;
    Eigen::Vector3d line_vel, pos;
    ros::Time last_enter;
    void update_orientation(const geometry_msgs::Vector3& ang_vel, const ros::Duration time_diff){
        Eigen::Matrix3d trans = Eigen::Matrix3d::Zero();
        double sigma = (Eigen::Vector3d(ang_vel.x, ang_vel.y, ang_vel.z)*(double)time_diff.toNSec()/1e9).norm();
        trans(0, 1) = -ang_vel.z;
        trans(0, 2) = ang_vel.y;
        trans(1, 0) = ang_vel.z;
        trans(1, 2) = -ang_vel.x;
        trans(2, 0) = -ang_vel.y;
        trans(2, 1) = ang_vel.x;
        trans *= (double)time_diff.toNSec()/1e9;
        C = C * (Eigen::Matrix3d::Identity() + trans*(sin(sigma)/sigma) + trans*trans*(((double)1-cos(sigma))/(sigma*sigma)) );
        //std::cout << C << std::endl;

    }
    void update_position(const geometry_msgs::Vector3& lin_accel, const ros::Duration time_diff){
        Eigen::Vector3d glo_acc = C * Eigen::Vector3d(lin_accel.x, lin_accel.y, lin_accel.z);
        this->line_vel += (glo_acc - gravity)*(double)time_diff.toNSec()/1e9;
        this->pos += this->line_vel*(double)time_diff.toNSec()/1e9;
        //std::cout << "acc_g"<<(glo_acc - gravity)<< std::endl;
        std::cout << line_vel[0] << std::endl;
        //std::cout << pos << std::endl;
    }

    void msg_cb(const sensor_msgs::Imu::ConstPtr& msg){
        ros::Time enter_time = msg->header.stamp;
        //std::cout << "linear_acc= \n" << msg->linear_acceleration;
        if(last_enter == ros::Time(0)){
            ROS_INFO("First msg");
            this->C = Eigen::Matrix3d::Identity(); // initial orientation
            this->line_vel = Eigen::Vector3d::Zero();
            this->pos = Eigen::Vector3d::Zero();
            this->gravity = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);  //Initialize gravity
            this->last_enter = enter_time;
            return;
        }
        //Propergate
        this->update_orientation(msg->angular_velocity, enter_time-this->last_enter);
        this->update_position(msg->linear_acceleration, enter_time-this->last_enter);
        
        //draw marker
        visualization_msgs::Marker marker;
        geometry_msgs::Point point;
        point.x = pos(0); point.y = pos(1); point.z = pos(2);
        points.push_back(point);
        marker.header.frame_id = "global";
        marker.header.stamp = ros::Time::now();
        marker.ns = "imu_pose";
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = pos(0);
        marker.pose.position.y = pos(1);
        marker.pose.position.z = pos(2);
        marker.points = this->points;
        marker.scale.x = 0.1;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        this->pub_marker.publish( marker );

        this->last_enter = enter_time;
        return;
    }
public:
    ImuOdom() {}
    ImuOdom(ros::NodeHandle nh){
        this->n = nh;
        ROS_INFO("Node Initializing");
        this->last_enter = ros::Time(0);
        this->sub_imu = n.subscribe("/imu_data", 1, &ImuOdom::msg_cb, this);
        this->pub_marker = n.advertise<visualization_msgs::Marker>("/position", 1);
        marker_id = 0;
        ROS_INFO("Initialize Done");
    }



};

int main(int argc, char** argv){
    ros::init(argc, argv, "imu_odom");
    ros::NodeHandle n;
    ImuOdom node(n);
    ros::spin();
    return 0;
}



*/