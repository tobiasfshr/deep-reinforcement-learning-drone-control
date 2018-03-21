#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#include "ros/ros.h"
#include "mav_msgs/default_topics.h"
#include "mav_msgs/RollPitchYawrateThrust.h"
#include "mav_msgs/Actuators.h"
#include "mav_msgs/eigen_mav_msgs.h"
#include "geometry_msgs/Pose.h"
#include "std_srvs/Empty.h"
#include "gazebo_msgs/GetModelState.h"
#include "gazebo_msgs/SetModelState.h"
#include "gazebo_msgs/ContactsState.h"
#include "rotors_comm/WindSpeed.h"
#include "gazebo_msgs/ModelState.h"

#include <image_transport/image_transport.h>

#include "rotors_reinforce/PerformAction.h"
#include "rotors_reinforce/GetState.h"

#include <sstream>
#include <random>

const double max_v_xy = 1.0;  // [m/s]
const double max_roll = 10.0 * M_PI / 180.0;  // [rad]
const double max_pitch = 10.0 * M_PI / 180.0;  // [rad]
const double max_rate_yaw = 45.0 * M_PI / 180.0;  // [rad/s]
const double max_thrust = 30.0;  // [N]

const double MAX_WIND_VELOCTIY = 3.4; // meter per second --> max 12km/h (equals wind force 3)

const double axes_roll_direction = -1.0;
const double axes_pitch_direction = 1.0;
const double axes_thrust_direction = 1.0;

class environmentTracker {

private:
    ros::NodeHandle n;
    ros::Publisher firefly_control_pub;
    ros::Publisher firefly_wind_pub;
    
    ros::ServiceClient firefly_reset_client;
    ros::ServiceClient get_position_client;
    ros::ServiceClient pause_physics;
    ros::ServiceClient unpause_physics;
    ros::ServiceClient set_state;


    ros::Subscriber firefly_position_sub;
    ros::Subscriber firefly_collision_sub;
    ros::Subscriber firefly_ground_collision_sub;
    ros::Subscriber firefly_camera_sub;
    ros::Subscriber firefly_camera_depth_sub;
    ros::ServiceServer perform_action_srv;
    ros::ServiceServer get_state_srv;

    std::default_random_engine re;

    sensor_msgs::Image current_img;
    sensor_msgs::Image current_img_depth;

    int step_counter;
    double current_yaw_vel_;
    bool crashed_flag;
    bool random_target;
    bool enable_wind;

public:
    std::vector<double> current_position;
    std::vector<double> current_orientation;
    std::vector<double> current_control_params;

    std::vector<double> target_position;

    environmentTracker(ros::NodeHandle node, const std::vector<double> target_pos, bool wind ) {
    	current_position.resize(3);
        //hard constants for target
        if (target_pos[0] != 0.0 || target_pos[1] != 0.0 || target_pos[2] != 0.0) {
            target_position = target_pos;
            random_target = false;
        }
        else {
            target_position = {3.0, 1.0, 7.5};
            random_target = true;
        }
        enable_wind = wind;
    	current_orientation.resize(4);
        current_control_params.resize(4, 0);
    	step_counter = 0;

        current_yaw_vel_ = 0.0;

        n = node;
        firefly_control_pub = n.advertise<mav_msgs::RollPitchYawrateThrust>("/firefly/command/roll_pitch_yawrate_thrust", 1000);
        firefly_wind_pub = n.advertise<rotors_comm::WindSpeed>("/firefly/wind_speed", 1000);
        firefly_collision_sub = n.subscribe("/rotor_collision", 100, &environmentTracker::onCollision, this);
        firefly_ground_collision_sub = n.subscribe("/base_collision", 100, &environmentTracker::onCollisionGround, this);
        firefly_reset_client = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
        firefly_camera_sub = n.subscribe("/firefly/vi_sensor/camera_depth/camera/image_raw", 1, &environmentTracker::getImage, this);
        //firefly_camera_depth_sub = n.subscribe("/firefly/vi_sensor/camera_depth/depth/disparity", 1, &environmentTracker::getImageDepth, this);

        pause_physics = n.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
        unpause_physics = n.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
        set_state = n.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");

        get_position_client = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
        perform_action_srv = n.advertiseService("env_tr_perform_action", &environmentTracker::performAction, this);
        get_state_srv = n.advertiseService("env_tr_get_state", &environmentTracker::getState, this);

        gazebo_msgs::ModelState modelstate;
        gazebo_msgs::SetModelState set_state_srv;
        modelstate.model_name = "firefly";
        modelstate.pose.position.x = 0.0;
        modelstate.pose.position.y = 0.0;
        modelstate.pose.position.z = 5.0;
        set_state_srv.request.model_state = modelstate;
        set_state.call(set_state_srv);
    }

    double round_number(double number){
        return round( number * 100.0 ) / 100.0;
    }

    void getImage(const sensor_msgs::ImageConstPtr& msg)
    {

        current_img = *msg;
    }

    void getImageDepth(const sensor_msgs::ImageConstPtr& msg)
    {

        current_img_depth = *msg;
    }

    void setCrossPosition(std::vector<double> pos) {
            gazebo_msgs::ModelState modelstate;

            gazebo_msgs::SetModelState set_state_srv;

            modelstate.model_name = "small box";
            modelstate.pose.position.x = pos[0];
            modelstate.pose.position.y = pos[1];
            modelstate.pose.position.z = 0.08;

            set_state_srv.request.model_state = modelstate;

            if (!set_state.call(set_state_srv)) {
                //ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
                ROS_ERROR("Failed to set position");
            }

            modelstate.model_name = "small box_0";
            modelstate.pose.position.x = pos[0];
            modelstate.pose.position.y = pos[1] + 0.2;
            modelstate.pose.position.z = 0.08;

            set_state_srv.request.model_state = modelstate;

            if (!set_state.call(set_state_srv)) {
                //ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
                ROS_ERROR("Failed to set position");
            }

            modelstate.model_name = "small box_1";
            modelstate.pose.position.x = pos[0];
            modelstate.pose.position.y = pos[1] - 0.2;
            modelstate.pose.position.z = 0.08;

            set_state_srv.request.model_state = modelstate;

            if (!set_state.call(set_state_srv)) {
                //ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
                ROS_ERROR("Failed to set position");
            }

            modelstate.model_name = "small box_2";
            modelstate.pose.position.x = pos[0] + 0.31;
            modelstate.pose.position.y = pos[1];
            modelstate.pose.position.z = 0.08;

            set_state_srv.request.model_state = modelstate;

            if (!set_state.call(set_state_srv)) {
                //ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
                ROS_ERROR("Failed to set position");
            }

            modelstate.model_name = "small box_3";
            modelstate.pose.position.x = pos[0] - 0.31;
            modelstate.pose.position.y = pos[1];
            modelstate.pose.position.z = 0.08;

            set_state_srv.request.model_state = modelstate;

            if (!set_state.call(set_state_srv)) {
                //ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);
                ROS_ERROR("Failed to set position");
            }
        }

    void respawn() {
        mav_msgs::RollPitchYawrateThrust msg;
        msg.roll = 0;
        msg.pitch = 0;
        msg.yaw_rate = 0;
        msg.thrust.z = 0;
        std_srvs::Empty srv;

        current_position = {0,0,0};
        current_orientation = {0,0,0,0};
        step_counter = 0;

        if (random_target) {
            std::uniform_real_distribution<double> unif(0, 8);

            target_position = {round_number(unif(re))-4, round_number(unif(re))-4, 7.5};//round_number(unifz(re))};
        }

        current_control_params.resize(4, 0);
        firefly_reset_client.call(srv);
        firefly_control_pub.publish(msg);
        setCrossPosition(target_position);
        gazebo_msgs::ModelState modelstate;
        gazebo_msgs::SetModelState set_state_srv;
        modelstate.model_name = "firefly";
        modelstate.pose.position.x = 0.0;
        modelstate.pose.position.y = 0.0;
        modelstate.pose.position.z = 5.0;
        set_state_srv.request.model_state = modelstate;
        set_state.call(set_state_srv);
    }

    void pausePhysics() {
    	std_srvs::Empty srv;
    	pause_physics.call(srv);
    }

    void unpausePhysics() {
    	std_srvs::Empty srv;
    	unpause_physics.call(srv);
    }

    void onCollisionGround(const gazebo_msgs::ContactsState::ConstPtr& msg) {
        if ((step_counter > 5) && (current_position[2] < 0.5 || current_position[2] > 12.5 || msg->states.size() > 0)) {
            ROS_INFO("Crash, respawn...");
            step_counter = 0;
            crashed_flag = true;
            respawn();
        }
    }

    void onCollision(const gazebo_msgs::ContactsState::ConstPtr& msg) {
        if ((step_counter > 5) && (current_position[2] < 0.5 || current_position[2] > 12.5 || msg->states.size() > 0)) {
                ROS_INFO("Crash, respawn...");
                step_counter = 0;
                crashed_flag = true;
                respawn();
        }
    }

    bool performAction(rotors_reinforce::PerformAction::Request  &req, rotors_reinforce::PerformAction::Response &res) {
        //respawn code
        if (req.action[3] == 42) {
            respawn();
            return true;
        }

        if ((step_counter > 5) && (current_position[2] < 0.5 || current_position[2] > 12.5)) {
            ROS_INFO("Crash, respawn...");
            step_counter = 0;
            crashed_flag = true;
            respawn();
        }

        mav_msgs::RollPitchYawrateThrust msg;
        msg.roll = req.action[0] * max_roll * axes_roll_direction;
        msg.pitch = req.action[1] * max_pitch * axes_pitch_direction;

        if(req.action[2] > 0.01) {
            current_yaw_vel_ = max_rate_yaw;
        }
        else if (req.action[2] < -0.01) {
            current_yaw_vel_ = max_rate_yaw;   
        }
        else {
            current_yaw_vel_ = 0.0;   
        }

        msg.yaw_rate = current_yaw_vel_;
        msg.thrust.z = req.action[3] * max_thrust * axes_thrust_direction;

    
        ROS_INFO("roll: %f, pitch: %f, yaw_rate: %f, thrust %f", msg.roll, msg.pitch, msg.yaw_rate, msg.thrust.z);

        if (enable_wind && step_counter % 20 == 0) { //new wind after every 20 steps (1 second)
            rotors_comm::WindSpeed wind_msg;
            std::uniform_real_distribution<double> unif(0, 1);
            if (unif(re) > 0.75) { //probability of 75% for no wind
                double rand_number = unif(re);
                if (rand_number < 0.34) { //wind comes only from one direction
                    wind_msg.velocity.z = 0;
                    wind_msg.velocity.y = 0;
                    wind_msg.velocity.x = unif(re) * MAX_WIND_VELOCTIY;
                } else if (rand_number < 0.67) {
                    wind_msg.velocity.x = 0;
                    wind_msg.velocity.z = 0;
                    wind_msg.velocity.y = unif(re) * MAX_WIND_VELOCTIY;
                } else {
                    wind_msg.velocity.x = 0;
                    wind_msg.velocity.y = 0;
                    wind_msg.velocity.z = unif(re) * MAX_WIND_VELOCTIY;
                }
                ROS_INFO("wind with velocity of x: %f, y: %f, z: %f", wind_msg.velocity.x, wind_msg.velocity.y, wind_msg.velocity.z);
            }
            else {
                ROS_INFO("no wind");
                wind_msg.velocity.x = 0;
                wind_msg.velocity.y = 0;
                wind_msg.velocity.z = 0;
            }
            firefly_wind_pub.publish(wind_msg);
        }
        unpausePhysics();
        unpausePhysics();
        firefly_control_pub.publish(msg);
        ros::Duration(0.05).sleep(); //sleep 50ms of simulation time
    	getPosition();
        pausePhysics();
        step_counter++;

        res.target_position = target_position;
        res.position = current_position;
        res.orientation = current_orientation;
        res.img = current_img;
        res.img_depth = current_img_depth;

        res.reward = getReward(step_counter);
        res.crashed = false;

        //crash check at the end        
        if(crashed_flag) {
            res.crashed = true;
            crashed_flag = false;
        }

        return true;
    }

    bool getState(rotors_reinforce::GetState::Request &req, rotors_reinforce::GetState::Response &res) {
        getPosition();
        res.target_position = target_position;
        res.position = current_position;
        res.orientation = current_orientation;
        res.img = current_img;
        res.img_depth = current_img_depth;

        res.reward = 0.0;
        res.crashed = crashed_flag;
        return true;
    }

    void getPosition() {
            gazebo_msgs::GetModelState srv;
            srv.request.model_name = "firefly";
            if (get_position_client.call(srv)) {
                ROS_INFO("Position: %f %f %f", (float)srv.response.pose.position.x, (float)srv.response.pose.position.y, (float)srv.response.pose.position.z);

                current_position[0] = (double)srv.response.pose.position.x;
                current_position[1] = (double)srv.response.pose.position.y;
                current_position[2] = (double)srv.response.pose.position.z;
                current_orientation[0] = (double)srv.response.pose.orientation.x;
                current_orientation[1] = (double)srv.response.pose.orientation.y;
                current_orientation[2] = (double)srv.response.pose.orientation.z;
                current_orientation[3] = (double)srv.response.pose.orientation.w;
            }
            else {
                ROS_ERROR("Failed to get position");
            }
    }

    double getReward(const int count) {
        double difx = current_position[0] - target_position[0];
        double dify = current_position[1] - target_position[1];
        double difz = current_position[2] - target_position[2];

        double current_distance = std::sqrt(difx * difx + dify * dify + 2.0 * difz * difz);
 
        double reward = 0.0;

        if (crashed_flag) {
            return 0.0;
        }

        double reward4position =  1/(current_distance + 1.0);
        //double reward4orientation = 1/((current_orientation[0] * current_orientation[0] + current_orientation[1] * current_orientation[1] + current_orientation[2] * current_orientation[2])/(current_orientation[3] * current_orientation[3]) + 1);

        reward = reward4position;//*0.95 + 0.05 * reward4orientation;

        return reward;
    }

};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "talker");

    ros::NodeHandle n;

    ros::Rate loop_rate(100);

    std::vector<double> target_position = {0.0, 0.0, 0.0};
    bool wind = false;

    if(argc > 1) {
        target_position[0] = atoi(argv[1]);
        target_position[1] = atoi(argv[2]);
        target_position[2] = atoi(argv[3]);
        wind = atoi(argv[4]);
    }

    environmentTracker tracker(n, target_position, wind);

    ROS_INFO("Comunication node ready");

    ros::spin();

    //delete tracker;
    return 0;
}
