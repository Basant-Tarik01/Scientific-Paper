// World plugin that moves the kinematic worker models (ped_<id>) in Gazebo Classic to the states
// published by the pedestrian_sim node on /pedestrians (amr_interfaces/PedestrianArray).
// Poses are extrapolated with the published velocity between messages so the motion is smooth at
// the physics rate. Inactive workers are parked outside the building, beyond LiDAR range.
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <amr_interfaces/msg/pedestrian_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>

namespace amr_gazebo_plugins
{
class PedestrianPosePlugin : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;
    node_ = gazebo_ros::Node::Get(sdf);
    prefix_ = sdf->HasElement("model_prefix") ? sdf->Get<std::string>("model_prefix") : "ped_";
    park_x_ = sdf->HasElement("park_x") ? sdf->Get<double>("park_x") : -20.0;
    park_y_ = sdf->HasElement("park_y") ? sdf->Get<double>("park_y") : -20.0;
    park_spacing_ = sdf->HasElement("park_spacing") ? sdf->Get<double>("park_spacing") : 1.5;
    sub_ = node_->create_subscription<amr_interfaces::msg::PedestrianArray>(
      "/pedestrians", rclcpp::QoS(5),
      [this](amr_interfaces::msg::PedestrianArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = msg;
        stamp_ = world_->SimTime();
      });
    update_conn_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&PedestrianPosePlugin::OnUpdate, this));
    RCLCPP_INFO(node_->get_logger(), "pedestrian_pose plugin loaded (prefix '%s')", prefix_.c_str());
  }

private:
  void OnUpdate()
  {
    amr_interfaces::msg::PedestrianArray::SharedPtr msg;
    gazebo::common::Time stamp;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      msg = latest_;
      stamp = stamp_;
    }
    if (!msg) {
      return;
    }
    const double dt = std::clamp((world_->SimTime() - stamp).Double(), 0.0, 0.25);
    for (const auto & ped : msg->pedestrians) {
      auto model = world_->ModelByName(prefix_ + std::to_string(ped.id));
      if (!model) {
        continue;
      }
      ignition::math::Pose3d pose;
      if (ped.active) {
        const double x = ped.pose.position.x + ped.twist.linear.x * dt;
        const double y = ped.pose.position.y + ped.twist.linear.y * dt;
        const auto & q = ped.pose.orientation;
        const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        pose.Set(x, y, 0.0, 0.0, 0.0, yaw);
      } else {
        pose.Set(park_x_ - park_spacing_ * ped.id, park_y_, 0.0, 0.0, 0.0, 0.0);
      }
      model->SetWorldPose(pose);
      model->SetLinearVel(ignition::math::Vector3d(ped.twist.linear.x, ped.twist.linear.y, 0.0));
      model->SetAngularVel(ignition::math::Vector3d::Zero);
    }
  }

  gazebo::physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Subscription<amr_interfaces::msg::PedestrianArray>::SharedPtr sub_;
  gazebo::event::ConnectionPtr update_conn_;
  std::mutex mutex_;
  amr_interfaces::msg::PedestrianArray::SharedPtr latest_;
  gazebo::common::Time stamp_;
  std::string prefix_;
  double park_x_{-20.0}, park_y_{-20.0}, park_spacing_{1.5};
};

GZ_REGISTER_WORLD_PLUGIN(PedestrianPosePlugin)
}  // namespace amr_gazebo_plugins
