////////// YOLO w/ KF
// yolov8_decoder_node.hpp

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
#define ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"

#include "std_msgs/msg/bool.hpp"     // For Bool message type
#include "std_msgs/msg/int8.hpp"     // For Int8 message type
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

// Include the YOLOKF class
#include "yolo_kf/yolo_kf.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

class YoloV8DecoderNode : public rclcpp::Node
{
public:
  explicit YoloV8DecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~YoloV8DecoderNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);
  void FsmFlagCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void FsmSelectTargetCallback(const std_msgs::msg::Int8::SharedPtr msg);
  // void kf_timer_callback();

  // Subscriptions for FSM flags and selected target
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr fsm_flag_sub_;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr fsm_select_target_sub_;

  // Subscription to input NitrosTensorList messages
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

  // Publishers for output Detection2DArray messages
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr selected_target_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr selected_target_kf_pub_;

  // // Timer for EKF
  // rclcpp::TimerBase::SharedPtr ekf_timer_;
    

  // Name of tensor in NitrosTensorList
  std::string tensor_name_{};

  // YOLOv8 Decoder Parameters
  double confidence_threshold_{};
  double nms_threshold_{};
  bool start_detection_flag_;  // Flag to control whether detection should be active
  int selected_target_id_;     // Selected target for detection

  // YOLOKF object to handle Kalman filtering for the selected target
  double fs_;
  std::shared_ptr<YOLOKF> yolo_kf_;  // Add a YOLOKF instance
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_


























////////// YOLO w/o KF
// // SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// // Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// // http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// // SPDX-License-Identifier: Apache-2.0

// #ifndef ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
// #define ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_

// #include <memory>
// #include <string>

// #include "rclcpp/rclcpp.hpp"

// #include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"

// #include "std_msgs/msg/bool.hpp"  // Add this for the Bool message type
// #include "std_msgs/msg/int8.hpp"  // Include this for Int8 message type
// #include "std_msgs/msg/string.hpp"
// #include "vision_msgs/msg/detection2_d_array.hpp"
// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

// namespace nvidia
// {
// namespace isaac_ros
// {
// namespace yolov8
// {

// class YoloV8DecoderNode : public rclcpp::Node
// {
// public:
//   explicit YoloV8DecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

//   ~YoloV8DecoderNode();

// private:
//   void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);
//   void FsmFlagCallback(const std_msgs::msg::Bool::SharedPtr msg);
//   void FsmSelectTargetCallback(const std_msgs::msg::Int8::SharedPtr msg);

//   // Subscription to FSM flag and selected target
//   rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr fsm_flag_sub_; // Subscription for the start detection flag
//   rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr fsm_select_target_sub_; // Subscription for the selected target


//   // Subscription to input NitrosTensorList messages
//   std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
//       nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

//   // Publisher for output Detection2DArray messages
//   rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;
//   rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr selected_target_pub_;

//   // Name of tensor in NitrosTensorList
//   std::string tensor_name_{};

//   // YOLOv8 Decoder Parameters
//   double confidence_threshold_{};
//   double nms_threshold_{};
//   bool start_detection_flag_; // Flag to control whether detection should be active
//   int selected_target_id_; // Selected target for detection
// };

// }  // namespace yolov8
// }  // namespace isaac_ros
// }  // namespace nvidia

// #endif  // ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_






























////////// Original code
// // SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// // Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// // http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// // SPDX-License-Identifier: Apache-2.0

// #ifndef ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
// #define ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_

// #include <memory>
// #include <string>

// #include "rclcpp/rclcpp.hpp"

// #include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"

// #include "std_msgs/msg/string.hpp"
// #include "vision_msgs/msg/detection2_d_array.hpp"
// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

// namespace nvidia
// {
// namespace isaac_ros
// {
// namespace yolov8
// {

// class YoloV8DecoderNode : public rclcpp::Node
// {
// public:
//   explicit YoloV8DecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

//   ~YoloV8DecoderNode();

// private:
//   void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);

//   // Subscription to input NitrosTensorList messages
//   std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
//       nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

//   // Publisher for output Detection2DArray messages
//   rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;

//   // Name of tensor in NitrosTensorList
//   std::string tensor_name_{};

//   // YOLOv8 Decoder Parameters
//   double confidence_threshold_{};
//   double nms_threshold_{};
// };

// }  // namespace yolov8
// }  // namespace isaac_ros
// }  // namespace nvidia

// #endif  // ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
