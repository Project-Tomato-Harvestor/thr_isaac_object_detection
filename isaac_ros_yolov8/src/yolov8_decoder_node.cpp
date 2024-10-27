////////// YOLO w/ KF, w/o timer
// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include "std_msgs/msg/bool.hpp"  // Add this for the Bool message type
#include "std_msgs/msg/int8.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// Include YOLOKF
#include "yolo_kf/yolo_kf.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{
YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV8DecoderNode::InputCallback, this,
      std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  selected_target_pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "selected_target_output", 50)},
  selected_target_kf_pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
    "selected_target_kf_output", 50)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  start_detection_flag_{false}, // Initialise detection flag to false
  selected_target_id_{1}, // Initialise selected target id to -1 (none of the classes)
  fs_(20.0), // Set the sampling frequency to 20 Hz
  yolo_kf_{std::make_shared<YOLOKF>(this)} // Instantiate YOLOKF
{
  // Add a subscriber to control detection start/stop and select target
  fsm_flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/fsm_flag/start_detection",
      10, std::bind(&YoloV8DecoderNode::FsmFlagCallback, this, std::placeholders::_1));
    
  fsm_select_target_sub_ = this->create_subscription<std_msgs::msg::Int8>(
      "/fsm_flag/select_target",
      10, std::bind(&YoloV8DecoderNode::FsmSelectTargetCallback, this, std::placeholders::_1));
}

YoloV8DecoderNode::~YoloV8DecoderNode() = default;

// Callback for the FSM flag and selected target
void YoloV8DecoderNode::FsmFlagCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  start_detection_flag_ = msg->data;
  // RCLCPP_INFO(this->get_logger(), "Detection flag updated: %s", start_detection_flag_ ? "true" : "false");
}

void YoloV8DecoderNode::FsmSelectTargetCallback(const std_msgs::msg::Int8::SharedPtr msg)
{
  selected_target_id_ = msg->data;
  RCLCPP_INFO(this->get_logger(), "Selected target updated: %d", selected_target_id_);
}

void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  // Check if detection is allowed
  if (!start_detection_flag_) {
    // RCLCPP_INFO(this->get_logger(), "Detection is currently disabled, skipping InputCallback.");
    return;
  }

  auto tensor = msg.GetNamedTensor(tensor_name_);
  size_t buffer_size{tensor.GetTensorSize()};
  std::vector<float> results_vector{};
  results_vector.resize(buffer_size);
  cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  int num_classes = 2;
  int out_dim = 8400;
  float * results_data = reinterpret_cast<float *>(results_vector.data());

  for (int i = 0; i < out_dim; i++) {
    float x = *(results_data + i);
    float y = *(results_data + (out_dim * 1) + i);
    float w = *(results_data + (out_dim * 2) + i);
    float h = *(results_data + (out_dim * 3) + i);

    float x1 = (x - (0.5 * w));
    float y1 = (y - (0.5 * h));
    float width = w;
    float height = h;

    std::vector<float> conf;
    for (int j = 0; j < num_classes; j++) {
      conf.push_back(*(results_data + (out_dim * (4 + j)) + i));
    }

    std::vector<float>::iterator ind_max_conf;
    ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
    int max_index = distance(std::begin(conf), ind_max_conf);
    float val_max_conf = *max_element(std::begin(conf), std::end(conf));

    bboxes.push_back(cv::Rect(x1, y1, width, height));
    indices.push_back(i);
    scores.push_back(val_max_conf);
    classes.push_back(max_index);
  }

  RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
  cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
  RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

  vision_msgs::msg::Detection2DArray final_detections_arr;
  vision_msgs::msg::Detection2DArray selected_detections_arr;

  for (size_t i = 0; i < indices.size(); i++) {
      // class ids: 
      // 0: 'Unripe tomato',
      // 1: 'Ripe tomato',

      int ind = indices[i];
      int class_id = classes.at(ind);

      // Process and publish all detections
      vision_msgs::msg::Detection2D detection;

      geometry_msgs::msg::Pose center;
      geometry_msgs::msg::Point position;
      geometry_msgs::msg::Quaternion orientation;

      // 2D object Bbox
      vision_msgs::msg::BoundingBox2D bbox;
      float scale_factor = 1.3; // Example scaling factor, you can adjust this value
      // if (class_id == selected_target_id_) {
      //   scale_factor = 1.0;
      // }

      // Original width and height
      float w = bboxes[ind].width;
      float h = bboxes[ind].height;

      // Calculate the scaled width and height
      float scaled_w = w * scale_factor;
      float scaled_h = h * scale_factor;

      // Keep the center the same, adjust the bounding box size
      float x_center = bboxes[ind].x + (0.5 * w);
      float y_center = bboxes[ind].y + (0.5 * h) - 80;
      // int y_shift = static_cast<int>(184 * 640 / 848);
      // float y_center = bboxes[ind].y + (0.5 * h) - y_shift;

      detection.bbox.center.position.x = x_center;
      detection.bbox.center.position.y = y_center;
      detection.bbox.size_x = scaled_w;
      detection.bbox.size_y = scaled_h;

      // Class probabilities
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = std::to_string(classes.at(ind));
      hyp.hypothesis.score = scores.at(ind);
      detection.results.push_back(hyp);

      detection.header.stamp.sec = msg.GetTimestampSeconds();
      detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

      final_detections_arr.detections.push_back(detection);
      

      // Only process and publish detections with the selected target id
      if (class_id == selected_target_id_) {
          selected_detections_arr.detections.push_back(detection);

          // Pass selected detection to YOLOKF
          yolo_kf_->set_rate(fs_);
          yolo_kf_->process_detection(detection);

          // Get KF estimated detection
          vision_msgs::msg::Detection2D kf_estimated_detection = yolo_kf_->get_kf_estimated_detection();

          // Publish the KF estimated detection
          vision_msgs::msg::Detection2DArray kf_estimated_detections_arr;
          kf_estimated_detections_arr.header = detection.header;
          kf_estimated_detections_arr.detections.push_back(kf_estimated_detection);
          selected_target_kf_pub_->publish(kf_estimated_detections_arr);
      }
  }


  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  pub_->publish(final_detections_arr);


  if (!selected_detections_arr.detections.empty()) {
    selected_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
    selected_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
    // selected_detections_arr.header.frame_id = "d435_color_optical_frame";
    selected_target_pub_->publish(selected_detections_arr);
  }





}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)












// ////////// YOLO w/ KF and timer
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

// #include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

// #include <chrono>
// #include <cuda_runtime.h>
// #include <algorithm>
// #include <cmath>
// #include <iostream>
// #include <vector>

// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


// #include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/dnn.hpp>
// #include <opencv4/opencv2/dnn/dnn.hpp>

// #include "std_msgs/msg/bool.hpp"  // Add this for the Bool message type
// #include "std_msgs/msg/int8.hpp"
// #include "vision_msgs/msg/detection2_d_array.hpp"
// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// // Include YOLOKF
// #include "yolo_kf/yolo_kf.hpp"

// using namespace std::chrono_literals;


// namespace nvidia
// {
// namespace isaac_ros
// {
// namespace yolov8
// {
// YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
// : rclcpp::Node("yolov8_decoder_node", options),
//   nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
//         nvidia::isaac_ros::nitros::NitrosTensorListView>>(
//       this,
//       "tensor_sub",
//       nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
//       std::bind(&YoloV8DecoderNode::InputCallback, this,
//       std::placeholders::_1))},
//   pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
//       "detections_output", 50)},
//   selected_target_pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
//       "selected_target_output", 50)},
//   selected_target_kf_pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
//     "selected_target_kf_output", 50)},
//   tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
//   confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
//   nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
//   start_detection_flag_{false}, // Initialise detection flag to false
//   selected_target_id_{-1}, // Initialise selected target id to -1 (none of the classes)
//   yolo_kf_{std::make_shared<YOLOKF>(this)} // Instantiate YOLOKF
// {
//   // Add a subscriber to control detection start/stop and select target
//   fsm_flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(
//       "/fsm_flag/start_detection",
//       10, std::bind(&YoloV8DecoderNode::FsmFlagCallback, this, std::placeholders::_1));
    
//   fsm_select_target_sub_ = this->create_subscription<std_msgs::msg::Int8>(
//       "/fsm_flag/select_target",
//       10, std::bind(&YoloV8DecoderNode::FsmSelectTargetCallback, this, std::placeholders::_1));

//   // Create a timer for the EKF iteration
//   ekf_timer_ = this->create_wall_timer(
//       std::chrono::milliseconds(static_cast<int>(1000.0 / yolo_kf_->get_fs())),
//       std::bind(&YoloV8DecoderNode::kf_timer_callback, this)
//   );
// }

// YoloV8DecoderNode::~YoloV8DecoderNode() = default;

// // Callback for the FSM flag and selected target
// void YoloV8DecoderNode::FsmFlagCallback(const std_msgs::msg::Bool::SharedPtr msg)
// {
//   start_detection_flag_ = msg->data;
//   // RCLCPP_INFO(this->get_logger(), "Detection flag updated: %s", start_detection_flag_ ? "true" : "false");
// }

// void YoloV8DecoderNode::FsmSelectTargetCallback(const std_msgs::msg::Int8::SharedPtr msg)
// {
//   selected_target_id_ = msg->data;
//   RCLCPP_INFO(this->get_logger(), "Selected target updated: %d", selected_target_id_);
// }

// // Implement the timer callback
// void YoloV8DecoderNode::kf_timer_callback() {
//     // Perform EKF iteration
//     yolo_kf_->ekf_iteration();

//     // Get the current estimated detection
//     vision_msgs::msg::Detection2D estimated_detection = yolo_kf_->get_current_estimated_detection();

//     // Publish the KF estimated detection
//     vision_msgs::msg::Detection2DArray kf_estimated_detections_arr;
//     kf_estimated_detections_arr.header.stamp = this->now();
//     kf_estimated_detections_arr.detections.push_back(estimated_detection);
//     selected_target_kf_pub_->publish(kf_estimated_detections_arr);
// }

// void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
// {
//   // Check if detection is allowed
//   if (!start_detection_flag_) {
//     // RCLCPP_INFO(this->get_logger(), "Detection is currently disabled, skipping InputCallback.");
//     return;
//   }

//   auto tensor = msg.GetNamedTensor(tensor_name_);
//   size_t buffer_size{tensor.GetTensorSize()};
//   std::vector<float> results_vector{};
//   results_vector.resize(buffer_size);
//   cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

//   std::vector<cv::Rect> bboxes;
//   std::vector<float> scores;
//   std::vector<int> indices;
//   std::vector<int> classes;

//   //  Output dimensions = [1, 5+4, 8400]
//   // class ids: 
//       // 0: "btn",
//       // 1: "gv",
//       // 2: "ov",
//       // 3: "rpb",
//       // 4: "tg",  
//   int num_classes = 5;
//   int out_dim = 8400;
//   float * results_data = reinterpret_cast<float *>(results_vector.data());

//   for (int i = 0; i < out_dim; i++) {
//     float x = *(results_data + i);
//     float y = *(results_data + (out_dim * 1) + i);
//     float w = *(results_data + (out_dim * 2) + i);
//     float h = *(results_data + (out_dim * 3) + i);

//     float x1 = (x - (0.5 * w));
//     float y1 = (y - (0.5 * h));
//     float width = w;
//     float height = h;

//     std::vector<float> conf;
//     for (int j = 0; j < num_classes; j++) {
//       conf.push_back(*(results_data + (out_dim * (4 + j)) + i));
//     }

//     std::vector<float>::iterator ind_max_conf;
//     ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
//     int max_index = distance(std::begin(conf), ind_max_conf);
//     float val_max_conf = *max_element(std::begin(conf), std::end(conf));

//     bboxes.push_back(cv::Rect(x1, y1, width, height));
//     indices.push_back(i);
//     scores.push_back(val_max_conf);
//     classes.push_back(max_index);
//   }

//   RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
//   cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
//   RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

//   vision_msgs::msg::Detection2DArray final_detections_arr;
//   vision_msgs::msg::Detection2DArray selected_detections_arr;

//   for (size_t i = 0; i < indices.size(); i++) {
//       // class ids: 
//         // 0: "btn",
//         // 1: "gv",
//         // 2: "ov",
//         // 3: "rpb",
//         // 4: "tg",   

//       int ind = indices[i];
//       int class_id = classes.at(ind);

//       // Process and publish all detections
//       vision_msgs::msg::Detection2D detection;

//       geometry_msgs::msg::Pose center;
//       geometry_msgs::msg::Point position;
//       geometry_msgs::msg::Quaternion orientation;

//       // 2D object Bbox
//       vision_msgs::msg::BoundingBox2D bbox;
//       float scale_factor = 1.0; // Example scaling factor, you can adjust this value

//       // Original width and height
//       float w = bboxes[ind].width;
//       float h = bboxes[ind].height;

//       // Calculate the scaled width and height
//       float scaled_w = w * scale_factor;
//       float scaled_h = h * scale_factor;

//       // Keep the center the same, adjust the bounding box size
//       float x_center = bboxes[ind].x + (0.5 * w);
//       float y_center = bboxes[ind].y + (0.5 * h) - 80;
//       // int y_shift = static_cast<int>(184 * 640 / 848);
//       // float y_center = bboxes[ind].y + (0.5 * h) - y_shift;

//       detection.bbox.center.position.x = x_center;
//       detection.bbox.center.position.y = y_center;
//       detection.bbox.size_x = scaled_w;
//       detection.bbox.size_y = scaled_h;

//       // Class probabilities
//       vision_msgs::msg::ObjectHypothesisWithPose hyp;
//       hyp.hypothesis.class_id = std::to_string(classes.at(ind));
//       hyp.hypothesis.score = scores.at(ind);
//       detection.results.push_back(hyp);

//       detection.header.stamp.sec = msg.GetTimestampSeconds();
//       detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

//       final_detections_arr.detections.push_back(detection);
      

//       // Only process and publish detections with the selected target id
//       if (class_id == selected_target_id_) {
//           selected_detections_arr.detections.push_back(detection);

//           // Pass selected detection to YOLOKF
//           yolo_kf_->process_detection(detection);

//           // // Get KF estimated detection
//           // vision_msgs::msg::Detection2D kf_estimated_detection = yolo_kf_->get_kf_estimated_detection();

//           // // Publish the KF estimated detection
//           // vision_msgs::msg::Detection2DArray kf_estimated_detections_arr;
//           // kf_estimated_detections_arr.header = detection.header;
//           // kf_estimated_detections_arr.detections.push_back(kf_estimated_detection);
//           // selected_target_kf_pub_->publish(kf_estimated_detections_arr);
//       }
//   }


//   final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
//   final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
//   pub_->publish(final_detections_arr);


//   // selected_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
//   // selected_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
//   // selected_target_pub_->publish(selected_detections_arr);

//   if (!selected_detections_arr.detections.empty()) {
//     selected_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
//     selected_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
//     // selected_detections_arr.header.frame_id = "d435_color_optical_frame";
//     selected_target_pub_->publish(selected_detections_arr);
//   }





// }

// }  // namespace yolov8
// }  // namespace isaac_ros
// }  // namespace nvidia

// // Register as component
// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)










// //////// YOLO w/o KF
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

// #include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

// #include <cuda_runtime.h>
// #include <algorithm>
// #include <cmath>
// #include <iostream>
// #include <vector>

// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


// #include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/dnn.hpp>
// #include <opencv4/opencv2/dnn/dnn.hpp>

// #include "std_msgs/msg/bool.hpp"  // Add this for the Bool message type
// #include "std_msgs/msg/int8.hpp"
// #include "vision_msgs/msg/detection2_d_array.hpp"
// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"


// namespace nvidia
// {
// namespace isaac_ros
// {
// namespace yolov8
// {
// YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
// : rclcpp::Node("yolov8_decoder_node", options),
//   nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
//         nvidia::isaac_ros::nitros::NitrosTensorListView>>(
//       this,
//       "tensor_sub",
//       nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
//       std::bind(&YoloV8DecoderNode::InputCallback, this,
//       std::placeholders::_1))},
//   pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
//       "detections_output", 50)},
//   selected_target_pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
//       "selected_target_output", 50)},
//   tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
//   confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
//   nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
//   start_detection_flag_{false}, // Initialise detection flag to false
//   selected_target_id_{-1} // Initialise selected target id to -1 (none of the classes)
// {
//   // Add a subscriber to control detection start/stop and select target
//   fsm_flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(
//       "/fsm_flag/start_detection",
//       10, std::bind(&YoloV8DecoderNode::FsmFlagCallback, this, std::placeholders::_1));
    
//   fsm_select_target_sub_ = this->create_subscription<std_msgs::msg::Int8>(
//       "/fsm_flag/select_target",
//       10, std::bind(&YoloV8DecoderNode::FsmSelectTargetCallback, this, std::placeholders::_1));
// }

// YoloV8DecoderNode::~YoloV8DecoderNode() = default;

// // Callback for the FSM flag and selected target
// void YoloV8DecoderNode::FsmFlagCallback(const std_msgs::msg::Bool::SharedPtr msg)
// {
//   start_detection_flag_ = msg->data;
//   // RCLCPP_INFO(this->get_logger(), "Detection flag updated: %s", start_detection_flag_ ? "true" : "false");
// }

// void YoloV8DecoderNode::FsmSelectTargetCallback(const std_msgs::msg::Int8::SharedPtr msg)
// {
//   selected_target_id_ = msg->data;
//   RCLCPP_INFO(this->get_logger(), "Selected target updated: %d", selected_target_id_);
// }

// void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
// {
//   // Check if detection is allowed
//   if (!start_detection_flag_) {
//     // RCLCPP_INFO(this->get_logger(), "Detection is currently disabled, skipping InputCallback.");
//     return;
//   }

//   auto tensor = msg.GetNamedTensor(tensor_name_);
//   size_t buffer_size{tensor.GetTensorSize()};
//   std::vector<float> results_vector{};
//   results_vector.resize(buffer_size);
//   cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

//   std::vector<cv::Rect> bboxes;
//   std::vector<float> scores;
//   std::vector<int> indices;
//   std::vector<int> classes;

//   //  Output dimensions = [1, 5+4, 8400]
//   // class ids: 
//       // 0: "btn",
//       // 1: "gv",
//       // 2: "ov",
//       // 3: "rpb",
//       // 4: "tg",  
//   int num_classes = 5;
//   int out_dim = 8400;
//   float * results_data = reinterpret_cast<float *>(results_vector.data());

//   for (int i = 0; i < out_dim; i++) {
//     float x = *(results_data + i);
//     float y = *(results_data + (out_dim * 1) + i);
//     float w = *(results_data + (out_dim * 2) + i);
//     float h = *(results_data + (out_dim * 3) + i);

//     float x1 = (x - (0.5 * w));
//     float y1 = (y - (0.5 * h));
//     float width = w;
//     float height = h;

//     std::vector<float> conf;
//     for (int j = 0; j < num_classes; j++) {
//       conf.push_back(*(results_data + (out_dim * (4 + j)) + i));
//     }

//     std::vector<float>::iterator ind_max_conf;
//     ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
//     int max_index = distance(std::begin(conf), ind_max_conf);
//     float val_max_conf = *max_element(std::begin(conf), std::end(conf));

//     bboxes.push_back(cv::Rect(x1, y1, width, height));
//     indices.push_back(i);
//     scores.push_back(val_max_conf);
//     classes.push_back(max_index);
//   }

//   RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
//   cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
//   RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

//   vision_msgs::msg::Detection2DArray final_detections_arr;
//   vision_msgs::msg::Detection2DArray selected_detections_arr;

//   for (size_t i = 0; i < indices.size(); i++) {
//       // class ids: 
//         // 0: "btn",
//         // 1: "gv",
//         // 2: "ov",
//         // 3: "rpb",
//         // 4: "tg",   

//       int ind = indices[i];
//       int class_id = classes.at(ind);

//       // Process and publish all detections
//       vision_msgs::msg::Detection2D detection;

//       geometry_msgs::msg::Pose center;
//       geometry_msgs::msg::Point position;
//       geometry_msgs::msg::Quaternion orientation;

//       // 2D object Bbox
//       vision_msgs::msg::BoundingBox2D bbox;
//       float scale_factor = 1.0; // Example scaling factor, you can adjust this value

//       // Original width and height
//       float w = bboxes[ind].width;
//       float h = bboxes[ind].height;

//       // Calculate the scaled width and height
//       float scaled_w = w * scale_factor;
//       float scaled_h = h * scale_factor;

//       // Keep the center the same, adjust the bounding box size
//       float x_center = bboxes[ind].x + (0.5 * w);
//       float y_center = bboxes[ind].y + (0.5 * h) - 80;
//       // int y_shift = static_cast<int>(184 * 640 / 848);
//       // float y_center = bboxes[ind].y + (0.5 * h) - y_shift;

//       detection.bbox.center.position.x = x_center;
//       detection.bbox.center.position.y = y_center;
//       detection.bbox.size_x = scaled_w;
//       detection.bbox.size_y = scaled_h;

//       // Class probabilities
//       vision_msgs::msg::ObjectHypothesisWithPose hyp;
//       hyp.hypothesis.class_id = std::to_string(classes.at(ind));
//       hyp.hypothesis.score = scores.at(ind);
//       detection.results.push_back(hyp);

//       detection.header.stamp.sec = msg.GetTimestampSeconds();
//       detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

//       final_detections_arr.detections.push_back(detection);
      

//       // Only process and publish detections with the selected target id
//       if (class_id == selected_target_id_) {
//           vision_msgs::msg::Detection2D detection;

//           geometry_msgs::msg::Pose center;
//           geometry_msgs::msg::Point position;
//           geometry_msgs::msg::Quaternion orientation;

//           // 2D object Bbox
//           vision_msgs::msg::BoundingBox2D bbox;
//           float scale_factor = 1.0; // 1.0, 1.5, 2.0
//           // 1.5 for rpb
//           // 1.0 for ov

//           // Original width and height
//           float w = bboxes[ind].width;
//           float h = bboxes[ind].height;

//           // Calculate the scaled width and height
//           float scaled_w = w * scale_factor;
//           float scaled_h = h * scale_factor;

//           // Keep the center the same, adjust the bounding box size
//           float x_center = bboxes[ind].x + (0.5 * w);
//           float y_center = bboxes[ind].y + (0.5 * h) - 80;
//           // int y_shift = static_cast<int>(184 * 640 / 848);
//           // float y_center = bboxes[ind].y + (0.5 * h) - y_shift;

//           detection.bbox.center.position.x = x_center;
//           detection.bbox.center.position.y = y_center;
//           detection.bbox.size_x = scaled_w;
//           detection.bbox.size_y = scaled_h;

//           // Class probabilities
//           vision_msgs::msg::ObjectHypothesisWithPose hyp;
//           hyp.hypothesis.class_id = std::to_string(class_id);
//           hyp.hypothesis.score = scores.at(ind);
//           detection.results.push_back(hyp);

//           detection.header.stamp.sec = msg.GetTimestampSeconds();
//           detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

//           selected_detections_arr.detections.push_back(detection);
//       }
//   }


//   final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
//   final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
//   pub_->publish(final_detections_arr);

//   if (!selected_detections_arr.detections.empty()) {
//     selected_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
//     selected_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
//     // selected_detections_arr.header.frame_id = "d435_color_optical_frame";
//     selected_target_pub_->publish(selected_detections_arr);
//   }






// }

// }  // namespace yolov8
// }  // namespace isaac_ros
// }  // namespace nvidia

// // Register as component
// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)











































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

// #include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

// #include <cuda_runtime.h>
// #include <algorithm>
// #include <cmath>
// #include <iostream>
// #include <vector>

// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


// #include <opencv4/opencv2/opencv.hpp>
// #include <opencv4/opencv2/dnn.hpp>
// #include <opencv4/opencv2/dnn/dnn.hpp>

// #include "vision_msgs/msg/detection2_d_array.hpp"
// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"


// namespace nvidia
// {
// namespace isaac_ros
// {
// namespace yolov8
// {
// YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
// : rclcpp::Node("yolov8_decoder_node", options),
//   nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
//         nvidia::isaac_ros::nitros::NitrosTensorListView>>(
//       this,
//       "tensor_sub",
//       nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
//       std::bind(&YoloV8DecoderNode::InputCallback, this,
//       std::placeholders::_1))},
//   pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
//       "detections_output", 50)},
//   tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
//   confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
//   nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)}
// {}

// YoloV8DecoderNode::~YoloV8DecoderNode() = default;

// void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
// {
//   auto tensor = msg.GetNamedTensor(tensor_name_);
//   size_t buffer_size{tensor.GetTensorSize()};
//   std::vector<float> results_vector{};
//   results_vector.resize(buffer_size);
//   cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

//   std::vector<cv::Rect> bboxes;
//   std::vector<float> scores;
//   std::vector<int> indices;
//   std::vector<int> classes;

//   // //  Output dimensions = [1, 84, 8400]
//   // int num_classes = 80;
//   //  Output dimensions = [1, 5+4, 8400]
//   int num_classes = 5;
//   int out_dim = 8400;
//   float * results_data = reinterpret_cast<float *>(results_vector.data());

//   for (int i = 0; i < out_dim; i++) {
//     float x = *(results_data + i);
//     float y = *(results_data + (out_dim * 1) + i);
//     float w = *(results_data + (out_dim * 2) + i);
//     float h = *(results_data + (out_dim * 3) + i);

//     float x1 = (x - (0.5 * w));
//     float y1 = (y - (0.5 * h));
//     float width = w;
//     float height = h;

//     std::vector<float> conf;
//     for (int j = 0; j < num_classes; j++) {
//       conf.push_back(*(results_data + (out_dim * (4 + j)) + i));
//     }

//     std::vector<float>::iterator ind_max_conf;
//     ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
//     int max_index = distance(std::begin(conf), ind_max_conf);
//     float val_max_conf = *max_element(std::begin(conf), std::end(conf));

//     bboxes.push_back(cv::Rect(x1, y1, width, height));
//     indices.push_back(i);
//     scores.push_back(val_max_conf);
//     classes.push_back(max_index);
//   }

//   RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
//   cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
//   RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

//   vision_msgs::msg::Detection2DArray final_detections_arr;

//   // for (size_t i = 0; i < indices.size(); i++) {
//   //   int ind = indices[i];
//   //   vision_msgs::msg::Detection2D detection;

//   //   geometry_msgs::msg::Pose center;
//   //   geometry_msgs::msg::Point position;
//   //   geometry_msgs::msg::Quaternion orientation;

//   //   // 2D object Bbox
//   //   vision_msgs::msg::BoundingBox2D bbox;
//   //   float scale_factor = 2.0; // Example scaling factor, you can adjust this value

//   //   // Original width and height
//   //   float w = bboxes[ind].width;
//   //   float h = bboxes[ind].height;

//   //   // Calculate the scaled width and height
//   //   float scaled_w = w * scale_factor;
//   //   float scaled_h = h * scale_factor;

//   //   // Keep the center the same, adjust the bounding box size
//   //   float x_center = bboxes[ind].x + (0.5 * w);
//   //   float y_center = bboxes[ind].y + (0.5 * h);

//   //   detection.bbox.center.position.x = x_center;
//   //   detection.bbox.center.position.y = y_center;
//   //   detection.bbox.size_x = scaled_w;
//   //   detection.bbox.size_y = scaled_h;



//   //   // Class probabilities
//   //   vision_msgs::msg::ObjectHypothesisWithPose hyp;
//   //   hyp.hypothesis.class_id = std::to_string(classes.at(ind));
//   //   hyp.hypothesis.score = scores.at(ind);
//   //   detection.results.push_back(hyp);

//   //   detection.header.stamp.sec = msg.GetTimestampSeconds();
//   //   detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

//   //   final_detections_arr.detections.push_back(detection);
//   // }

//   for (size_t i = 0; i < indices.size(); i++) {
//       int ind = indices[i];
//       int class_id = classes.at(ind);

//       // Only process and publish detections with the desired class_id
//       // class ids: 
//       // 0: "btn",
//       // 1: "gv",
//       // 2: "ov",
//       // 3: "rpb",
//       // 4: "tg",   
//       if (class_id == 3) {
//           vision_msgs::msg::Detection2D detection;

//           geometry_msgs::msg::Pose center;
//           geometry_msgs::msg::Point position;
//           geometry_msgs::msg::Quaternion orientation;

//           // 2D object Bbox
//           vision_msgs::msg::BoundingBox2D bbox;
//           float scale_factor = 1.0; // 1.0, 1.5, 2.0
//           // 1.5 for rpb
//           // 1.0 for ov

//           // Original width and height
//           float w = bboxes[ind].width;
//           float h = bboxes[ind].height;

//           // Calculate the scaled width and height
//           float scaled_w = w * scale_factor;
//           float scaled_h = h * scale_factor;

//           // Keep the center the same, adjust the bounding box size
//           float x_center = bboxes[ind].x + (0.5 * w);
//           float y_center = bboxes[ind].y + (0.5 * h) - 80;
//           // int y_shift = static_cast<int>(184 * 640 / 848);
//           // float y_center = bboxes[ind].y + (0.5 * h) - y_shift;

//           detection.bbox.center.position.x = x_center;
//           detection.bbox.center.position.y = y_center;
//           detection.bbox.size_x = scaled_w;
//           detection.bbox.size_y = scaled_h;

//           // Class probabilities
//           vision_msgs::msg::ObjectHypothesisWithPose hyp;
//           hyp.hypothesis.class_id = std::to_string(class_id);
//           hyp.hypothesis.score = scores.at(ind);
//           detection.results.push_back(hyp);

//           detection.header.stamp.sec = msg.GetTimestampSeconds();
//           detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

//           final_detections_arr.detections.push_back(detection);
//       }
//   }


//   final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
//   final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
//   pub_->publish(final_detections_arr);
// }

// }  // namespace yolov8
// }  // namespace isaac_ros
// }  // namespace nvidia

// // Register as component
// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)




















