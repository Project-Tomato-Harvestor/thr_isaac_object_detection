#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This script listens for images and object detections on the image,
# then renders the output boxes on top of the image and publishes
# the result as an image message

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

names = {
    0: 'Unripe tomato',
    1: 'Ripe tomato',
}



# detection_info = {
#     0: {'name': 'Unripe tomato', 'color': (0, 0, 255)},  # Red
#     1: {'name': 'Ripe tomato', 'color': (0, 255, 0)}     # Green
# }

class Yolov8Visualizer(Node):
    QUEUE_SIZE = 10
    color = (0, 255, 0)
    bbox_thickness = 2

    def __init__(self):
        super().__init__('yolov8_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._processed_image_pub = self.create_publisher(
            Image, 'yolov8_processed_image',  self.QUEUE_SIZE)

        self._detections_subscription = message_filters.Subscriber(
            self,
            Detection2DArray,
            '/d435/detections_output')
        self._image_subscription = message_filters.Subscriber(
            self,
            Image,
            '/d435/color/image_raw')

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [self._detections_subscription, self._image_subscription],
            self.QUEUE_SIZE)

        self.time_synchronizer.registerCallback(self.detections_callback)

    def detections_callback(self, detections_msg, img_msg):
        txt_color = (255, 0, 255)
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg)
        for detection in detections_msg.detections:
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y

#         label = detection_info[class_id]['name']
#         color = detection_info[class_id]['color']

            label = names[int(detection.results[0].hypothesis.class_id)]
            conf_score = detection.results[0].hypothesis.score
            label = f'{label} {conf_score:.2f}'

            min_pt = (round(center_x - (width / 2.0)),
                      round(center_y - (height / 2.0)))
            max_pt = (round(center_x + (width / 2.0)),
                      round(center_y + (height / 2.0)))

            lw = max(round((img_msg.height + img_msg.width) / 2 * 0.003), 2)  # line width
            tf = max(lw - 1, 1)  # font thickness
            # text width, height
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = min_pt[1] - h >= 3

            cv2.rectangle(cv2_img, min_pt, max_pt,
                          self.color, self.bbox_thickness)
            cv2.putText(cv2_img, label, (min_pt[0], min_pt[1]-2 if outside else min_pt[1]+h+2),
                        0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

        processed_img = self._bridge.cv2_to_imgmsg(
            cv2_img, encoding=img_msg.encoding)
        self._processed_image_pub.publish(processed_img)

    # def detections_callback(self, detections_msg, img_msg):
    #     # 定義影像訊息及座標訊息
    #     processed_image_msgs = Image()
    #     cluster_2Dcoord_msgs = Int32MultiArray()
    #     cluster_3Dcoord_msgs = Float64MultiArray()
    #     topmost_3Dcoord_msgs = Float64MultiArray()
    #     # Temp variables
    #     bbox_x_mean = []
    #     bbox_y_mean = []
    #     bbox_average_centre = [0, 0]
    #     topmost_bbox = None  # Initialize variable to hold the topmost bbox
        
    #     # Convert image message to OpenCV image
    #     cv2_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")

    #     # 處理每個偵測框以繪製 bounding box 和標籤
    #     for detection in detections_msg.detections:
    #         center_x = detection.bbox.center.position.x
    #         center_y = detection.bbox.center.position.y
    #         width = detection.bbox.size_x
    #         height = detection.bbox.size_y

    #         # Check if it's the topmost bbox
    #         if topmost_bbox is None or center_y < topmost_bbox.bbox.center.position.y:
    #             topmost_bbox = detection

    #         bbox_x_mean.append(center_x)
    #         bbox_y_mean.append(center_y)
    #         `
    #         # Get the detected class id
    #         class_id = int(detection.results[0].hypothesis.class_id)
    #         label = detection_info[class_id]['name']
    #         color = detection_info[class_id]['color']
    #         conf_score = detection.results[0].hypothesis.score

    #         min_pt = (round(center_x - (width / 2.0)),
    #                   round(center_y - (height / 2.0)))
    #         max_pt = (round(center_x + (width / 2.0)),
    #                   round(center_y + (height / 2.0)))

    #         lw = max(round((img_msg.height + img_msg.width) / 2 * 0.003), 2)  # line width
    #         tf = max(lw - 1, 1)  # font thickness
    #         # text width, height
    #         w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
    #         outside = min_pt[1] - h >= 3

    #         cv2.rectangle(cv2_img, min_pt, max_pt, color, self.bbox_thickness)
    #         cv2.putText(cv2_img, label, (min_pt[0], min_pt[1]-2 if outside else min_pt[1]+h+2),
    #                     0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

    #     # Calculate the average centre of all bounding boxes
    #     bbox_average_centre[0] = sum(bbox_x_mean) / len(bbox_x_mean)
    #     bbox_average_centre[1] = sum(bbox_y_mean) / len(bbox_y_mean)

    #     # 發布座標資訊
    #     self.cluster_2d_coord_pub.publish(cluster_2Dcoord_msgs)
    #     self.cluster_3d_coord_pub.publish(cluster_3Dcoord_msgs)
    #     self.topmost_3d_coord_pub.publish(topmost_3Dcoord_msgs)

    #     # 發布處理後的影像
    #     processed_image_msgs = self._bridge.cv2_to_imgmsg(processed_image, encoding=img_msg.encoding)
    #     self._processed_image_pub.publish(processed_image_msgs)


def main():
    rclpy.init()
    rclpy.spin(Yolov8Visualizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
