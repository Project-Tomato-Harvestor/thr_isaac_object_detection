#ifndef YOLO_KF_HPP
#define YOLO_KF_HPP

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "ekf/ekf.hpp"  // Include your EKF implementation

using Eigen::MatrixXd;
using Eigen::VectorXd;

class YOLOKF {
public:
    explicit YOLOKF(rclcpp::Node* node)
        : node_(node),
          fs_(20.0),
          Ts_(1.0 / fs_),
          data_ready_(false),
          last_frame_id_("camera_frame")
    {
        bbox_meas_ = VectorXd::Zero(4);  // x, y, w, h
        initTranslationEKF();
    }

    // Method to set the sampling rate
    void set_rate(double fs) {
        fs_ = fs;
        Ts_ = 1.0 / fs_;
    }

    // Method to process selected detection
    void process_detection(const vision_msgs::msg::Detection2D &detection) {
        if (detection.bbox.size_x == 0 || detection.bbox.size_y == 0) {
            RCLCPP_WARN(node_->get_logger(), "[yolo_kf.hpp] [process_detection()] Invalid detection size, skipping this iteration.");
            return;
        }
        bbox_meas_ << detection.bbox.center.position.x, detection.bbox.center.position.y,
                      detection.bbox.size_x, detection.bbox.size_y;

        selected_class_id_ = detection.results[0].hypothesis.class_id;
        selected_target_score_ = detection.results[0].hypothesis.score;
        last_frame_id_ = detection.header.frame_id;

        data_ready_ = true;
        ekf_iteration();
    }


    // Method to get the KF estimated detection
    vision_msgs::msg::Detection2D get_kf_estimated_detection() {
        vision_msgs::msg::Detection2D estimated_detection;
        estimated_detection.header.stamp = node_->now();
        estimated_detection.header.frame_id = last_frame_id_;

        // Use EKF state for estimated position
        VectorXd estimated_state = trans_ekf_->getState();

        estimated_detection.bbox.center.position.x = static_cast<float>(estimated_state(0));
        estimated_detection.bbox.center.position.y = static_cast<float>(estimated_state(1));
        estimated_detection.bbox.size_x = static_cast<float>(bbox_meas_(2));  // Keep size from measurement
        estimated_detection.bbox.size_y = static_cast<float>(bbox_meas_(3));

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = selected_class_id_;
        hyp.hypothesis.score = static_cast<float>(selected_target_score_);
        estimated_detection.results.push_back(hyp);

        return estimated_detection;
    }

private:
    void initTranslationEKF() {
        int dim_x = 4;  // State dimension (x, y, xdot, ydot)
        int dim_z = 2;  // Measurement dimension (x, y)
        trans_ekf_ = std::make_shared<ExtendedKalmanFilter>(dim_x, dim_z);

        VectorXd initial_state(4);
        initial_state << 0.0, 0.0, 0.0, 0.0;
        trans_ekf_->setState(initial_state);

        // State transition matrix F
        trans_ekf_->setF(trans_jacobian_F());

        // Measurement matrix H
        MatrixXd H(dim_z, dim_x);
        H << 1, 0, 0, 0,
             0, 1, 0, 0;
        trans_ekf_->setH(H);

        // Process noise covariance Q
        MatrixXd Q = MatrixXd::Identity(dim_x, dim_x) * 1e-3;
        trans_ekf_->setProcessNoise(Q);

        // Measurement noise covariance R
        MatrixXd R = MatrixXd::Identity(dim_z, dim_z) * 1e-2;
        trans_ekf_->setMeasurementNoise(R);
    }

    void ekf_iteration() {
        if (!data_ready_) {
            RCLCPP_WARN(node_->get_logger(), "No data ready for KF, skipping this iteration.");
            return;
        }

        // Prediction step
        trans_ekf_->predict(trans_jacobian_F(), trans_fx());

        // Update step
        VectorXd trans_meas(2);
        trans_meas << bbox_meas_(0), bbox_meas_(1);
        trans_ekf_->update(trans_meas, trans_ekf_->getH(), trans_hx());

        data_ready_ = false;  // Reset the flag
    }

    // EKF functions
    VectorXd trans_fx() {
        // State transition function
        VectorXd x = trans_ekf_->getState();
        VectorXd x_pred(4);
        x_pred(0) = x(0) + Ts_ * x(2);
        x_pred(1) = x(1) + Ts_ * x(3);
        x_pred(2) = x(2);
        x_pred(3) = x(3);
        return x_pred;
    }

    MatrixXd trans_jacobian_F() {
        MatrixXd F(4, 4);
        F << 1, 0, Ts_, 0,
             0, 1, 0, Ts_,
             0, 0, 1, 0,
             0, 0, 0, 1;
        return F;
    }

    VectorXd trans_hx() {
        // Measurement function
        VectorXd x = trans_ekf_->getState();
        VectorXd z_pred(2);
        z_pred(0) = x(0);
        z_pred(1) = x(1);
        return z_pred;
    }

    // Member variables
    rclcpp::Node* node_;
    double fs_, Ts_;
    bool data_ready_;
    std::string last_frame_id_;

    std::shared_ptr<ExtendedKalmanFilter> trans_ekf_;
    VectorXd bbox_meas_;
    std::string selected_class_id_;
    double selected_target_score_;
};

#endif // YOLO_KF_HPP

















// #ifndef YOLO_KF_HPP
// #define YOLO_KF_HPP

// #include <rclcpp/rclcpp.hpp>
// #include <Eigen/Dense>
// #include "vision_msgs/msg/detection2_d_array.hpp"
// #include "ekf/ekf.hpp"

// using Eigen::MatrixXd;
// using Eigen::VectorXd;

// class YOLOKF {
// public:
//     explicit YOLOKF(rclcpp::Node* node)
//         : node_(node),
//           fs_(60.0),
//           Ts_(1.0 / fs_),
//           data_ready_(false),
//           last_frame_id_("camera_frame")
//     {
//         bbox_meas_ = VectorXd::Zero(4);  // x, y, w, h
//         initTranslationEKF();
//     }

//     void process_detection(const vision_msgs::msg::Detection2D &detection) {
//         if (detection.bbox.size_x == 0 || detection.bbox.size_y == 0) {
//             RCLCPP_WARN(node_->get_logger(), "[yolo_kf.hpp] [process_detection()] Invalid detection size, skipping this iteration.");
//             return;
//         }
//         bbox_meas_ << detection.bbox.center.position.x, detection.bbox.center.position.y,
//                       detection.bbox.size_x, detection.bbox.size_y;

//         selected_class_id_ = detection.results[0].hypothesis.class_id;
//         selected_target_score_ = detection.results[0].hypothesis.score;
//         last_frame_id_ = detection.header.frame_id;

//         data_ready_ = true;
//     }

//     void ekf_iteration() {
//         // Always perform the prediction step
//         trans_ekf_->predict(trans_jacobian_F(), trans_fx());

//         if (data_ready_) {
//             // Update step with the latest measurement
//             VectorXd trans_meas(2);
//             trans_meas << bbox_meas_(0), bbox_meas_(1);
//             trans_ekf_->update(trans_meas, trans_ekf_->getH(), trans_hx());

//             data_ready_ = false;  // Reset the flag
//         }
//     }

//     vision_msgs::msg::Detection2D get_current_estimated_detection() {
//         vision_msgs::msg::Detection2D estimated_detection;
//         estimated_detection.header.stamp = node_->now();
//         estimated_detection.header.frame_id = last_frame_id_;

//         // Use EKF state for estimated position
//         VectorXd estimated_state = trans_ekf_->getState();

//         estimated_detection.bbox.center.position.x = static_cast<float>(estimated_state(0));
//         estimated_detection.bbox.center.position.y = static_cast<float>(estimated_state(1));
//         estimated_detection.bbox.size_x = static_cast<float>(bbox_meas_(2));  // Keep size from last measurement
//         estimated_detection.bbox.size_y = static_cast<float>(bbox_meas_(3));

//         vision_msgs::msg::ObjectHypothesisWithPose hyp;
//         hyp.hypothesis.class_id = selected_class_id_;
//         hyp.hypothesis.score = static_cast<float>(selected_target_score_);
//         estimated_detection.results.push_back(hyp);

//         return estimated_detection;
//     }

//     double get_fs() const {
//         return fs_;
//     }

// private:
//     void initTranslationEKF() {
//         int dim_x = 4;  // State dimension (x, y, xdot, ydot)
//         int dim_z = 2;  // Measurement dimension (x, y)
//         trans_ekf_ = std::make_shared<ExtendedKalmanFilter>(dim_x, dim_z);

//         VectorXd initial_state(4);
//         initial_state << 0.0, 0.0, 0.0, 0.0;
//         trans_ekf_->setState(initial_state);

//         // State transition matrix F
//         trans_ekf_->setF(trans_jacobian_F());

//         // Measurement matrix H
//         MatrixXd H(dim_z, dim_x);
//         H << 1, 0, 0, 0,
//              0, 1, 0, 0;
//         trans_ekf_->setH(H);

//         // Process noise covariance Q
//         MatrixXd Q = MatrixXd::Identity(dim_x, dim_x) * 1e-4;
//         trans_ekf_->setProcessNoise(Q);

//         // Measurement noise covariance R
//         MatrixXd R = MatrixXd::Identity(dim_z, dim_z) * 1e-2;
//         trans_ekf_->setMeasurementNoise(R);
//     }

//     // EKF functions
//     VectorXd trans_fx() {
//         // State transition function
//         VectorXd x = trans_ekf_->getState();
//         VectorXd x_pred(4);
//         x_pred(0) = x(0) + Ts_ * x(2);
//         x_pred(1) = x(1) + Ts_ * x(3);
//         x_pred(2) = x(2);
//         x_pred(3) = x(3);
//         return x_pred;
//     }

//     MatrixXd trans_jacobian_F() {
//         MatrixXd F(4, 4);
//         F << 1, 0, Ts_, 0,
//              0, 1, 0, Ts_,
//              0, 0, 1, 0,
//              0, 0, 0, 1;
//         return F;
//     }

//     VectorXd trans_hx() {
//         // Measurement function
//         VectorXd x = trans_ekf_->getState();
//         VectorXd z_pred(2);
//         z_pred(0) = x(0);
//         z_pred(1) = x(1);
//         return z_pred;
//     }

//     // Member variables
//     rclcpp::Node* node_;
//     double fs_, Ts_;
//     bool data_ready_;
//     std::string last_frame_id_;

//     std::shared_ptr<ExtendedKalmanFilter> trans_ekf_;
//     VectorXd bbox_meas_;
//     std::string selected_class_id_;
//     double selected_target_score_;
// };

// #endif // YOLO_KF_HPP
