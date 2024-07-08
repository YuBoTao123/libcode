#pragma once

#include <iostream>
// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <cmath>
// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "type_def.h"

#include <typeinfo>
#include <typeindex>

#define CHECK_TYPE(x, type) (std::type_index(typeid(type)) == std::type_index(typeid(x)))

namespace point_cloud {

template<typename P>
class RangeImage {
  public:
    RangeImage(int H, int W) :
        H_(H), W_(W),
        distance_mat_(H, W, CV_64FC1, cv::Scalar::all(-1.0)),
        intensity_mat_(H, W, CV_64FC1, cv::Scalar::all(-1.0)),
        point_mat_(H, W, CV_64FC3, cv::Scalar::all(-1.0)),
        index_mat_(H, W, CV_32SC1, cv::Scalar::all(-1.0)),
        mask_mat_(H, W, CV_8UC1, cv::Scalar::all(0)) {}
    virtual ~RangeImage() = default;

    // 
    void createImageFromPointCloud(const P& cloud, double fov_up, double fov_down) {
        double fov_up_rad = fov_up * 180 / M_PI;
        double fov_down_rad = fov_down * 180 / M_PI;
        auto fov_rad = std::fabs(fov_up_rad) + std::fabs(fov_down_rad);

        int proj_x_index(-1), proj_y_index(-1);
        double intensity(0.0), distance(0.0);

        for (int i = 0; i < cloud.points.size(); i++) {
            const auto& point_iter = cloud.points.at(i);
            distance = std::sqrt( point_iter.x * point_iter.x + point_iter.y * point_iter.y +
                                    point_iter.z * point_iter.z);
            double pitch = asin(point_iter.z / distance);
            double yaw = atan2(point_iter.y, point_iter.x); // ?
            double proj_x = (yaw / M_PI + 1.0) * 0.5;
            double proj_y = (pitch + std::fabs(fov_down_rad)) / fov_rad;
            proj_x_index = static_cast<int>(std::floor(proj_x));
            proj_x_index = std::min(proj_x_index, W_ - 1);
            proj_x_index = std::max(0, proj_x_index);

            proj_y_index = static_cast<int>(std::floor(proj_y));
            proj_y_index = std::min(proj_y_index, H_ - 1);
            proj_y_index = (std::max(0, proj_y_index));

            // check intensity type
            if (CHECK_TYPE(point_iter.intensity, float) || CHECK_TYPE(point_iter.intensity, double)) {
                intensity = point_iter.intensity;
            } else {
                intensity = point_iter.intensity / 255.0; // default scale
            }
            distance_mat_.at<double>(proj_y_index, proj_x_index) = distance;
            intensity_mat_.at<double>(proj_y_index, proj_x_index) = intensity;
            point_mat_.at<cv::Vec3d>(proj_y_index, proj_x_index) = cv::Vec3d(point_iter.x, point_iter.y, point_iter.z);
            index_mat_.at<int32_t>(proj_y_index, proj_x_index) = i;
            mask_mat_.at<u_char>(proj_y_index, proj_x_index) = 1;
        }
    }

    bool save();

    void show() {
        cv::imshow("distance_mat_", distance_mat_);
        cv::imshow("inte_", intensity_mat_);
        // cv::imshow("point", point_mat_);
        // cv::imshow("index", index_mat_);
        cv::imshow("mask", mask_mat_);
        cv::waitKey(0); 
    }
  
  private:
    int H_;
    int W_;
    cv::Mat distance_mat_;
    cv::Mat intensity_mat_;
    cv::Mat point_mat_;
    cv::Mat index_mat_;
    cv::Mat mask_mat_;
};


}
