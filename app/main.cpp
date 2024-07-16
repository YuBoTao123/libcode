#include "range_image.h"
#include "range_tensor.h"
#include <pcl/range_image/range_image.h>

#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/cloud_viewer.h>

#include "utils.hpp"

int main(int argc, char** argv) {
  using namespace point_cloud;

  // load calibration file
  auto calibration_matrix = utils::LoadCalibrationFile::loadFile("../pose.txt");
  std::cout << "calibratin size is " << calibration_matrix.rows() << std::endl;
  std::cout << "calibratin size is " << calibration_matrix.cols() << std::endl;

  // load pcd data
  std::string path = "../1718164051300270080.pcd";
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  if (-1 == pcl::io::loadPCDFile<pcl::PointXYZI>(path, *cloud)) {
    std::cout << "load pcd file error" << std::endl;
    return -1;
  } else {
    std::cout << "points size is " << cloud->width * cloud->height << std::endl;
  }
  RangeImage<pcl::PointCloud<pcl::PointXYZI>> range_img(64, 1024); //height, width (40, 1024)
  range_img.createImageFromPointCloud(*cloud, 15, -25);
  range_img.show();

  std::vector<float> unproj_range;
  std::vector<float> unproj_remissions;
  std::vector<std::string> files;
  std::vector<float> means_values;
  std::vector<float> stds_values;
  auto tensor = range_sensor::processScanMat(cloud, unproj_range, unproj_remissions,
  range_img.getMat("distance"),
  range_img.getMat("point"), range_img.getMat("intensity"), range_img.getMat("mask"),
  files, true, 1, means_values, stds_values);
  return 0;
}