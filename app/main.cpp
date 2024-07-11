#include "range_image.h"
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

  return 0;
}