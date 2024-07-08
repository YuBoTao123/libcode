#include "range_image.h"

int main(int argc, char** argv) {
  using namespace point_cloud;
  pcl::PointCloud<pcl::PointXYZI> pointCloud;
    // Generate the data
    for (float y=-0.5f; y<=0.5f; y+=0.01f) {
        for (float z=-0.5f; z<=0.5f; z+=0.01f) {
        pcl::PointXYZI point;
        point.x = 2.0f - y;
        point.y = y;
        point.z = z;
        point.intensity = 0.1;
        pointCloud.push_back(point);
        }
    }
    pointCloud.width = pointCloud.size();
    pointCloud.height = 1;


    RangeImage<pcl::PointCloud<pcl::PointXYZI>> range_img(1024, 2048);
    range_img.createImageFromPointCloud(pointCloud, 25, -5);
    range_img.show();

    return 0;
}