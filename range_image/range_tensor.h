#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

constexpr int kMaxPoints = 150000;
constexpr int krows = 64;
constexpr int kcols = 2048;

typedef Eigen::Matrix<double, krows, kcols> DImgMatrix;
typedef Eigen::Matrix<int, krows, kcols> IImgMatrix;
typedef Eigen::Matrix<bool, krows, kcols> BImgMatrix;

namespace range_sensor {

template<typename P>
torch::Tensor processScan(const std::shared_ptr<P>& points,
                 const std::vector<float>& unproj_range, // distance info
                 const std::vector<float>& unproj_remissions, // intensity info
                 const DImgMatrix& proj_range,
                 const DImgMatrix& proj_xyz,
                 const DImgMatrix& proj_remission,
                 const BImgMatrix& proj_mask,
                 const IImgMatrix& proj_x,
                 const IImgMatrix& proj_y,
                 const std::vector<std::string>& residual_files,
                 bool use_residual,
                 int n_input_scans,
                 const std::vector<float>& sensor_img_means,
                 const std::vector<float>& sensor_img_stds) {

    // 获取点的数量
    int unproj_n_points = points->size();

    // 初始化张量
    torch::Tensor unproj_xyz = torch::full({kMaxPoints, 3}, -1.0, torch::kFloat32);
    unproj_xyz.slice(0, 0, unproj_n_points) = torch::from_blob(points->data(), {unproj_n_points, 3}, torch::kFloat32);

    torch::Tensor unproj_range_tensor = torch::full({kMaxPoints}, -1.0, torch::kFloat32);
    unproj_range_tensor.slice(0, 0, unproj_n_points) = torch::from_blob(unproj_range.data(), {unproj_n_points}, torch::kFloat32);

    torch::Tensor unproj_remissions_tensor = torch::full({kMaxPoints}, -1.0, torch::kFloat32);
    unproj_remissions_tensor.slice(0, 0, unproj_n_points) = torch::from_blob(unproj_remissions.data(), {unproj_n_points}, torch::kFloat32);

    // 将 Eigen 矩阵转换为 Torch 张量
    torch::Tensor proj_range_tensor = torch::from_blob(proj_range.data(), {krows, kcols}, torch::kDouble).clone();
    torch::Tensor proj_xyz_tensor = torch::from_blob(proj_xyz.data(), {krows, kcols, 3}, torch::kDouble).clone();
    torch::Tensor proj_remission_tensor = torch::from_blob(proj_remission.data(), {krows, kcols}, torch::kDouble).clone();

    // 归一化处理
    torch::Tensor proj = torch::cat({proj_range_tensor.unsqueeze(0),
                                      proj_xyz_tensor.permute({1, 0, 2}),
                                      proj_remission_tensor.unsqueeze(0)});

    proj = (proj - torch::from_blob(sensor_img_means.data(), {1, 3}).clone().unsqueeze(0)) /
           torch::from_blob(sensor_img_stds.data(), {1, 3}).clone().unsqueeze(0);

    return proj;
}

} // namespace range_sensor
