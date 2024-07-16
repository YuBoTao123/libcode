#include <iostream>
#include <vector>
#include <cmath> // for std::sqrt
#include <torch/torch.h> // Assuming you have PyTorch C++ API available

constexpr int kMaxPoints = 150000;

template<typename P>
void processScan(const std::shared_ptr<P>& points,
                 const std::vector<float>& unproj_range, // distance info
                 const std::vector<float>& unproj_remissions, // intensity info
                 const std::vector<float>& proj_range,
                 const std::vector<float>& proj_xyz,
                 const std::vector<float>& proj_remission,
                 const std::vector<bool>& proj_mask,
                 const std::vector<int>& proj_x,
                 const std::vector<int>& proj_y,
                 const std::vector<std::string>& residual_files,
                 bool use_residual,
                 int n_input_scans,
                 const std::vector<float>& sensor_img_means,
                 const std::vector<float>& sensor_img_stds) {

    // Make tensors for uncompressed data
    int unproj_n_points = points.size();
    torch::Tensor unproj_xyz = torch::full({kMaxPoints, 3}, -1.0, torch::kFloat);
    std::memcpy(unproj_xyz.data_ptr<float>(), points.data(), unproj_n_points * sizeof(float));
    // 一维数组
    torch::Tensor unproj_range = torch::full({kMaxPoints}, -1.0, torch::kFloat);
    std::memcpy(unproj_range.data_ptr<float>(), unproj_range.data(), unproj_n_points * sizeof(float));

    torch::Tensor unproj_remissions = torch::full({kMaxPoints}, -1.0, torch::kFloat);
    std::memcpy(unproj_remissions.data_ptr<float>(), unproj_remissions.data(), unproj_n_points * sizeof(float));

    // Get points and labels proj 是二维图像信息: 深度，点信息，反射率，掩码
    torch::Tensor proj_range = torch::from_blob((void*)proj_range.data(), {1, static_cast<long>(proj_range.size())}, torch::kFloat).clone();
    torch::Tensor proj_xyz = torch::from_blob((void*)proj_xyz.data(), {3, static_cast<long>(proj_xyz.size() / 3)}, torch::kFloat).clone();
    torch::Tensor proj_remission = torch::from_blob((void*)proj_remission.data(), {1, static_cast<long>(proj_remission.size())}, torch::kFloat).clone();
    torch::Tensor proj_mask = torch::from_blob((void*)proj_mask.data(), {1, static_cast<long>(proj_mask.size())}, torch::kBool).clone();
    // image index
    torch::Tensor proj_x = torch::full({kMaxPoints}, -1, torch::kLong);
    std::memcpy(proj_x.data_ptr<long>(), proj_x.data(), unproj_n_points * sizeof(long));

    torch::Tensor proj_y = torch::full({kMaxPoints}, -1, torch::kLong);
    std::memcpy(proj_y.data_ptr<long>(), proj_y.data(), unproj_n_points * sizeof(long));

    torch::Tensor proj_full = torch::cat({proj_range.unsqueeze(0).clone(),
                                          proj_xyz.clone().permute({2, 0, 1}),
                                          proj_remission.unsqueeze(0).clone()}, 0);

    // Assuming sensor_img_means and sensor_img_stds are std::vector<float>
    torch::Tensor sensor_img_means_tensor = torch::from_blob((void*)sensor_img_means.data(), {3}, torch::kFloat).clone();
    torch::Tensor sensor_img_stds_tensor = torch::from_blob((void*)sensor_img_stds.data(), {3}, torch::kFloat).clone();

    proj_full = (proj_full - sensor_img_means_tensor.view({3, 1, 1})) / sensor_img_stds_tensor.view({3, 1, 1});

    // Add residual channel
    if (use_residual) {
        for (int i = 0; i < n_input_scans; ++i) {
            std::string residual_file = residual_files[i];
            torch::Tensor proj_residuals = torch::from_file(residual_file).clone();
            proj_full = torch::cat({proj_full, proj_residuals.unsqueeze(0)}, 0);
        }
    }

    proj_full = proj_full * proj_mask.to(torch::kFloat);

    // Get name and sequence
    std::string scan_file = "your_scan_file.bin";
    std::string path_norm = std::filesystem::path(scan_file).lexically_normal();
    std::vector<std::string> path_split;
    std::string delimiter = "/";
    size_t pos = 0;
    std::string token;
    while ((pos = path_norm.find(delimiter)) != std::string::npos) {
        token = path_norm.substr(0, pos);
        path_split.push_back(token);
        path_norm.erase(0, pos + delimiter.length());
    }
    std::string path_seq = path_split[path_split.size() - 3];
    std::string path_name = path_split[path_split.size() - 1].replace(".bin", ".label");
}
