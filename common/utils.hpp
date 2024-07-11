#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <Eigen/Core>

namespace utils {

// time(int) ns, double: x, y, z, qw, qx, qy, qz
class LoadCalibrationFile {
public:
    static Eigen::MatrixXd loadFile(const std::string& file, bool header = true) {
        std::ifstream ifs(file);
        if (!ifs.is_open()) return Eigen::MatrixXd();
        std::string header_line;
        if (header) {
            std::getline(ifs, header_line);
        }
        std::vector<std::vector<double>> calibration_data;
        std::string row_str;
        while (getline(ifs, row_str)) {
            std::stringstream ss(row_str);
            std::vector<double> data;
            double value;
            while (ss >> value) {
                data.emplace_back(value);
                if (ss.peek() == ',') ss.ignore();
            }
            calibration_data.emplace_back(data);
        }
        ifs.close();
        if (calibration_data.empty()) return Eigen::MatrixXd();

        Eigen::Map<Eigen::MatrixXd> eigen_matrix(calibration_data[0].data(), calibration_data.size(), 
            calibration_data[0].size());
        return eigen_matrix;
    }
};

}