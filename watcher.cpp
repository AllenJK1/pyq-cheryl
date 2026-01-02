#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <filesystem>
#include <chrono>
#include <thread>
#include <mutex>
#include <ctime>
#include <set>
#include <algorithm>
#include <iomanip>  // For std::put_time

// Function to convert Unix timestamp to Date and Time format (YYYY-MM-DD HH:MM:SS)
std::string convertUnixToDateTime(long unix_timestamp) {
    std::time_t time = unix_timestamp;
    std::tm* tm_ptr = std::gmtime(&time);  // Use localtime if you need local timezone instead of UTC

    std::stringstream ss;
    ss << std::put_time(tm_ptr, "%Y-%m-%d %H:%M:%S");  // Full date and time format
    return ss.str();
}

// Function to process each asset file
void processAssetFile(const std::string& assetName) {
    std::cout << "Processing " << assetName << "..." << std::endl;

    try {
        // Open the CSV file
        std::ifstream file(assetName + ".csv");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << assetName << ".csv" << std::endl;
            return;
        }

        std::vector<std::vector<std::string>> data;
        std::string line;

        // Read the file content
        bool headerSkipped = false;
        while (std::getline(file, line)) {
            if (!headerSkipped) {
                headerSkipped = true;
                continue;  // Skip header line
            }

            std::stringstream ss(line);
            std::vector<std::string> row;
            std::string cell;

            // Parse the line into columns (comma separated)
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            if (row.size() == 5) {
                data.push_back(row);
            }
        }
        file.close();

        if (data.empty()) {
            std::cerr << "No data found in " << assetName << ".csv" << std::endl;
            return;
        }

        // Sort data by the first column (time)
        std::sort(data.begin(), data.end(), [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
            return std::stol(a[0]) < std::stol(b[0]);
        });

        // Remove duplicates based on the 'time' column
        std::vector<std::vector<std::string>> unique_data;
        std::set<long> seen_times;

        for (const auto& row : data) {
            long time = std::stol(row[0]);
            if (seen_times.find(time) == seen_times.end()) {
                unique_data.push_back(row);
                seen_times.insert(time);
            }
        }

        // Convert Unix timestamps to Date and Time format in the 'time' column
        for (auto& row : unique_data) {
            long time = std::stol(row[0]);
            row[0] = convertUnixToDateTime(time);  // Now converting to full date and time
        }

        // Save the cleaned and sorted data to a new CSV file (overwrite existing file)
        std::ofstream outputFile(assetName + "_sorted.csv");
        if (outputFile.is_open()) {
            outputFile << "time,open,high,low,close\n"; // Write the header

            // Write the data
            for (const auto& row : unique_data) {
                for (size_t i = 0; i < row.size(); ++i) {
                    outputFile << row[i];
                    if (i < row.size() - 1) outputFile << ",";
                }
                outputFile << "\n";
            }
            outputFile.close();
        } else {
            std::cerr << "Error: Could not write to file " << assetName << "_sorted.csv" << std::endl;
        }

        std::cout << "File cleaned, sorted, and saved as '" << assetName << "_sorted.csv'" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error processing " << assetName << ": " << e.what() << std::endl;
    }
}

// Class to handle file watching and processing
class FileWatcher {
public:
    FileWatcher(const std::vector<std::string>& assets) : assetsToMonitor(assets) {}

    void startWatching() {
        std::cout << "Monitoring CSV files for modifications..." << std::endl;

        while (true) {
            for (const auto& asset : assetsToMonitor) {
                std::string filename = asset + ".csv";
                if (std::filesystem::exists(filename)) {
                    std::filesystem::file_time_type ftime = std::filesystem::last_write_time(filename);
                    std::lock_guard<std::mutex> lock(fileMutex);

                    if (fileModificationTimes.find(asset) == fileModificationTimes.end()) {
                        fileModificationTimes[asset] = ftime;
                    }

                    // Check if the file has been modified
                    if (ftime != fileModificationTimes[asset]) {
                        std::cout << "Detected modification in " << filename << ". Processing..." << std::endl;
                        processAssetFile(asset);
                        fileModificationTimes[asset] = ftime;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Poll every 100 ms
        }
    }

private:
    std::vector<std::string> assetsToMonitor;
    std::map<std::string, std::filesystem::file_time_type> fileModificationTimes;
    std::mutex fileMutex;
};

int main() {
    // Hardcode the assets
    std::vector<std::string> assetList = {"CADCHF_otc", "USDDZD_otc", "USDPHP_otc", "USDPKR_otc"};

    // Start file watcher
    FileWatcher fileWatcher(assetList);
    fileWatcher.startWatching();

    return 0;
}
