#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <array>
#include "Configurations.h" // Define constants like BANDS

using namespace sycl;

// Function to simulate hyperspectral data loading
std::vector<unsigned short> loadHyperspectralBlock(const std::string &filename, int blockSize, int bands) {
    std::vector<unsigned short> data(blockSize * bands, 1);  // Mock data
    return data;
}

void calculateCentroidDPC(const std::vector<unsigned short> &imageBlock, std::array<int, BANDS> &centroid, int blockSize) {
    sycl::queue q{sycl::default_selector{}};

    // Create buffers
    sycl::buffer<unsigned short, 1> imgBuffer(imageBlock.data(), sycl::range<1>(blockSize * BANDS));
    sycl::buffer<int, 1> centroidBuffer(centroid.data(), sycl::range<1>(BANDS));

    // Submit task to queue
    q.submit([&](sycl::handler &h) {
        // Accessors
        auto imgAcc = imgBuffer.get_access<sycl::access::mode::read>(h);
        auto centroidAcc = centroidBuffer.get_access<sycl::access::mode::write>(h);

        // Kernel range
        sycl::range<1> workRange(BANDS);

        // Parallel computation
        h.parallel_for(workRange, [=](sycl::id<1> band) {
            int sum = 0;
            for (int i = 0; i < blockSize; ++i) {
                sum += imgAcc[i * BANDS + band];
            }
            centroidAcc[band] = sum / blockSize; // Average computation
        });
    });

    // Wait and retrieve results
    q.wait();
    sycl::host_accessor resultAcc(centroidBuffer, sycl::access::mode::read);
    for (int i = 0; i < BANDS; ++i) {
        centroid[i] = resultAcc[i];
    }
}

int main() {
    const std::string filename = "hyperspectral_data_block.dat";
    const int blockSize = 100;  // Example block size
    const int bands = BANDS;    // Defined in Configurations.h

    // Load a block of hyperspectral data
    std::vector<unsigned short> imageBlock = loadHyperspectralBlock(filename, blockSize, bands);

    // Prepare the centroid array
    std::array<int, BANDS> centroid = {0};

    // Calculate the centroid using DPC
    calculateCentroidDPC(imageBlock, centroid, blockSize);

    // Output the centroid
    for (int i = 0; i < bands; ++i) {
        std::cout << "Centroid[" << i << "] = " << centroid[i] << std::endl;
    }

    return 0;
}
