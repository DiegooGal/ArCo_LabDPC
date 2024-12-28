#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <array>
#include "Configurations.h"  // Define constants like BANDS

using namespace sycl;

// Function to simulate hyperspectral data loading
std::vector<unsigned short> loadHyperspectralBlock(const std::string &filename, int blockSize, int bands) {
    std::vector<unsigned short> data(blockSize * bands, 1);  // Mock data
    return data;
}

void calculateDPC(const std::vector<unsigned short> &imageBlock, 
                  std::array<int, BANDS> &centroid,
                  std::array<double, BANDS> &mean,
                  std::array<double, BANDS> &variance,
                  std::array<int, BANDS> &sumOfSquares,
                  int blockSize) {
    sycl::queue q{sycl::default_selector{}};

    // Create buffers
    sycl::buffer<unsigned short, 1> imgBuffer(imageBlock.data(), sycl::range<1>(blockSize * BANDS));
    sycl::buffer<int, 1> centroidBuffer(centroid.data(), sycl::range<1>(BANDS));
    sycl::buffer<double, 1> meanBuffer(mean.data(), sycl::range<1>(BANDS));
    sycl::buffer<double, 1> varianceBuffer(variance.data(), sycl::range<1>(BANDS));
    sycl::buffer<int, 1> sumOfSquaresBuffer(sumOfSquares.data(), sycl::range<1>(BANDS));

    // Submit task to queue
    q.submit([&](sycl::handler &h) {
        // Accessors
        auto imgAcc = imgBuffer.get_access<sycl::access::mode::read>(h);
        auto centroidAcc = centroidBuffer.get_access<sycl::access::mode::write>(h);
        auto meanAcc = meanBuffer.get_access<sycl::access::mode::write>(h);
        auto varianceAcc = varianceBuffer.get_access<sycl::access::mode::write>(h);
        auto sumOfSquaresAcc = sumOfSquaresBuffer.get_access<sycl::access::mode::write>(h);

        // Kernel range
        sycl::range<1> workRange(BANDS);

        // Parallel computation
        h.parallel_for(workRange, [=](sycl::id<1> band) {
            int sum = 0;
            int sumSq = 0;
            double sumForVariance = 0;
            for (int i = 0; i < blockSize; ++i) {
                int pixelValue = imgAcc[i * BANDS + band];
                sum += pixelValue;
                sumSq += pixelValue * pixelValue;
                sumForVariance += pixelValue;
            }

            centroidAcc[band] = sum / blockSize; // Centroid calculation
            meanAcc[band] = sumForVariance / blockSize; // Mean
            varianceAcc[band] = sumSq / blockSize - meanAcc[band] * meanAcc[band]; // Variance
            sumOfSquaresAcc[band] = sumSq; // Sum of Squares
        });
    });

    // Wait and retrieve results
    q.wait();
    sycl::host_accessor centroidAcc(centroidBuffer, sycl::access::mode::read);
    sycl::host_accessor meanAcc(meanBuffer, sycl::access::mode::read);
    sycl::host_accessor varianceAcc(varianceBuffer, sycl::access::mode::read);
    sycl::host_accessor sumOfSquaresAcc(sumOfSquaresBuffer, sycl::access::mode::read);

    // Assign computed results to the output arrays
    for (int i = 0; i < BANDS; ++i) {
        centroid[i] = centroidAcc[i];
        mean[i] = meanAcc[i];
        variance[i] = varianceAcc[i];
        sumOfSquares[i] = sumOfSquaresAcc[i];
    }
}

int main() {
    const std::string filename = "hyperspectral_data_block.dat";
    const int blockSize = 100;  // Example block size
    const int bands = BANDS;    // Defined in Configurations.h

    // Load a block of hyperspectral data
    std::vector<unsigned short> imageBlock = loadHyperspectralBlock(filename, blockSize, bands);

    // Prepare arrays for the centroid, mean, variance, and sum of squares
    std::array<int, BANDS> centroid = {0};
    std::array<double, BANDS> mean = {0.0};
    std::array<double, BANDS> variance = {0.0};
    std::array<int, BANDS> sumOfSquares = {0};

    // Calculate the centroid, mean, variance, and sum of squares using DPC
    calculateDPC(imageBlock, centroid, mean, variance, sumOfSquares, blockSize);

    // Output results
    for (int i = 0; i < bands; ++i) {
        std::cout << "Centroid[" << i << "] = " << centroid[i] << std::endl;
        std::cout << "Mean[" << i << "] = " << mean[i] << std::endl;
        std::cout << "Variance[" << i << "] = " << variance[i] << std::endl;
        std::cout << "SumOfSquares[" << i << "] = " << sumOfSquares[i] << std::endl;
    }

    return 0;
}
