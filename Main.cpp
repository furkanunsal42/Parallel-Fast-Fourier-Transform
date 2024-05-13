#include <iostream>
#include <chrono>

#include "OMP_FFT.h"
#include "FFT.h"

int main() {

	for (int size = 1024; size <= 1024 * 1024; size *= 2) {
		std::vector<FFT::complex> signal = FFT::generate_cos_signal(size, 2, 4, FFT::PI / 2);
		std::vector<FFT::complex> signal_copy = signal;

		//std::cout << "original: " << std::endl;
		//for (int i = 0; i < signal.size(); i++) {
		//	std::cout << signal[i] << " ";
		//}
		//std::cout << std::endl;

		auto start = std::chrono::system_clock::now();
		FFT::parallel_fft_radix2(signal);

		//std::cout << "fourier: " << std::endl;
		//for (int i = 0; i < signal.size(); i++) {
		//	std::cout << signal[i] << " ";
		//}
		//std::cout << std::endl;

		FFT::parallel_inverse_fft_radix2(signal);
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "fft and ifft took " << elapsed_seconds.count() << " seconds with " << signal.size() << " elements" << std::endl;

		//std::cout << "inverse: " << std::endl;
		//for (int i = 0; i < signal.size(); i++) {
		//	std::cout << signal[i] << " ";
		//}
		//std::cout << std::endl;


		// compute maximum error between original signal and inverse_fourier(fourier(signal))
		//double max_error = -1 << 16;
		//for (int i = 0; i < signal.size(); i++) {
		//	max_error = std::max(std::abs(signal[i].r - signal_copy[i].r), max_error);
		//}
		//std::cout << "maximum error after taking parallel_fft and parallel_inverse_fft of signal is: " << max_error << std::endl;

	}
	std::cin.get();
}