#include <iostream>
#include <chrono>

#include "FFT.h"



int main() {
	std::vector<FFT::complex> signal = FFT::generate_cos_signal(1 << 24, 1, 1, FFT::PI / 2);
	
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


	std::cin.get();
}