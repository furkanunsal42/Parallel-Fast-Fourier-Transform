#pragma once

#include <vector>
#include <string>

// Fast Fourier Transform
namespace FFT {

	const float PI = 3.14154965f;
	const int PARALLEL_FFT_THREAD_COUNT = 1;

	// complex numbers
	struct complex {
		complex(float r = 0, float i = 0) :
			r(r), i(i) {}
		float r;
		float i;
	};

	float magnitude(complex a);

	complex add(complex a, complex b);
	complex add(complex a, float b);
	complex add(float a, complex b);

	complex mult(complex a, complex b);
	complex mult(complex a, float b);
	complex mult(float a, complex b);

	complex polar(float magnitude, float phase);

	// test signal generation
	std::vector<complex> read_signal_from_file(const std::string& filepath);
	std::vector<FFT::complex> generate_cos_signal(int size, float amplitude, float frequency, float phase);

	// fft
	void fft_radix2(std::vector<complex>& vec);
	void inverse_fft_radix2(std::vector<complex >& vec);

	// helper functions for fft
	int _floor_log2(int n);
	size_t _reverse_bits(size_t x, int n);

	// parallel fft
	void parallel_fft_radix2(std::vector<complex>& vec);
	void parallel_inverse_fft_radix2(std::vector<FFT::complex>& vec);

	// helper functions for parallel fft
	void _parallel_fft_reverse_bit_order(std::vector<FFT::complex>& vec);
	void _parallel_fft_single_step(const std::vector<FFT::complex>& read_vector, std::vector<FFT::complex>& write_vector, const std::vector<FFT::complex>& exp_table, int step_index);

	// multithreaded functions
	void _parallel_fft_reverse_bit_order_thread_function(std::vector<FFT::complex>& vec, int index, int log2_size, int computation_size_per_thread);
	void _parallel_fft_single_step_thread_function(const std::vector<FFT::complex>& read_vector, std::vector<FFT::complex>& write_vector, const std::vector<FFT::complex>& exp_table, int i, int size, int computation_size_per_thread);
	void _conjugate_signal_thread_function(std::vector<FFT::complex>& vector, int i, int computation_size_per_thread);
	void _divide_signal_thread_function(std::vector<FFT::complex>& vector, int i, float divisor, int computation_size_per_thread);
}

std::ostream& operator<<(std::ostream& stream, const FFT::complex& complex);