#pragma once

#include <vector>
#include <string>

// Fast Fourier Transform
namespace OMP_FFT {

	const double PI = 3.14154965f;
	const int PARALLEL_OMP_FFT_THREAD_COUNT = 2;

	// complex numbers
	struct complex {
		complex(double r = 0, double i = 0) :
			r(r), i(i) {}
		double r;
		double i;
	};

	double magnitude(complex a);

	complex add(complex a, complex b);
	complex add(complex a, double b);
	complex add(double a, complex b);

	complex mult(complex a, complex b);
	complex mult(complex a, double b);
	complex mult(double a, complex b);

	complex polar(double magnitude, double phase);

	// test signal generation
	std::vector<complex> read_signal_from_file(const std::string& filepath);
	std::vector<OMP_FFT::complex> generate_cos_signal(int size, double amplitude, double frequency, double phase);

	// fft
	void fft_radix2(std::vector<complex>& vec);
	void inverse_fft_radix2(std::vector<complex >& vec);

	// helper functions for fft
	int _floor_log2(int n);
	size_t _reverse_bits(size_t x, int n);

	// parallel fft
	void parallel_fft_radix2(std::vector<complex>& vec);
	void parallel_inverse_fft_radix2(std::vector<OMP_FFT::complex>& vec);

	// helper functions for parallel fft
	void _parallel_fft_reverse_bit_order(std::vector<OMP_FFT::complex>& vec);
	void _parallel_fft_single_step(const std::vector<OMP_FFT::complex>& read_vector, std::vector<OMP_FFT::complex>& write_vector, const std::vector<OMP_FFT::complex>& exp_table, int step_index);
}

std::ostream& operator<<(std::ostream& stream, const OMP_FFT::complex& complex);