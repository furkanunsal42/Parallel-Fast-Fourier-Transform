#include "FFT.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>

#include <functional>
#include <thread>

double FFT::magnitude(complex a)
{
	return (a.r * a.r + a.i * a.i);
}

FFT::complex FFT::add(complex a, complex b) {
	return complex(a.r + b.r, a.i + b.i);
}

FFT::complex FFT::add(complex a, double b) {
	return complex(a.r + b, a.i);
}

FFT::complex FFT::add(double a, complex b) {
	return complex(a + b.r, b.i);
}

FFT::complex FFT::mult(complex a, complex b) {
	return complex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

FFT::complex FFT::mult(complex a, double b) {
	return complex(a.r * b, a.i * b);
}

FFT::complex FFT::mult(double a, complex b) {
	return complex(b.r * a, b.i * a);
}

FFT::complex FFT::polar(double magnitude, double phase) {
	return FFT::complex(std::cos(phase) * magnitude, std::sin(phase) * magnitude);
}

std::vector<FFT::complex> FFT::read_signal_from_file(const std::string& filepath) {
	std::fstream file(filepath, std::ios::in);
	std::vector<FFT::complex> signal;
	std::string word;
	while (!file.eof()) {
		file >> word;
		signal.push_back(FFT::complex(std::stof(word), 0));
	}
	return signal;
}

std::vector<FFT::complex> FFT::generate_cos_signal(int size, double amplitude, double frequency, double phase)
{
	std::vector<FFT::complex> signal(size);
	for (int i = 0; i < size; i++) {
		signal[i].r = std::cos((double)i / size * frequency * FFT::PI * 2 + phase) * amplitude;
		signal[i].i = 0;
	}
	return signal;
}

int FFT::_floor_log2(int n)
{
	int log2_size = 0;
	for (size_t temp = n; temp > 1U; temp >>= 1)
		log2_size++;
	return log2_size;
}

size_t FFT::_reverse_bits(size_t x, int n) {
	size_t result = 0;
	for (int i = 0; i < n; i++, x >>= 1)
		result = (result << 1) | (x & 1U);
	return result;
}

void FFT::fft_radix2(std::vector<FFT::complex>& vec) {
	// Length variables
	size_t n = vec.size();
	int log2_size = _floor_log2(n);

	if (static_cast<size_t>(1U) << log2_size != n)
		throw std::domain_error("Length is not a power of 2");

	std::vector<FFT::complex> exp_table(n / 2);
	size_t i;
	for (i = 0; i < n / 2; i++)
		exp_table[i] = polar(1.0f, -2 * FFT::PI * i / n);

	for (i = 0; i < n; i++) {
		size_t j = _reverse_bits(i, log2_size);
		if (j > i)
			std::swap(vec[i], vec[j]);
	}

	size_t halfsize;
	size_t tablestep;
	FFT::complex temp;
	size_t size, j, k;
	std::vector<FFT::complex> temp_buffer = vec;

	for (size = 2; size <= n; size *= 2) {
		halfsize = size / 2;
		tablestep = n / size;
		for (i = 0; i < n; i += size) {
			for (j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
				vec[j + halfsize] = add(temp_buffer[j], mult(-1, mult(temp_buffer[j + halfsize], exp_table[k])));
				vec[j] = add(temp_buffer[j], mult(temp_buffer[j + halfsize], exp_table[k]));
			}
		}
		if (size == n)  // Prevent overflow in 'size *= 2'
			break;
		temp_buffer = vec;
	}
}


void FFT::inverse_fft_radix2(std::vector<FFT::complex>& vec) {
	for (int i = 0; i < vec.size(); i++)
		vec[i].i *= -1;
	FFT::fft_radix2(vec);
	for (int i = 0; i < vec.size(); i++)
		vec[i].i *= -1;

	for (int i = 0; i < vec.size(); i++)
		vec[i] = FFT::complex(vec[i].r / vec.size(), vec[i].i / vec.size());
}

void FFT::parallel_fft_radix2(std::vector<FFT::complex>& vec) {
	
	// organize values into reverse bit ordering
	auto start = std::chrono::system_clock::now();
	_parallel_fft_reverse_bit_order(vec);
	//std::cout << "_parallel_fft_reverse_bit_order took " << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << " seconds" << std::endl;

	// comptue number of steps required by finding log2(size)
	int size = vec.size();
	int log2_size = _floor_log2(size);

	// fft requires the input to be a power of 2 thus if not, throw an error
	if (static_cast<size_t>(1U) << log2_size != size)
		throw std::domain_error("Length is not a power of 2");

	// precompute required complex factors beforehand to reuse in algorithm
	std::vector<FFT::complex> exp_table(size / 2);
	size_t i;
	for (i = 0; i < size / 2; i++)
		exp_table[i] = polar(1.0f, -2 * FFT::PI * i / size);

	// run each step sequancially
	std::vector<FFT::complex> vec_copy = vec;

	for (int i = 0; i < log2_size; i++) {
		// swap the read and write buffers every iteration to avoid rece-condition and unnecessary copying
		std::vector<FFT::complex>& vec_to_read = i % 2 == 0 ? vec : vec_copy;
		std::vector<FFT::complex>& vec_to_write = i % 2 == 0 ? vec_copy : vec;
		
		auto start = std::chrono::system_clock::now();
		_parallel_fft_single_step(vec_to_read, vec_to_write, exp_table, i);
		//std::cout << "_parallel_fft_single_step took " << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << " seconds" << std::endl;
	}

	// if the last step wrote to vec_copy, copy the final result back to original vector
	if ((log2_size - 1) % 2 == 0)
		vec = vec_copy;
}
void FFT::_conjugate_signal_thread_function(std::vector<FFT::complex>& vector, int i, int computation_size_per_thread) {
	for (int computation_index = 0; computation_index < computation_size_per_thread; computation_index++) {
		if (i > vector.size()) return;
		vector[i].i *= -1;
		i++;
	}
}

void FFT::_divide_signal_thread_function(std::vector<FFT::complex>& vector, int i, double divisor, int computation_size_per_thread) {
	for (int computation_index = 0; computation_index < computation_size_per_thread; computation_index++) {
		if (i > vector.size()) return;
		vector[i] = FFT::complex(vector[i].r / divisor, vector[i].i / divisor);
		i++;
	}
}

void FFT::parallel_inverse_fft_radix2(std::vector<FFT::complex>& vec)
{
	std::vector<std::thread> thread_pool;
	int thread_count = PARALLEL_FFT_THREAD_COUNT;
	int computation_size_per_thread = (int)std::ceil((double)vec.size() / thread_count);

	// conjugate the complex values of original signal
	for (int i = 0; i < thread_count; i++)
		thread_pool.push_back(std::thread(FFT::_conjugate_signal_thread_function, std::ref(vec), i * computation_size_per_thread, computation_size_per_thread));

	for (int thread_index = 0; thread_index < thread_pool.size(); thread_index++)
		thread_pool[thread_index].join();
	thread_pool.clear();

	// compute fourier transform
	FFT::parallel_fft_radix2(vec);
	
	// conjugate it again
	for (int i = 0; i < thread_count; i++)
		thread_pool.push_back(std::thread(FFT::_conjugate_signal_thread_function, std::ref(vec), i * computation_size_per_thread, computation_size_per_thread));

	for (int thread_index = 0; thread_index < thread_pool.size(); thread_index++)
		thread_pool[thread_index].join();
	thread_pool.clear();

	// normalize by dividing the signal to sample count
	for (int i = 0; i < thread_count; i++)
		thread_pool.push_back(std::thread(FFT::_divide_signal_thread_function, std::ref(vec), i * computation_size_per_thread, vec.size(), computation_size_per_thread));

	for (int thread_index = 0; thread_index < thread_pool.size(); thread_index++)
		thread_pool[thread_index].join();
	thread_pool.clear();
}

// reverse the bit ordering of given integer eg. 110100 -> 001011
void FFT::_parallel_fft_reverse_bit_order_thread_function(std::vector<FFT::complex>& vec, int index, int log2_size, int computation_size_per_thread) {
	for (int computation_index = 0; computation_index < computation_size_per_thread; computation_index++) {
		size_t reversed_index = FFT::_reverse_bits(index, log2_size);
		if (reversed_index > index)
			std::swap(vec[index], vec[reversed_index]);
		index++;
	}
}

void FFT::_parallel_fft_reverse_bit_order(std::vector<FFT::complex>& vec)
{
	int size = vec.size();
	int log2_size = _floor_log2(size);

	std::vector<std::thread> thread_pool;

	int thread_count = PARALLEL_FFT_THREAD_COUNT;
	int computation_size_per_thread = (int)std::ceil((double)size / thread_count);

	for (int i = 0; i < thread_count; i++) {
		thread_pool.push_back(std::thread(FFT::_parallel_fft_reverse_bit_order_thread_function, std::ref(vec), i * computation_size_per_thread, log2_size, computation_size_per_thread));
	}

	for (int thread_index = 0; thread_index < thread_pool.size(); thread_index++)
		thread_pool[thread_index].join();

}

void FFT::_parallel_fft_single_step_thread_function(const std::vector<FFT::complex>& read_vector, std::vector<FFT::complex>& write_vector, const std::vector<FFT::complex>& exp_table, int i, int size, int computation_size_per_thread) {
	int n = write_vector.size();

	size_t halfsize = size / 2;
	size_t tablestep = n / size;

	for (int computation_index = 0; computation_index < computation_size_per_thread; computation_index++) {
		if (i >= n) return;

		int local_i = (i - (i / size) * size);
		int k = (local_i % halfsize) * tablestep;
		bool is_odd_term = local_i >= halfsize;
		if (is_odd_term) write_vector[i] = add(read_vector[i - halfsize], mult(-1, mult(read_vector[i], exp_table[k])));
		else write_vector[i] = add(read_vector[i], mult(read_vector[i + halfsize], exp_table[k]));
		i++;
	}
}

void FFT::_parallel_fft_single_step(const std::vector<FFT::complex>& read_vector, std::vector<FFT::complex>& write_vector, const std::vector<FFT::complex>& exp_table, int step_index)
{
	if (read_vector.size() != write_vector.size())
		throw("sizes of read and write vectors have to be same");

	int n = read_vector.size();
	int size = 2 << step_index;

	std::vector<std::thread> thread_pool;
	int thread_count = PARALLEL_FFT_THREAD_COUNT;
	int computation_size_per_thread = (int)std::ceil((double)n / thread_count);

	for (int i = 0; i < thread_count; i++) {
		thread_pool.push_back(std::thread(FFT::_parallel_fft_single_step_thread_function, std::ref(read_vector), std::ref(write_vector), std::ref(exp_table), i * computation_size_per_thread, size, computation_size_per_thread));
	}

	for (int thread_index = 0; thread_index < thread_pool.size(); thread_index++) {
		thread_pool[thread_index].join();
	}

}

std::ostream& operator<<(std::ostream& stream, const FFT::complex& complex) {
	return stream << "(" << complex.r << ", " << complex.i << "i)";
}
