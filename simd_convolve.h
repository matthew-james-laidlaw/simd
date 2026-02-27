#pragma once

#include <cstdint>
#include <immintrin.h>

namespace simd
{

inline auto ConvolveScalar(int32_t const* src, int32_t* dst, int32_t const* kernel, size_t height, size_t width, size_t kernel_height, size_t kernel_width) -> void
{
	size_t half_kernel_height = kernel_height / 2;
	size_t half_kernel_width = kernel_width / 2;

	for (size_t y = 0; y < height; ++y)
	{
		for (size_t x = 0; x < width; ++x)
		{
			int32_t sum = 0;
			for (ptrdiff_t dy = 0; dy < static_cast<ptrdiff_t>(kernel_height); ++dy)
			{
				for (ptrdiff_t dx = 0; dx < static_cast<ptrdiff_t>(kernel_width); ++dx)
				{
					ptrdiff_t sy = static_cast<ptrdiff_t>(y) + (dy - half_kernel_height);
					ptrdiff_t sx = static_cast<ptrdiff_t>(x) + (dx - half_kernel_width);

					if (sy < 0 || sy >= static_cast<ptrdiff_t>(height )|| sx < 0 || sx >= static_cast<ptrdiff_t>(width))
					{
						continue;
					}

					sum += src[sy * width + sx] * kernel[dy * kernel_width + dx];
				}
			}
			dst[y * width + x] = sum;
		}
	}
}

inline auto ConvolveAVX2(int32_t const* src, int32_t* dst, int32_t const* kernel, size_t height, size_t width, size_t kernel_height, size_t kernel_width) -> void
{
	size_t half_kernel_height = kernel_height / 2;
	size_t half_kernel_width = kernel_width / 2;

	const size_t V = 8;

	__m256i k00 = _mm256_set1_epi32(kernel[0]); __m256i k01 = _mm256_set1_epi32(kernel[1]); __m256i k02 = _mm256_set1_epi32(kernel[2]);
	__m256i k10 = _mm256_set1_epi32(kernel[3]); __m256i k11 = _mm256_set1_epi32(kernel[4]); __m256i k12 = _mm256_set1_epi32(kernel[5]);
	__m256i k20 = _mm256_set1_epi32(kernel[6]); __m256i k21 = _mm256_set1_epi32(kernel[7]); __m256i k22 = _mm256_set1_epi32(kernel[8]);

	for (size_t y = 0; y < height - 2; ++y)
	{
		size_t x = 0;
		for (; x <= width - 2 - V; x+= V)
		{
			__m256i a0 = _mm256_setzero_si256();
			__m256i a1 = _mm256_setzero_si256();
			__m256i a2 = _mm256_setzero_si256();

			int32_t const* r0 = src + (y + 0) * width + x;
			int32_t const* r1 = src + (y + 1) * width + x;
			int32_t const* r2 = src + (y + 2) * width + x;

			// process row 0
			__m256i r00 = _mm256_loadu_si256((__m256i*)(r0 + 0));
			__m256i r01 = _mm256_loadu_si256((__m256i*)(r0 + 1));
			__m256i r02 = _mm256_loadu_si256((__m256i*)(r0 + 2));

			a0 = _mm256_add_epi32(a0, _mm256_mullo_epi32(r00, k00));
			a0 = _mm256_add_epi32(a0, _mm256_mullo_epi32(r01, k01));
			a0 = _mm256_add_epi32(a0, _mm256_mullo_epi32(r02, k02));

			// process row 1
			__m256i r10 = _mm256_loadu_si256((__m256i*)(r1 + 0));
			__m256i r11 = _mm256_loadu_si256((__m256i*)(r1 + 1));
			__m256i r12 = _mm256_loadu_si256((__m256i*)(r1 + 2));

			a1 = _mm256_add_epi32(a1, _mm256_mullo_epi32(r10, k10));
			a1 = _mm256_add_epi32(a1, _mm256_mullo_epi32(r11, k11));
			a1 = _mm256_add_epi32(a1, _mm256_mullo_epi32(r12, k12));

			// process row 2
			__m256i r20 = _mm256_loadu_si256((__m256i*)(r2 + 0));
			__m256i r21 = _mm256_loadu_si256((__m256i*)(r2 + 1));
			__m256i r22 = _mm256_loadu_si256((__m256i*)(r2 + 2));

			a2 = _mm256_add_epi32(a2, _mm256_mullo_epi32(r20, k20));
			a2 = _mm256_add_epi32(a2, _mm256_mullo_epi32(r21, k21));
			a2 = _mm256_add_epi32(a2, _mm256_mullo_epi32(r22, k22));

			// merge all accumulators into one
			a0 = _mm256_add_epi32(a0, a1);
			a0 = _mm256_add_epi32(a0, a2);

			// store results
			_mm256_storeu_si256((__m256i*)(dst + y * width + x), a0);
		}

		for (; x < width - 2; ++x)
		{
			int32_t sum = 0;

			for (size_t yy = 0; yy < 3; ++yy)
			{
				for (size_t xx = 0; xx < 3; ++xx)
				{
					sum += src[(y + yy) * width + (x + xx)] * kernel[yy * kernel_width + xx];
				}
			}

			dst[y * width + x] = sum;
		}
	}
}

} // namespace simd
