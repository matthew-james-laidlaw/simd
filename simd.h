#pragma once

#include <cstdint>
#include <immintrin.h>

namespace simd
{

struct SSE2
{
	static constexpr size_t kLanes = 4;

	using V = __m128i;

	static auto Load(int32_t const* data) -> V
	{
		return _mm_loadu_si128((V const*)data);
	}

	static auto Store(int32_t* data, V v) -> void
	{
		_mm_storeu_si128((V*)data, v);
	}
};

struct AVX2
{
	static constexpr size_t kLanes = 8;

	using V = __m256i;

	static auto Load(int32_t const* data) -> V
	{
		return _mm256_loadu_si256((V const*)data);
	}

	static auto Store(int32_t* data, V v) -> void
	{
		_mm256_storeu_si256((V*)data, v);
	}
};

template <typename ISA>
struct SimdAdd {};

template <>
struct SimdAdd<SSE2>
{
	using V = SSE2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a + b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm_add_epi32(a, b);
	}
};

template <>
struct SimdAdd<AVX2>
{
	using V = AVX2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a + b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm256_add_epi32(a, b);
	}
};

template <typename ISA>
struct SimdSub {};

template <>
struct SimdSub<SSE2>
{
	using V = SSE2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a - b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm_sub_epi32(a, b);
	}
};

template <>
struct SimdSub<AVX2>
{
	using V = AVX2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a - b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm256_sub_epi32(a, b);
	}
};

template <typename ISA>
struct SimdMul {};

template <>
struct SimdMul<SSE2>
{
	using V = SSE2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a * b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm_mullo_epi32(a, b);
	}
};

template <>
struct SimdMul<AVX2>
{
	using V = AVX2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a * b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm256_mullo_epi32(a, b);
	}
};

template <typename ISA>
struct SimdDiv {};

template <>
struct SimdDiv<SSE2>
{
	using V = SSE2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a / b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm_div_epi32(a, b);
	}
};

template <>
struct SimdDiv<AVX2>
{
	using V = AVX2::V;

	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a / b;
	}

	static auto Process(V a, V b) -> V
	{
		return _mm256_div_epi32(a, b);
	}
};

struct SimdMinBase
{
	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return std::min(a, b);
	}
};

template <typename ISA>
struct SimdMin {};

template <>
struct SimdMin<SSE2> : public SimdMinBase
{
	using V = SSE2::V;

	static auto Identity() -> V
	{
		return _mm_set1_epi32(std::numeric_limits<int32_t>::max());
	}

	static auto Process(V a, V b) -> V
	{
		return _mm_min_epi32(a, b);
	}
};

template <>
struct SimdMin<AVX2> : public SimdMinBase
{
	using V = AVX2::V;

	static auto Identity() -> V
	{
		return _mm256_set1_epi32(std::numeric_limits<int32_t>::max());
	}

	static auto Process(V a, V b) -> V
	{
		return _mm256_min_epi32(a, b);
	}
};

struct SimdMaxBase
{
	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return std::max(a, b);
	}
};

template <typename ISA>
struct SimdMax {};

template <>
struct SimdMax<SSE2> : public SimdMaxBase
{
	using V = SSE2::V;

	static auto Identity() -> V
	{
		return _mm_set1_epi32(std::numeric_limits<int32_t>::min());
	}

	static auto Process(V a, V b) -> V
	{
		return _mm_max_epi32(a, b);
	}
};

template <>
struct SimdMax<AVX2> : public SimdMaxBase
{
	using V = AVX2::V;

	static auto Identity() -> V
	{
		return _mm256_set1_epi32(std::numeric_limits<int32_t>::min());
	}

	static auto Process(V a, V b) -> V
	{
		return _mm256_max_epi32(a, b);
	}
};

struct SimdSumBase
{
	static auto ProcessScalar(int32_t a, int32_t b) -> int32_t
	{
		return a + b;
	}
};

template <typename ISA>
struct SimdSum {};

template <>
struct SimdSum<SSE2> : public SimdSumBase
{
	using V = SSE2::V;

	static auto Identity() -> V
	{
		return _mm_setzero_si128();
	}

	static auto Process(V a, V b) -> V
	{
		return _mm_add_epi32(a, b);
	}
};

template <>
struct SimdSum<AVX2> : public SimdSumBase
{
	using V = AVX2::V;

	static auto Identity() -> V
	{
		return _mm256_setzero_si256();
	}

	static auto Process(V a, V b) -> V
	{
		return _mm256_add_epi32(a, b);
	}
};

struct SimdSquareBase
{
	static auto ProcessScalar(int32_t a) -> int32_t
	{
		return a * a;
	}
};

template <typename ISA>
struct SimdSquare {};

template <>
struct SimdSquare<SSE2> : public SimdSquareBase
{
	using V = SSE2::V;

	static auto Process(V a) -> V
	{
		return _mm_mullo_epi32(a, a);
	}
};

template <>
struct SimdSquare<AVX2> : public SimdSquareBase
{
	using V = AVX2::V;

	static auto Process(V a) -> V
	{
		return _mm256_mullo_epi32(a, a);
	}
};

} // namespace simd
