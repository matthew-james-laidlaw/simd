#pragma once

#pragma once

#include "simd.h"

namespace simd
{

/**
 * @brief Apply an elementwise operation to an array, storing the result in a separate array.
 *
 * @tparam ISA The SIMD instruction set to use.
 * @tparam Op The elementwise operation, e.g., square.
 *
 * @param src The input array.
 * @param dst The output array.
 * @param n The number of elements in the array.
 */
template <typename ISA, template <typename> typename Op>
auto Map(int32_t const* src, int32_t* dst, size_t n) -> void
{
	using V = typename ISA::V;
	constexpr size_t lanes = ISA::kLanes;

	size_t i = 0;

	// while at least `lanes` elements remain, process a full vector
	for (; i + lanes <= n; i += lanes)
	{
		V v1 = ISA::Load(src + i);
		V v2 = Op<ISA>::Process(v1);
		ISA::Store(dst + i, v2);
	}

	// linearly process any remaining elements that do not fill a vector
	for (; i < n; ++i)
	{
		dst[i] = Op<ISA>::ProcessScalar(src[i]);
	}
}

template <typename ISA>
auto Square(int32_t const* src, int32_t* dst, size_t n) -> void
{
	Map<ISA, SimdSquare>(src, dst, n);
}

} // namespace simd
