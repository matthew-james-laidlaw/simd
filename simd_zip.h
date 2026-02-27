#pragma once

#include "simd.h.h"

namespace simd
{

/**
 * @brief Apply an elementwise operation to two arrays, storing the result in a separate array.
 * 
 * @tparam ISA The SIMD instruction set to use.
 * @tparam Op The elementwise operation, e.g., addition.
 * 
 * @param src1 The left input array.
 * @param src2 The right input array.
 * @param dst The output array.
 * @param n The number of elements in each array.
 */
template <typename ISA, template <typename> typename Op>
auto Zip(int32_t const* src1, int32_t const* src2, int32_t* dst, size_t n) -> void
{
	using V = typename ISA::V;
	constexpr size_t lanes = ISA::kLanes;

	size_t i = 0;

	// while at least `lanes` elements remain, process a full vector
	for (; i + lanes <= n; i += lanes)
	{
		V v1 = ISA::Load(src1 + i);
		V v2 = ISA::Load(src2 + i);
		V v3 = Op<ISA>::Process(v1, v2);
		ISA::Store(dst + i, v3);
	}

	// linearly process any remaining elements that do not fill a vector
	for (; i < n; ++i)
	{
		dst[i] = Op<ISA>::ProcessScalar(src1[i], src2[i]);
	}
}

template <typename ISA>
auto Add(int32_t const* src1, int32_t const* src2, int32_t* dst, size_t n) -> void
{
	Zip<ISA, SimdAdd>(src1, src2, dst, n);
}

template <typename ISA>
auto Sub(int32_t const* src1, int32_t const* src2, int32_t* dst, size_t n) -> void
{
	Zip<ISA, SimdSub>(src1, src2, dst, n);
}

template <typename ISA>
auto Mul(int32_t const* src1, int32_t const* src2, int32_t* dst, size_t n) -> void
{
	Zip<ISA, SimdMul>(src1, src2, dst, n);
}

template <typename ISA>
auto Div(int32_t const* src1, int32_t const* src2, int32_t* dst, size_t n) -> void
{
	Zip<ISA, SimdDiv>(src1, src2, dst, n);
}

} // namespace simd
