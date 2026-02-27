#pragma once

#include "simd.h.h"

#include <array>

namespace simd
{

/**
 * @brief Accumulate an operation across an entire array.
 * 
 * @tparam ISA The SIMD instruction set to use.
 * @tparam Op The accumulation operation, e.g., sum.
 * @tparam AccumulatorCount How many SIMD registers to use for the main loop. This is a tuning parameter.
 * 
 * @param src The input array.
 * @param n The number of elements in the array.
 * 
 * @returns The accumulated value.
 */
template <typename ISA, template <typename> typename Op, size_t AccumulatorCount>
auto Reduce(int32_t const* src, const size_t n) -> int32_t
{
	using V = typename ISA::V;
	constexpr size_t lanes = ISA::kLanes;
	constexpr size_t chunk = AccumulatorCount * lanes;

	// set each accumulator to the given operation's identity element
	std::array<V, AccumulatorCount> accumulators;
	for (size_t i = 0; i < AccumulatorCount; ++i)
	{
		accumulators[i] = Op<ISA>::Identity();
	}

	// while at least `chunk` elements remain, process them using the full bank of accumulators
	size_t i = 0;
	for (; i + chunk <= n; i += chunk)
	{
		for (size_t j = 0; j < AccumulatorCount; ++j)
		{
			V v = ISA::Load(src + i + j * lanes);
			accumulators[j] = Op<ISA>::Process(accumulators[j], v);
		}
	}

	// merge all accumulators into one
	V accumulator = accumulators[0];
	for (size_t j = 1; j < AccumulatorCount; ++j)
	{
		accumulator = Op<ISA>::Process(accumulator, accumulators[j]);
	}

	// while at least one full vector's worth of data remains, process one vector at a time
	for (; i + lanes <= n; i += lanes)
	{
		V v = ISA::Load(src + i);
		accumulator = Op<ISA>::Process(accumulator, v);
	}

	// horizontally accumulate the final SIMD vector into a single value
	std::array<int32_t, lanes> horizontal_accumulator;
	ISA::Store(horizontal_accumulator.data(), accumulator);
	int32_t result = horizontal_accumulator[0];
	for (size_t j = 1; j < lanes; ++j)
	{
		result = Op<ISA>::ProcessScalar(result, horizontal_accumulator[j]);
	}

	// process any remaining elements that do do not fill a full SIMD vector
	for (; i < n; ++i)
	{
		result = Op<ISA>::ProcessScalar(result, src[i]);
	}
		
	return result;
}

static constexpr size_t kAccumulatorCount = 4;

template <typename ISA>
auto Min(int32_t const* src, size_t n) -> int32_t
{
	return Reduce<ISA, SimdMin, kAccumulatorCount>(src, n);
}

template <typename ISA>
auto Max(int32_t const* src, size_t n) -> int32_t
{
	return Reduce<ISA, SimdMax, kAccumulatorCount>(src, n);
}

template <typename ISA>
auto Sum(int32_t const* src, size_t n) -> int32_t
{
	return Reduce<ISA, SimdSum, kAccumulatorCount>(src, n);
}

} // namespace simd
