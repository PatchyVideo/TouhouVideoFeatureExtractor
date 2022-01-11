#pragma once

// CUDA Driver API
#include <cuda.h>
// CUDA Runtime API
#include <cuda_runtime_api.h>


#include <utility>
#include <algorithm>
#include <memory>
#include <vector>

#include <exception>
#include <stdexcept>
#include <cassert>
#include <chrono>
#include <string_view>
#include <iostream>


//#include <thrust/host_vector.h>

inline void ThrowIfFailed(CUresult const& ret, int line, char const* filename)
{
	if (ret != CUresult::CUDA_SUCCESS)
	{
		char const* desc{ nullptr }, * name{ nullptr };
		cuGetErrorName(ret, std::addressof(name));
		cuGetErrorString(ret, std::addressof(desc));
		std::cout << "[*] CUDA Driver Error at file " << filename << " line " << line << "\n";
		if (desc && name)
		{
			std::cout << "[*] " << name << "\n";
			std::cout << "[*] " << desc << "\n";
		}
		else
		{
			std::cout << "[*] Error acquiring description for this error\n";
		}
		throw std::runtime_error("CUDA Driver Error");
	}
}

inline void ThrowIfFailed(cudaError_t const& ret, int line, char const* filename)
{
	if (ret != cudaError_t::cudaSuccess)
	{
		char const* desc{ cudaGetErrorString(ret) }, * name{ cudaGetErrorName(ret) };
		std::cout << "[*] CUDA Runtime Error at file " << filename << " line " << line << "\n";
		if (desc && name)
		{
			std::cout << "[*] " << name << "\n";
			std::cout << "[*] " << desc << "\n";
		}
		else
		{
			std::cout << "[*] Error acquiring description for this error\n";
		}
		throw std::runtime_error("CUDA Runtime Error");
	}
}

#define ck2(call) ThrowIfFailed(call, __LINE__, __FILE__)

// from http://reedbeta.com/blog/python-like-enumerate-in-cpp17/
template <typename T,
	typename TIter = decltype(std::begin(std::declval<T>())),
	typename = decltype(std::end(std::declval<T>()))>
	constexpr auto enumerate(T&& iterable)
{
	struct iterator
	{
		size_t i;
		TIter iter;
		bool operator != (const iterator& other) const { return iter != other.iter; }
		void operator ++ () { ++i; ++iter; }
		auto operator * () const { return std::tie(i, *iter); }
	};
	struct iterable_wrapper
	{
		T iterable;
		auto begin() { return iterator{ 0, std::begin(iterable) }; }
		auto end() { return iterator{ 0, std::end(iterable) }; }
	};
	return iterable_wrapper{ std::forward<T>(iterable) };
}

class ScopeTimer
{
	std::string_view s;
	std::chrono::high_resolution_clock::time_point t0;
public:
	ScopeTimer(std::string_view s) :s(s), t0(std::chrono::high_resolution_clock::now()) {}
	ScopeTimer(const ScopeTimer&) = delete;
	ScopeTimer& operator=(const ScopeTimer&) = delete;
	~ScopeTimer()
	{
		std::chrono::duration<float> diff(std::chrono::high_resolution_clock::now() - t0);
		std::cout << s << " " << diff.count() << "s\n";
	}
};

struct pair_hash
{
	template <class T1, class T2>
	std::size_t operator () (const std::pair<T1, T2>& p) const
	{
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);

		// Mainly for demonstration purposes, i.e. works but is overly simple
		// In the real world, use sth. like boost.hash_combine
		h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
		return h1;
	}
};
