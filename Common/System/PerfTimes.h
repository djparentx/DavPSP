// Copyright (c) 2026- DavPSP Project, based on PPSSPP.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2.0 or later versions.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License 2.0 for more details.

// A copy of the GPL 2.0 should have been included with the program.
// If not, see http://www.gnu.org/licenses/

#pragma once

// DavPSP performance counters written by the GPU backends (Common) and read by Core/PerfCounters.
// Header only, so Common does not need to link against Core.

#include <atomic>
#include <cstdint>

#ifndef _WIN32
#include <pthread.h>
#include <time.h>
#endif

namespace perf {

// Present call time (vkQueuePresentKHR / SDL_GL_SwapWindow), render thread.
inline std::atomic<uint64_t> g_presentNs{ 0 };
inline std::atomic<uint64_t> g_presentMaxNs{ 0 };
inline std::atomic<uint32_t> g_presentCount{ 0 };
// Time the emulation thread waits for the render thread / GPU in BeginFrame.
inline std::atomic<uint64_t> g_syncWaitNs{ 0 };
// CPU clock of the render thread, -1 when it has no thread of its own.
inline std::atomic<int> g_renderClock{ -1 };
// Audio resampler underruns (audio thread).
inline std::atomic<uint32_t> g_audioUnderruns{ 0 };

inline uint64_t SecondsToNs(double seconds) {
	return seconds > 0.0 ? (uint64_t)(seconds * 1000000000.0) : 0;
}

inline void AddPresent(double seconds) {
	const uint64_t ns = SecondsToNs(seconds);
	g_presentNs.fetch_add(ns, std::memory_order_relaxed);
	g_presentCount.fetch_add(1, std::memory_order_relaxed);
	uint64_t prev = g_presentMaxNs.load(std::memory_order_relaxed);
	while (ns > prev && !g_presentMaxNs.compare_exchange_weak(prev, ns, std::memory_order_relaxed)) {
	}
}

inline void AddSyncWait(double seconds) {
	g_syncWaitNs.fetch_add(SecondsToNs(seconds), std::memory_order_relaxed);
}

// Call on the thread that does the rendering work.
inline void RegisterRenderThread() {
#ifndef _WIN32
	clockid_t cid;
	if (pthread_getcpuclockid(pthread_self(), &cid) == 0)
		g_renderClock.store((int)cid, std::memory_order_relaxed);
#endif
}

inline void UnregisterRenderThread() {
	g_renderClock.store(-1, std::memory_order_relaxed);
}

}  // namespace perf
