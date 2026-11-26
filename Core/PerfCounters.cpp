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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#ifndef _WIN32
#include <time.h>
#endif

#include "Common/Profiler/Profiler.h"
#include "Common/System/PerfTimes.h"
#include "Core/ELF/ParamSFO.h"
#include "Core/PerfCounters.h"
#include "Core/System.h"
#include "GPU/GPU.h"

namespace perf {

using Clock = std::chrono::steady_clock;

static constexpr double UPDATE_INTERVAL_MS = 1000.0;
static constexpr double PSP_VBLANKS_PER_SEC = 59.94;

// TEMPORARY diagnostics, remove after optimization
static constexpr const char *DIAG_LOG_PATH = "/opt/davpsp/davpsp_diag.log";
static constexpr const char *CPU_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
static constexpr const char *GPU_FREQ_PATH = "/sys/class/devfreq/ff400000.gpu/cur_freq";
static constexpr const char *DMC_FREQ_PATH = "/sys/class/devfreq/dmc/cur_freq";

// Interval accumulators
static bool started;
static Clock::time_point intervalStart;
static Clock::time_point lastFlip;
static bool haveLastFlip;
static unsigned int hostFrames, vblanks, actualFlips, skippedFrames;
static float minFrameAcc, avgFrameAcc, maxFrameAcc;
static unsigned int frameTimeCount;
static double emuCpuStartMs, rendCpuStartMs;
static int rendClockAtStart;

// Published values (read by the overlay on this same thread)
static float minFrameTime, avgFrameTime, maxFrameTime;
static float emuCpuUsage, emuCpuTime;
static float rendCpuUsage = -1.0f, rendCpuTime;

// TEMPORARY diagnostics: GPU statistics, summed per interval
enum GpuStat {
	GS_DRAWS, GS_FLUSHES, GS_VERTS, GS_TEXDECODE, GS_TEXHASH, GS_TEXINVAL,
	GS_UPLOADS, GS_READBACKS, GS_BLOCKREADBACKS, GS_FBOS, GS_COUNT
};
static double gpuStatAcc[GS_COUNT];
static unsigned int gpuStatFrames;

// TEMPORARY diagnostics: profiler categories shown as log columns, ms per host frame.
// "Flush" (Vulkan) and "flush" (GLES) share a column.
struct ProfColumn {
	const char *header;
	const char *names[2];
};
static const ProfColumn profColumns[] = {
	{ "jit",    { "jit", nullptr } },
	{ "jitc",   { "jitc", nullptr } },
	{ "jhash",  { "jithash", nullptr } },
	{ "syscl",  { "syscall", nullptr } },
	{ "timing", { "timing", nullptr } },
	{ "advnc",  { "advance", nullptr } },
	{ "gpucl",  { "gpu_call", nullptr } },
	{ "gloop",  { "gpuloop", nullptr } },
	{ "exprm",  { "execprim", nullptr } },
	{ "flush",  { "Flush", "flush" } },
	{ "block",  { "block", nullptr } },
	{ "texh",   { "texhash", nullptr } },
	{ "dectx",  { "decodetex", nullptr } },
	{ "shcmp",  { "shadercomp", nullptr } },
	{ "plbld",  { "pipelinebuild", nullptr } },
	{ "rmq",    { "renderman_q", nullptr } },
	{ "mixer",  { "mixer", nullptr } },
	{ "io_rw",  { "io_rw", nullptr } },
};
static constexpr int NUM_PROF_COLUMNS = (int)(sizeof(profColumns) / sizeof(profColumns[0]));

static FILE *diagLog;
static bool diagLogTried;
static Clock::time_point diagStart;
static std::string lastLoggedGame;

static double ThreadCpuMs(int clock) {
#ifndef _WIN32
	timespec ts;
	if (clock_gettime((clockid_t)clock, &ts) == 0)
		return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
	return -1.0;
}

static double EmuCpuMs() {
#ifndef _WIN32
	return ThreadCpuMs((int)CLOCK_THREAD_CPUTIME_ID);
#else
	return -1.0;
#endif
}

static double ElapsedMs(Clock::time_point from, Clock::time_point to) {
	return std::chrono::duration<double, std::milli>(to - from).count();
}

// Reads a frequency from sysfs and converts it to MHz
static unsigned int ReadFreqMHz(const char *path, unsigned int divider) {
	FILE *f = fopen(path, "r");
	if (!f)
		return 0;
	unsigned long long value = 0;
	if (fscanf(f, "%llu", &value) != 1)
		value = 0;
	fclose(f);
	return (unsigned int)(value / divider);
}

static void ResetBaseline(Clock::time_point now) {
	intervalStart = now;
	hostFrames = vblanks = actualFlips = skippedFrames = 0;
	minFrameAcc = avgFrameAcc = maxFrameAcc = 0.0f;
	frameTimeCount = 0;
	emuCpuStartMs = EmuCpuMs();
	rendClockAtStart = g_renderClock.load(std::memory_order_relaxed);
	rendCpuStartMs = rendClockAtStart >= 0 ? ThreadCpuMs(rendClockAtStart) : -1.0;
	g_presentNs.store(0, std::memory_order_relaxed);
	g_presentMaxNs.store(0, std::memory_order_relaxed);
	g_presentCount.store(0, std::memory_order_relaxed);
	g_syncWaitNs.store(0, std::memory_order_relaxed);
	g_audioUnderruns.store(0, std::memory_order_relaxed);
	for (double &acc : gpuStatAcc)
		acc = 0.0;
	gpuStatFrames = 0;
}

// TEMPORARY diagnostics: average ms per host frame of one profiler column over the last frames.
static float ProfColumnMs(const ProfColumn &col, int frames, float *history) {
#ifdef USE_PROFILER
	const int numCats = Profiler_GetNumCategories();
	const int numThreads = Profiler_GetNumThreads();
	double sum = 0.0;
	for (int c = 0; c < numCats; c++) {
		const char *name = Profiler_GetCategoryName(c);
		if (!name)
			continue;
		bool match = false;
		for (const char *want : col.names) {
			if (want && strcmp(name, want) == 0)
				match = true;
		}
		if (!match)
			continue;
		for (int t = 0; t < numThreads; t++) {
			// Called at the start of NativeFrame, before PROFILE_END_FRAME: with count = frames + 1,
			// entry frames is the host frame that just finished and entry 0 is one frame too old.
			Profiler_GetHistory(c, t, history, frames + 1);
			for (int i = 1; i <= frames; i++)
				sum += history[i];
		}
	}
	return (float)(sum * 1000.0 / std::max(frames, 1));
#else
	return 0.0f;
#endif
}

// TEMPORARY diagnostics: all profiled time, ms per host frame.
static float ProfTotalMs(int frames, float *history) {
#ifdef USE_PROFILER
	const int numCats = Profiler_GetNumCategories();
	const int numThreads = Profiler_GetNumThreads();
	double sum = 0.0;
	for (int c = 0; c < numCats; c++) {
		for (int t = 0; t < numThreads; t++) {
			Profiler_GetHistory(c, t, history, frames + 1);
			for (int i = 1; i <= frames; i++)
				sum += history[i];
		}
	}
	return (float)(sum * 1000.0 / std::max(frames, 1));
#else
	return 0.0f;
#endif
}

// TEMPORARY diagnostics: one line per second to DIAG_LOG_PATH.
static void WriteDiagLog(float fps, float speed, float presAvg, float presMax, float syncMs) {
	if (!diagLogTried) {
		diagLogTried = true;
		diagLog = fopen(DIAG_LOG_PATH, "w");
		if (diagLog) {
			diagStart = Clock::now();
			fprintf(diagLog, "# DavPSP diagnostics. Times in ms. GPU stats and profiler columns are per host frame.\n");
			fprintf(diagLog, "%7s %6s %6s %8s %8s %6s %7s %6s %7s %8s %8s %7s %5s %6s %6s %7s %6s %6s %6s %5s %5s %5s %4s %5s %6s %6s %6s",
				"time_s", "fps", "speed%", "frm_avg", "frm_max",
				"emu%", "emu_ms", "rend%", "rend_ms",
				"pres_avg", "pres_max", "sync_ms", "skip",
				"draws", "flush", "verts", "texdec", "texhsh", "texinv",
				"upld", "rdbk", "brdbk", "fbo", "undrn",
				"cpuMHz", "gpuMHz", "dmcMHz");
			for (const ProfColumn &col : profColumns)
				fprintf(diagLog, " %6s", col.header);
			fprintf(diagLog, " %6s %6s\n", "other", "prof");
		}
	}
	if (!diagLog)
		return;

	// Mark which game the following lines belong to.
	std::string game = g_paramSFO.GetValueString("DISC_ID") + " " + g_paramSFO.GetValueString("TITLE");
	if (game != lastLoggedGame) {
		lastLoggedGame = game;
		fprintf(diagLog, "# game: %s\n", game.c_str());
	}

	const double t = std::chrono::duration<double>(Clock::now() - diagStart).count();
	const float gf = (float)std::max(gpuStatFrames, 1u);
	fprintf(diagLog, "%7.1f %6.1f %6.1f %8.2f %8.2f %6.1f %7.2f %6.1f %7.2f %8.2f %8.2f %7.2f %5u %6.0f %6.0f %7.0f %6.1f %6.1f %6.1f %5.1f %5.1f %5.1f %4.1f %5u %6u %6u %6u",
		t, fps, speed, avgFrameTime, maxFrameTime,
		emuCpuUsage, emuCpuTime, rendCpuUsage, rendCpuTime,
		presAvg, presMax, syncMs, skippedFrames,
		gpuStatAcc[GS_DRAWS] / gf, gpuStatAcc[GS_FLUSHES] / gf, gpuStatAcc[GS_VERTS] / gf,
		gpuStatAcc[GS_TEXDECODE] / gf, gpuStatAcc[GS_TEXHASH] / gf, gpuStatAcc[GS_TEXINVAL] / gf,
		gpuStatAcc[GS_UPLOADS] / gf, gpuStatAcc[GS_READBACKS] / gf, gpuStatAcc[GS_BLOCKREADBACKS] / gf,
		gpuStatAcc[GS_FBOS] / gf, g_audioUnderruns.load(std::memory_order_relaxed),
		ReadFreqMHz(CPU_FREQ_PATH, 1000), ReadFreqMHz(GPU_FREQ_PATH, 1000000), ReadFreqMHz(DMC_FREQ_PATH, 1000000));

	// The profiler keeps 128 frames of history; only the frames of this interval are summed.
	const int frames = std::min((int)hostFrames, 126);
	float history[128];
	float listed = 0.0f;
	for (const ProfColumn &col : profColumns) {
		const float ms = ProfColumnMs(col, frames, history);
		listed += ms;
		fprintf(diagLog, " %6.2f", ms);
	}
	const float total = ProfTotalMs(frames, history);
	fprintf(diagLog, " %6.2f %6.2f\n", std::max(total - listed, 0.0f), total);
	fflush(diagLog);
}

void OnVblank() {
	vblanks++;
}

void OnActualFlip() {
	const Clock::time_point now = Clock::now();
	actualFlips++;
	if (haveLastFlip) {
		const float ms = (float)ElapsedMs(lastFlip, now);
		minFrameAcc = frameTimeCount == 0 ? ms : std::min(minFrameAcc, ms);
		maxFrameAcc = std::max(maxFrameAcc, ms);
		avgFrameAcc += ms;
		frameTimeCount++;
	}
	lastFlip = now;
	haveLastFlip = true;
}

void OnSkippedFrame() {
	skippedFrames++;
}

void OnGpuStats(const GPUStatistics &stats) {
	gpuStatAcc[GS_DRAWS] += stats.numDrawCalls;
	gpuStatAcc[GS_FLUSHES] += stats.numFlushes;
	gpuStatAcc[GS_VERTS] += stats.numVertsSubmitted;
	gpuStatAcc[GS_TEXDECODE] += stats.numTexturesDecoded;
	gpuStatAcc[GS_TEXHASH] += stats.numTexturesHashed;
	gpuStatAcc[GS_TEXINVAL] += stats.numTextureInvalidations;
	gpuStatAcc[GS_UPLOADS] += stats.numUploads;
	gpuStatAcc[GS_READBACKS] += stats.numReadbacks;
	gpuStatAcc[GS_BLOCKREADBACKS] += stats.numBlockingReadbacks;
	gpuStatAcc[GS_FBOS] += stats.numFBOsCreated;
	gpuStatFrames++;
}

void OnHostFrame() {
	const Clock::time_point now = Clock::now();
	if (!PSP_IsInited()) {
		// Back in the menus: start fresh when the next game runs.
		started = false;
		haveLastFlip = false;
		return;
	}
	if (!started) {
		ResetBaseline(now);
		started = true;
		return;
	}
	hostFrames++;

	const double intervalMs = ElapsedMs(intervalStart, now);
	if (intervalMs < UPDATE_INTERVAL_MS)
		return;

	const double emuNowMs = EmuCpuMs();
	const int rendClock = g_renderClock.load(std::memory_order_relaxed);
	// Emulation thread restarted (CPU time went backwards) or the render thread changed: start a new interval.
	if (emuNowMs < emuCpuStartMs || rendClock != rendClockAtStart) {
		ResetBaseline(now);
		return;
	}

	// Frame times between game frames sent to the screen
	if (frameTimeCount > 0) {
		minFrameTime = minFrameAcc;
		avgFrameTime = avgFrameAcc / (float)frameTimeCount;
		maxFrameTime = maxFrameAcc;
	} else {
		minFrameTime = avgFrameTime = maxFrameTime = 0.0f;
	}

	// Emulation thread CPU usage, time per emulated vblank
	const double emuDeltaMs = emuNowMs - emuCpuStartMs;
	emuCpuUsage = (float)(emuDeltaMs / intervalMs * 100.0);
	emuCpuTime = (float)(emuDeltaMs / std::max(vblanks, 1u));

	// Render thread CPU usage, time per presented frame
	const unsigned int presents = g_presentCount.load(std::memory_order_relaxed);
	const double rendNowMs = rendClock >= 0 ? ThreadCpuMs(rendClock) : -1.0;
	if (rendClock >= 0 && rendNowMs >= 0.0 && rendCpuStartMs >= 0.0 && rendNowMs >= rendCpuStartMs) {
		const double rendDeltaMs = rendNowMs - rendCpuStartMs;
		rendCpuUsage = (float)(rendDeltaMs / intervalMs * 100.0);
		rendCpuTime = (float)(rendDeltaMs / std::max(presents > 0 ? presents : hostFrames, 1u));
	} else {
		rendCpuUsage = -1.0f;
		rendCpuTime = 0.0f;
	}

	// TEMPORARY diagnostics
	const double intervalSec = intervalMs / 1000.0;
	const float fps = (float)(actualFlips / intervalSec);
	const float speed = (float)(vblanks / intervalSec / PSP_VBLANKS_PER_SEC * 100.0);
	const float presAvg = (float)((double)g_presentNs.load(std::memory_order_relaxed) / 1000000.0 / std::max(presents, 1u));
	const float presMax = (float)((double)g_presentMaxNs.load(std::memory_order_relaxed) / 1000000.0);
	const float syncMs = (float)((double)g_syncWaitNs.load(std::memory_order_relaxed) / 1000000.0 / std::max(hostFrames, 1u));
	WriteDiagLog(fps, speed, presAvg, presMax, syncMs);

	ResetBaseline(now);
}

float GetMinFrameTime() { return minFrameTime; }
float GetAvgFrameTime() { return avgFrameTime; }
float GetMaxFrameTime() { return maxFrameTime; }
float GetEmuCpuUsage() { return emuCpuUsage; }
float GetEmuCpuTime() { return emuCpuTime; }
float GetRendCpuUsage() { return rendCpuUsage; }
float GetRendCpuTime() { return rendCpuTime; }

}  // namespace perf
