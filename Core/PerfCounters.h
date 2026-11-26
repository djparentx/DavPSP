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

// DavPSP performance counters for the on-screen CPU/GPU overlay.
// Modeled on DavCast's perf_counters. Everything here runs on the emulation thread
// (the thread that calls NativeFrame) unless noted.

struct GPUStatistics;

namespace perf {

// Once per host frame, from NativeFrame. Publishes the values once a second.
void OnHostFrame();
// Emulated vblank.
void OnVblank();
// A new game frame was sent to the screen.
void OnActualFlip();
// Frameskip dropped a frame.
void OnSkippedFrame();
// Per host frame GPU statistics, read just before they are reset.
void OnGpuStats(const GPUStatistics &stats);

float GetMinFrameTime();
float GetAvgFrameTime();
float GetMaxFrameTime();
float GetEmuCpuUsage();
float GetEmuCpuTime();
// Negative when the render thread is not known (render work runs on the emulation thread).
float GetRendCpuUsage();
float GetRendCpuTime();

}  // namespace perf
