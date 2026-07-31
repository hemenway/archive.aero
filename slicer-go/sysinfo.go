package main

// System introspection: memory snapshots and per-phase GDAL environment.
//
// The Python original set GDAL config options process-globally
// (gdal.SetConfigOption / gdal.SetCacheMax). Here every GDAL operation is its
// own subprocess, so the same knobs are passed as environment variables per
// invocation instead.

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

var (
	cpuCount       = runtime.NumCPU()
	isAppleSilicon = runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
)

type MemSnapshot struct {
	AvailableMB int
	TotalMB     int
}

// getMemorySnapshotMB returns a best-effort memory snapshot in MB
// (macOS: sysctl hw.memsize + vm_stat free/inactive pages, mirroring what
// psutil reports as "available"). Falls back to the given defaults.
func getMemorySnapshotMB(defaultAvailableMB, defaultTotalMB int) MemSnapshot {
	totalMB, errT := darwinTotalMB()
	availMB, errA := darwinAvailableMB()
	if errT != nil || errA != nil || totalMB <= 0 || availMB <= 0 {
		fallbackTotal := max(defaultTotalMB, defaultAvailableMB)
		return MemSnapshot{
			AvailableMB: max(1, defaultAvailableMB),
			TotalMB:     max(1, fallbackTotal),
		}
	}
	return MemSnapshot{
		AvailableMB: max(1, availMB),
		TotalMB:     max(max(1, availMB), totalMB),
	}
}

func darwinTotalMB() (int, error) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0, err
	}
	bytes, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0, err
	}
	return int(bytes / (1024 * 1024)), nil
}

func darwinAvailableMB() (int, error) {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0, err
	}
	lines := strings.Split(string(out), "\n")
	pageSize := int64(16384)
	var freePages, inactivePages, speculativePages int64
	for _, line := range lines {
		if strings.Contains(line, "page size of") {
			fields := strings.Fields(line)
			for i, f := range fields {
				if f == "of" && i+1 < len(fields) {
					if v, err := strconv.ParseInt(fields[i+1], 10, 64); err == nil {
						pageSize = v
					}
				}
			}
			continue
		}
		parseVal := func() int64 {
			idx := strings.LastIndex(line, ":")
			if idx < 0 {
				return 0
			}
			v := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(line[idx+1:]), "."))
			n, _ := strconv.ParseInt(v, 10, 64)
			return n
		}
		switch {
		case strings.HasPrefix(line, "Pages free"):
			freePages = parseVal()
		case strings.HasPrefix(line, "Pages inactive"):
			inactivePages = parseVal()
		case strings.HasPrefix(line, "Pages speculative"):
			speculativePages = parseVal()
		}
	}
	availBytes := (freePages + inactivePages + speculativePages) * pageSize
	if availBytes <= 0 {
		return 0, fmt.Errorf("vm_stat parse produced no pages")
	}
	return int(availBytes / (1024 * 1024)), nil
}

// recommendedFreeRAMFloorMB is the RAM headroom to preserve for the OS and
// transient process spikes.
func recommendedFreeRAMFloorMB(totalRAMMB int) int {
	totalRAMMB = max(1024, totalRAMMB)
	return max(768, min(2048, totalRAMMB/10))
}

func getAvailableRAMMB(defaultMB int) int {
	return getMemorySnapshotMB(defaultMB, 8192).AvailableMB
}

// baseGDALEnv is the environment shared by every GDAL subprocess (the
// equivalent of the module-level gdal.SetConfigOption calls in slicer.py).
func baseGDALEnv(cacheMB int) []string {
	env := os.Environ()
	env = append(env,
		fmt.Sprintf("GDAL_CACHEMAX=%d", cacheMB),
		"COMPRESS_OVERVIEW=DEFLATE",
		"GDAL_TIFF_INTERNAL_MASK=YES",
	)
	return env
}

// gdalEnvForPhase builds the subprocess environment for a processing phase.
//
//   - "warp": one single-threaded GDAL per worker process. The Python original
//     sized ONE shared block cache at available/workers; with per-process
//     caches the equivalent total is available/workers split across workers,
//     i.e. available/workers² per process (floor 256 MB).
//   - "geotiff": a single mosaic subprocess with full CPU and a large cache.
func gdalEnvForPhase(phase string, workers int, availableRAMMB int) []string {
	if availableRAMMB <= 0 {
		availableRAMMB = getAvailableRAMMB(4096)
	}
	switch phase {
	case "warp":
		w := max(1, workers)
		cache := max(256, availableRAMMB/(w*w))
		env := baseGDALEnv(cache)
		env = append(env, "GDAL_NUM_THREADS=1")
		return env
	case "geotiff":
		cache := max(512, min(8192, availableRAMMB*3/4))
		env := baseGDALEnv(cache)
		env = append(env, "GDAL_NUM_THREADS=ALL_CPUS")
		return env
	}
	return baseGDALEnv(max(512, min(8192, availableRAMMB*3/4)))
}
