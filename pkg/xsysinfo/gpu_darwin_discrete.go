package xsysinfo

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/mudler/xlog"
)

// getMacDiscreteGPUMemory reports non-Apple GPUs on macOS.
//
// getAppleGPUMemory only accepts cards whose spdisplays_vendor contains
// "apple", which covers Apple Silicon and nothing else. On an Intel Mac with a
// discrete AMD or NVIDIA card -- or an Intel iGPU -- every detection path in
// GetGPUMemoryUsage misses: nvidia-smi and rocm-smi are not installed, clinfo
// is absent, and the Apple probe filters the card out by vendor. LocalAI then
// reports `GPU vendor=""` and `Total available VRAM 0` and sizes its hardware
// defaults as if the machine had no GPU at all.
//
// system_profiler knows the answer. A discrete card reports its dedicated VRAM
// in spdisplays_vram; an integrated one reports its shared carve-out in
// spdisplays_vram_shared (or _spdisplays_vram). Live usage comes from IOKit,
// where each accelerator node carries inUseVidMemoryBytes.
func getMacDiscreteGPUMemory() []GPUMemoryInfo {
	if runtime.GOOS != "darwin" {
		return nil
	}
	if _, err := exec.LookPath("system_profiler"); err != nil {
		return nil
	}

	cmd := exec.Command("system_profiler", "SPDisplaysDataType", "-json")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		xlog.Debug("system_profiler failed", "error", err, "stderr", stderr.String())
		return nil
	}

	var result struct {
		SPDisplaysDataType []struct {
			Name       string `json:"_name"`
			Model      string `json:"sppci_model"`
			DeviceType string `json:"sppci_device_type"`
			Bus        string `json:"sppci_bus"`
			Vendor     string `json:"spdisplays_vendor"`
			VRAM       string `json:"spdisplays_vram"`
			VRAMShared string `json:"spdisplays_vram_shared"`
			VRAMLegacy string `json:"_spdisplays_vram"`
		} `json:"SPDisplaysDataType"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		xlog.Debug("failed to parse system_profiler output", "error", err)
		return nil
	}

	usage := ioregVidMemoryInUse()

	var gpus []GPUMemoryInfo
	for _, d := range result.SPDisplaysDataType {
		if d.DeviceType != "spdisplays_gpu" {
			continue
		}
		vendor := macVendor(d.Vendor)
		if vendor == VendorApple {
			continue // handled by getAppleGPUMemory (unified memory)
		}

		total := parseVRAMSize(firstNonEmpty(d.VRAM, d.VRAMShared, d.VRAMLegacy))
		if total == 0 {
			continue
		}

		name := firstNonEmpty(d.Model, d.Name, "GPU")
		info := GPUMemoryInfo{
			Index:     len(gpus),
			Name:      name,
			Vendor:    vendor,
			TotalVRAM: total,
		}
		// Attribute live usage by vendor: IOKit names the accelerator class
		// (AMDRadeonX6000_..., IntelAccelerator, NVDAStartup), which is enough
		// on a machine with one card per vendor -- the normal Mac case.
		if used, ok := usage[vendor]; ok && used <= total {
			info.UsedVRAM = used
			info.FreeVRAM = total - used
			info.UsagePercent = float64(used) / float64(total) * 100
		}
		xlog.Debug("macOS GPU detected via system_profiler",
			"device", name, "vendor", vendor, "bus", d.Bus,
			"total_vram", info.TotalVRAM, "used_vram", info.UsedVRAM)
		gpus = append(gpus, info)
	}
	return gpus
}

func macVendor(v string) string {
	s := strings.ToLower(v)
	switch {
	case strings.Contains(s, "apple"):
		return VendorApple
	case strings.Contains(s, "amd"), strings.Contains(s, "ati"):
		return VendorAMD
	case strings.Contains(s, "nvidia"):
		return VendorNVIDIA
	case strings.Contains(s, "intel"):
		return VendorIntel
	}
	return VendorUnknown
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// parseVRAMSize converts system_profiler's human-readable sizes ("4 GB",
// "1536 MB") to bytes. Returns 0 when the string is not recognised.
func parseVRAMSize(s string) uint64 {
	m := regexp.MustCompile(`(?i)^\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)\s*$`).FindStringSubmatch(s)
	if m == nil {
		return 0
	}
	n, err := strconv.ParseFloat(m[1], 64)
	if err != nil || n <= 0 {
		return 0
	}
	switch strings.ToLower(m[2]) {
	case "kb":
		return uint64(n * 1024)
	case "mb":
		return uint64(n * 1024 * 1024)
	case "gb":
		return uint64(n * 1024 * 1024 * 1024)
	case "tb":
		return uint64(n * 1024 * 1024 * 1024 * 1024)
	}
	return 0
}

var (
	ioregClassRe = regexp.MustCompile(`"IOClass"\s*=\s*"([^"]+)"`)
	ioregInUseRe = regexp.MustCompile(`"inUseVidMemoryBytes"\s*=\s*([0-9]+)`)
)

// ioregVidMemoryInUse returns bytes of VRAM in use, keyed by vendor, from the
// IOAccelerator nodes. This is the same figure Activity Monitor shows and the
// only cross-runtime ground truth on macOS -- a Vulkan/Metal process does not
// otherwise report where its buffers landed.
func ioregVidMemoryInUse() map[string]uint64 {
	out := map[string]uint64{}
	if runtime.GOOS != "darwin" {
		return out
	}
	if _, err := exec.LookPath("ioreg"); err != nil {
		return out
	}
	cmd := exec.Command("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		xlog.Debug("ioreg failed", "error", err)
		return out
	}

	// Nodes are emitted in blocks; take the largest inUseVidMemoryBytes seen in
	// each block and attribute it to the vendor named by that block's IOClass.
	for _, block := range strings.Split(stdout.String(), "+-o ") {
		cm := ioregClassRe.FindStringSubmatch(block)
		if cm == nil {
			continue
		}
		vendor := macVendor(cm[1])
		if vendor == VendorUnknown {
			continue
		}
		var best uint64
		for _, um := range ioregInUseRe.FindAllStringSubmatch(block, -1) {
			if v, err := strconv.ParseUint(um[1], 10, 64); err == nil && v > best {
				best = v
			}
		}
		if best > out[vendor] {
			out[vendor] = best
		}
	}
	return out
}
