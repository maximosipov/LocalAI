package xsysinfo

import "testing"

func TestMacDiscreteDetectionOnThisHost(t *testing.T) {
	gpus := getMacDiscreteGPUMemory()
	if len(gpus) == 0 {
		t.Skip("no non-Apple GPU reported on this host")
	}
	for _, g := range gpus {
		t.Logf("idx=%d name=%q vendor=%s total=%d MB used=%d MB (%.1f%%)",
			g.Index, g.Name, g.Vendor, g.TotalVRAM/1024/1024, g.UsedVRAM/1024/1024, g.UsagePercent)
	}
	if gpus[0].TotalVRAM == 0 {
		t.Fatalf("total VRAM still 0")
	}
}

func TestParseVRAMSize(t *testing.T) {
	cases := map[string]uint64{
		"4 GB": 4 << 30, "1536 MB": 1536 << 20, "8GB": 8 << 30,
		"": 0, "N/A": 0, "garbage": 0,
	}
	for in, want := range cases {
		if got := parseVRAMSize(in); got != want {
			t.Errorf("parseVRAMSize(%q) = %d, want %d", in, got, want)
		}
	}
}
