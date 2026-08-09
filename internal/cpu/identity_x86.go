//go:build amd64 || 386

package cpu

import "encoding/binary"

func detectX86Identity() (vendor string, family, model uint32) {
	maxLeaf, ebx, ecx, edx := cpuid(0, 0)
	if maxLeaf < 1 {
		return "", 0, 0
	}

	vendorBytes := make([]byte, 12)
	binary.LittleEndian.PutUint32(vendorBytes[0:4], ebx)
	binary.LittleEndian.PutUint32(vendorBytes[4:8], edx)
	binary.LittleEndian.PutUint32(vendorBytes[8:12], ecx)

	eax, _, _, _ := cpuid(1, 0)
	baseFamily := (eax >> 8) & 0xf
	baseModel := (eax >> 4) & 0xf
	extendedFamily := (eax >> 20) & 0xff
	extendedModel := (eax >> 16) & 0xf

	family = baseFamily
	if baseFamily == 0xf {
		family += extendedFamily
	}

	model = baseModel
	if baseFamily == 0x6 || baseFamily == 0xf {
		model |= extendedModel << 4
	}

	return string(vendorBytes), family, model
}
