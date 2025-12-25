//go:build amd64 && !purego

#include "textflag.h"

TEXT ·stubAsm(SB), NOSPLIT|NOFRAME, $0-0
	RET
