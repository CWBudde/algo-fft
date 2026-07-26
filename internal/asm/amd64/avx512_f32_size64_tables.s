//go:build amd64 && !purego

// ===========================================================================
// Shared RODATA tables for the AVX-512 size-64 complex64 codelets
// ===========================================================================
//
// Both size-64 AVX-512 codelets (avx512_f32_size64_radix4.s and
// avx512_f32_size64_radix2.s) use the same two tables, so they live in their
// own file with package scope symbols.
//
// avx512F32Size64TransIdx  two VPERMI2PD index vectors for the 8x8 transpose:
//                   level 2 of the transpose exchanges register bit 1 with
//                   lane bit 1, which needs a two-source qword permute.
//                   Indices 0..7 select the first source, 8..15 the second.
//
// avx512F32Size64CrossTw   the four-step (8x8) twiddle matrix w^(l*k2), w =
//                   exp(-2*pi*i/64), for rows k2 = 1..7 (row 0 is all ones and
//                   is skipped). Each row occupies 128 bytes: the first ZMM
//                   holds Re(w) duplicated into both float32 slots of every
//                   complex64 lane, the second holds Im(w) the same way, which
//                   is exactly the VMOVSLDUP / VMOVSHDUP split the complex
//                   multiply needs, precomputed.
//
//                   The values are mathematically fixed for n = 64: each entry
//                   is float32(cos/sin(-2*pi*k/64)) for the same k that
//                   internal/math.ComputeTwiddleFactors[complex64](64) would
//                   produce, so this table is bit-identical to a strided read
//                   of the twiddle argument (which would cost 21 permutes or 7
//                   gathers to assemble at run time). A wrong entry here is
//                   caught by the registry reference sweeps in
//                   internal/kernels, which compare against a naive DFT.
// ===========================================================================

#include "textflag.h"

DATA ·avx512F32Size64TransIdx+0(SB)/8, $0
DATA ·avx512F32Size64TransIdx+8(SB)/8, $1
DATA ·avx512F32Size64TransIdx+16(SB)/8, $8
DATA ·avx512F32Size64TransIdx+24(SB)/8, $9
DATA ·avx512F32Size64TransIdx+32(SB)/8, $4
DATA ·avx512F32Size64TransIdx+40(SB)/8, $5
DATA ·avx512F32Size64TransIdx+48(SB)/8, $12
DATA ·avx512F32Size64TransIdx+56(SB)/8, $13
DATA ·avx512F32Size64TransIdx+64(SB)/8, $2
DATA ·avx512F32Size64TransIdx+72(SB)/8, $3
DATA ·avx512F32Size64TransIdx+80(SB)/8, $10
DATA ·avx512F32Size64TransIdx+88(SB)/8, $11
DATA ·avx512F32Size64TransIdx+96(SB)/8, $6
DATA ·avx512F32Size64TransIdx+104(SB)/8, $7
DATA ·avx512F32Size64TransIdx+112(SB)/8, $14
DATA ·avx512F32Size64TransIdx+120(SB)/8, $15
GLOBL ·avx512F32Size64TransIdx(SB), RODATA|NOPTR, $128

// row k2=1: w^(l*1) for l = 0..7
DATA ·avx512F32Size64CrossTw+0(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+4(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+8(SB)/4, $0x3f7ec46d // Re(w^1)
DATA ·avx512F32Size64CrossTw+12(SB)/4, $0x3f7ec46d // Re(w^1)
DATA ·avx512F32Size64CrossTw+16(SB)/4, $0x3f7b14be // Re(w^2)
DATA ·avx512F32Size64CrossTw+20(SB)/4, $0x3f7b14be // Re(w^2)
DATA ·avx512F32Size64CrossTw+24(SB)/4, $0x3f74fa0b // Re(w^3)
DATA ·avx512F32Size64CrossTw+28(SB)/4, $0x3f74fa0b // Re(w^3)
DATA ·avx512F32Size64CrossTw+32(SB)/4, $0x3f6c835e // Re(w^4)
DATA ·avx512F32Size64CrossTw+36(SB)/4, $0x3f6c835e // Re(w^4)
DATA ·avx512F32Size64CrossTw+40(SB)/4, $0x3f61c598 // Re(w^5)
DATA ·avx512F32Size64CrossTw+44(SB)/4, $0x3f61c598 // Re(w^5)
DATA ·avx512F32Size64CrossTw+48(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+52(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+56(SB)/4, $0x3f45e403 // Re(w^7)
DATA ·avx512F32Size64CrossTw+60(SB)/4, $0x3f45e403 // Re(w^7)
DATA ·avx512F32Size64CrossTw+64(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+68(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+72(SB)/4, $0xbdc8bd36 // Im(w^1)
DATA ·avx512F32Size64CrossTw+76(SB)/4, $0xbdc8bd36 // Im(w^1)
DATA ·avx512F32Size64CrossTw+80(SB)/4, $0xbe47c5c2 // Im(w^2)
DATA ·avx512F32Size64CrossTw+84(SB)/4, $0xbe47c5c2 // Im(w^2)
DATA ·avx512F32Size64CrossTw+88(SB)/4, $0xbe94a031 // Im(w^3)
DATA ·avx512F32Size64CrossTw+92(SB)/4, $0xbe94a031 // Im(w^3)
DATA ·avx512F32Size64CrossTw+96(SB)/4, $0xbec3ef15 // Im(w^4)
DATA ·avx512F32Size64CrossTw+100(SB)/4, $0xbec3ef15 // Im(w^4)
DATA ·avx512F32Size64CrossTw+104(SB)/4, $0xbef15aea // Im(w^5)
DATA ·avx512F32Size64CrossTw+108(SB)/4, $0xbef15aea // Im(w^5)
DATA ·avx512F32Size64CrossTw+112(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+116(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+120(SB)/4, $0xbf226799 // Im(w^7)
DATA ·avx512F32Size64CrossTw+124(SB)/4, $0xbf226799 // Im(w^7)
// row k2=2: w^(l*2) for l = 0..7
DATA ·avx512F32Size64CrossTw+128(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+132(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+136(SB)/4, $0x3f7b14be // Re(w^2)
DATA ·avx512F32Size64CrossTw+140(SB)/4, $0x3f7b14be // Re(w^2)
DATA ·avx512F32Size64CrossTw+144(SB)/4, $0x3f6c835e // Re(w^4)
DATA ·avx512F32Size64CrossTw+148(SB)/4, $0x3f6c835e // Re(w^4)
DATA ·avx512F32Size64CrossTw+152(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+156(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+160(SB)/4, $0x3f3504f3 // Re(w^8)
DATA ·avx512F32Size64CrossTw+164(SB)/4, $0x3f3504f3 // Re(w^8)
DATA ·avx512F32Size64CrossTw+168(SB)/4, $0x3f0e39da // Re(w^10)
DATA ·avx512F32Size64CrossTw+172(SB)/4, $0x3f0e39da // Re(w^10)
DATA ·avx512F32Size64CrossTw+176(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+180(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+184(SB)/4, $0x3e47c5c2 // Re(w^14)
DATA ·avx512F32Size64CrossTw+188(SB)/4, $0x3e47c5c2 // Re(w^14)
DATA ·avx512F32Size64CrossTw+192(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+196(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+200(SB)/4, $0xbe47c5c2 // Im(w^2)
DATA ·avx512F32Size64CrossTw+204(SB)/4, $0xbe47c5c2 // Im(w^2)
DATA ·avx512F32Size64CrossTw+208(SB)/4, $0xbec3ef15 // Im(w^4)
DATA ·avx512F32Size64CrossTw+212(SB)/4, $0xbec3ef15 // Im(w^4)
DATA ·avx512F32Size64CrossTw+216(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+220(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+224(SB)/4, $0xbf3504f3 // Im(w^8)
DATA ·avx512F32Size64CrossTw+228(SB)/4, $0xbf3504f3 // Im(w^8)
DATA ·avx512F32Size64CrossTw+232(SB)/4, $0xbf54db31 // Im(w^10)
DATA ·avx512F32Size64CrossTw+236(SB)/4, $0xbf54db31 // Im(w^10)
DATA ·avx512F32Size64CrossTw+240(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+244(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+248(SB)/4, $0xbf7b14be // Im(w^14)
DATA ·avx512F32Size64CrossTw+252(SB)/4, $0xbf7b14be // Im(w^14)
// row k2=3: w^(l*3) for l = 0..7
DATA ·avx512F32Size64CrossTw+256(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+260(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+264(SB)/4, $0x3f74fa0b // Re(w^3)
DATA ·avx512F32Size64CrossTw+268(SB)/4, $0x3f74fa0b // Re(w^3)
DATA ·avx512F32Size64CrossTw+272(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+276(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+280(SB)/4, $0x3f226799 // Re(w^9)
DATA ·avx512F32Size64CrossTw+284(SB)/4, $0x3f226799 // Re(w^9)
DATA ·avx512F32Size64CrossTw+288(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+292(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+296(SB)/4, $0x3dc8bd36 // Re(w^15)
DATA ·avx512F32Size64CrossTw+300(SB)/4, $0x3dc8bd36 // Re(w^15)
DATA ·avx512F32Size64CrossTw+304(SB)/4, $0xbe47c5c2 // Re(w^18)
DATA ·avx512F32Size64CrossTw+308(SB)/4, $0xbe47c5c2 // Re(w^18)
DATA ·avx512F32Size64CrossTw+312(SB)/4, $0xbef15aea // Re(w^21)
DATA ·avx512F32Size64CrossTw+316(SB)/4, $0xbef15aea // Re(w^21)
DATA ·avx512F32Size64CrossTw+320(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+324(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+328(SB)/4, $0xbe94a031 // Im(w^3)
DATA ·avx512F32Size64CrossTw+332(SB)/4, $0xbe94a031 // Im(w^3)
DATA ·avx512F32Size64CrossTw+336(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+340(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+344(SB)/4, $0xbf45e403 // Im(w^9)
DATA ·avx512F32Size64CrossTw+348(SB)/4, $0xbf45e403 // Im(w^9)
DATA ·avx512F32Size64CrossTw+352(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+356(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+360(SB)/4, $0xbf7ec46d // Im(w^15)
DATA ·avx512F32Size64CrossTw+364(SB)/4, $0xbf7ec46d // Im(w^15)
DATA ·avx512F32Size64CrossTw+368(SB)/4, $0xbf7b14be // Im(w^18)
DATA ·avx512F32Size64CrossTw+372(SB)/4, $0xbf7b14be // Im(w^18)
DATA ·avx512F32Size64CrossTw+376(SB)/4, $0xbf61c598 // Im(w^21)
DATA ·avx512F32Size64CrossTw+380(SB)/4, $0xbf61c598 // Im(w^21)
// row k2=4: w^(l*4) for l = 0..7
DATA ·avx512F32Size64CrossTw+384(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+388(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+392(SB)/4, $0x3f6c835e // Re(w^4)
DATA ·avx512F32Size64CrossTw+396(SB)/4, $0x3f6c835e // Re(w^4)
DATA ·avx512F32Size64CrossTw+400(SB)/4, $0x3f3504f3 // Re(w^8)
DATA ·avx512F32Size64CrossTw+404(SB)/4, $0x3f3504f3 // Re(w^8)
DATA ·avx512F32Size64CrossTw+408(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+412(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+416(SB)/4, $0x248d3132 // Re(w^16)
DATA ·avx512F32Size64CrossTw+420(SB)/4, $0x248d3132 // Re(w^16)
DATA ·avx512F32Size64CrossTw+424(SB)/4, $0xbec3ef15 // Re(w^20)
DATA ·avx512F32Size64CrossTw+428(SB)/4, $0xbec3ef15 // Re(w^20)
DATA ·avx512F32Size64CrossTw+432(SB)/4, $0xbf3504f3 // Re(w^24)
DATA ·avx512F32Size64CrossTw+436(SB)/4, $0xbf3504f3 // Re(w^24)
DATA ·avx512F32Size64CrossTw+440(SB)/4, $0xbf6c835e // Re(w^28)
DATA ·avx512F32Size64CrossTw+444(SB)/4, $0xbf6c835e // Re(w^28)
DATA ·avx512F32Size64CrossTw+448(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+452(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+456(SB)/4, $0xbec3ef15 // Im(w^4)
DATA ·avx512F32Size64CrossTw+460(SB)/4, $0xbec3ef15 // Im(w^4)
DATA ·avx512F32Size64CrossTw+464(SB)/4, $0xbf3504f3 // Im(w^8)
DATA ·avx512F32Size64CrossTw+468(SB)/4, $0xbf3504f3 // Im(w^8)
DATA ·avx512F32Size64CrossTw+472(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+476(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+480(SB)/4, $0xbf800000 // Im(w^16)
DATA ·avx512F32Size64CrossTw+484(SB)/4, $0xbf800000 // Im(w^16)
DATA ·avx512F32Size64CrossTw+488(SB)/4, $0xbf6c835e // Im(w^20)
DATA ·avx512F32Size64CrossTw+492(SB)/4, $0xbf6c835e // Im(w^20)
DATA ·avx512F32Size64CrossTw+496(SB)/4, $0xbf3504f3 // Im(w^24)
DATA ·avx512F32Size64CrossTw+500(SB)/4, $0xbf3504f3 // Im(w^24)
DATA ·avx512F32Size64CrossTw+504(SB)/4, $0xbec3ef15 // Im(w^28)
DATA ·avx512F32Size64CrossTw+508(SB)/4, $0xbec3ef15 // Im(w^28)
// row k2=5: w^(l*5) for l = 0..7
DATA ·avx512F32Size64CrossTw+512(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+516(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+520(SB)/4, $0x3f61c598 // Re(w^5)
DATA ·avx512F32Size64CrossTw+524(SB)/4, $0x3f61c598 // Re(w^5)
DATA ·avx512F32Size64CrossTw+528(SB)/4, $0x3f0e39da // Re(w^10)
DATA ·avx512F32Size64CrossTw+532(SB)/4, $0x3f0e39da // Re(w^10)
DATA ·avx512F32Size64CrossTw+536(SB)/4, $0x3dc8bd36 // Re(w^15)
DATA ·avx512F32Size64CrossTw+540(SB)/4, $0x3dc8bd36 // Re(w^15)
DATA ·avx512F32Size64CrossTw+544(SB)/4, $0xbec3ef15 // Re(w^20)
DATA ·avx512F32Size64CrossTw+548(SB)/4, $0xbec3ef15 // Re(w^20)
DATA ·avx512F32Size64CrossTw+552(SB)/4, $0xbf45e403 // Re(w^25)
DATA ·avx512F32Size64CrossTw+556(SB)/4, $0xbf45e403 // Re(w^25)
DATA ·avx512F32Size64CrossTw+560(SB)/4, $0xbf7b14be // Re(w^30)
DATA ·avx512F32Size64CrossTw+564(SB)/4, $0xbf7b14be // Re(w^30)
DATA ·avx512F32Size64CrossTw+568(SB)/4, $0xbf74fa0b // Re(w^35)
DATA ·avx512F32Size64CrossTw+572(SB)/4, $0xbf74fa0b // Re(w^35)
DATA ·avx512F32Size64CrossTw+576(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+580(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+584(SB)/4, $0xbef15aea // Im(w^5)
DATA ·avx512F32Size64CrossTw+588(SB)/4, $0xbef15aea // Im(w^5)
DATA ·avx512F32Size64CrossTw+592(SB)/4, $0xbf54db31 // Im(w^10)
DATA ·avx512F32Size64CrossTw+596(SB)/4, $0xbf54db31 // Im(w^10)
DATA ·avx512F32Size64CrossTw+600(SB)/4, $0xbf7ec46d // Im(w^15)
DATA ·avx512F32Size64CrossTw+604(SB)/4, $0xbf7ec46d // Im(w^15)
DATA ·avx512F32Size64CrossTw+608(SB)/4, $0xbf6c835e // Im(w^20)
DATA ·avx512F32Size64CrossTw+612(SB)/4, $0xbf6c835e // Im(w^20)
DATA ·avx512F32Size64CrossTw+616(SB)/4, $0xbf226799 // Im(w^25)
DATA ·avx512F32Size64CrossTw+620(SB)/4, $0xbf226799 // Im(w^25)
DATA ·avx512F32Size64CrossTw+624(SB)/4, $0xbe47c5c2 // Im(w^30)
DATA ·avx512F32Size64CrossTw+628(SB)/4, $0xbe47c5c2 // Im(w^30)
DATA ·avx512F32Size64CrossTw+632(SB)/4, $0x3e94a031 // Im(w^35)
DATA ·avx512F32Size64CrossTw+636(SB)/4, $0x3e94a031 // Im(w^35)
// row k2=6: w^(l*6) for l = 0..7
DATA ·avx512F32Size64CrossTw+640(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+644(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+648(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+652(SB)/4, $0x3f54db31 // Re(w^6)
DATA ·avx512F32Size64CrossTw+656(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+660(SB)/4, $0x3ec3ef15 // Re(w^12)
DATA ·avx512F32Size64CrossTw+664(SB)/4, $0xbe47c5c2 // Re(w^18)
DATA ·avx512F32Size64CrossTw+668(SB)/4, $0xbe47c5c2 // Re(w^18)
DATA ·avx512F32Size64CrossTw+672(SB)/4, $0xbf3504f3 // Re(w^24)
DATA ·avx512F32Size64CrossTw+676(SB)/4, $0xbf3504f3 // Re(w^24)
DATA ·avx512F32Size64CrossTw+680(SB)/4, $0xbf7b14be // Re(w^30)
DATA ·avx512F32Size64CrossTw+684(SB)/4, $0xbf7b14be // Re(w^30)
DATA ·avx512F32Size64CrossTw+688(SB)/4, $0xbf6c835e // Re(w^36)
DATA ·avx512F32Size64CrossTw+692(SB)/4, $0xbf6c835e // Re(w^36)
DATA ·avx512F32Size64CrossTw+696(SB)/4, $0xbf0e39da // Re(w^42)
DATA ·avx512F32Size64CrossTw+700(SB)/4, $0xbf0e39da // Re(w^42)
DATA ·avx512F32Size64CrossTw+704(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+708(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+712(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+716(SB)/4, $0xbf0e39da // Im(w^6)
DATA ·avx512F32Size64CrossTw+720(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+724(SB)/4, $0xbf6c835e // Im(w^12)
DATA ·avx512F32Size64CrossTw+728(SB)/4, $0xbf7b14be // Im(w^18)
DATA ·avx512F32Size64CrossTw+732(SB)/4, $0xbf7b14be // Im(w^18)
DATA ·avx512F32Size64CrossTw+736(SB)/4, $0xbf3504f3 // Im(w^24)
DATA ·avx512F32Size64CrossTw+740(SB)/4, $0xbf3504f3 // Im(w^24)
DATA ·avx512F32Size64CrossTw+744(SB)/4, $0xbe47c5c2 // Im(w^30)
DATA ·avx512F32Size64CrossTw+748(SB)/4, $0xbe47c5c2 // Im(w^30)
DATA ·avx512F32Size64CrossTw+752(SB)/4, $0x3ec3ef15 // Im(w^36)
DATA ·avx512F32Size64CrossTw+756(SB)/4, $0x3ec3ef15 // Im(w^36)
DATA ·avx512F32Size64CrossTw+760(SB)/4, $0x3f54db31 // Im(w^42)
DATA ·avx512F32Size64CrossTw+764(SB)/4, $0x3f54db31 // Im(w^42)
// row k2=7: w^(l*7) for l = 0..7
DATA ·avx512F32Size64CrossTw+768(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+772(SB)/4, $0x3f800000 // Re(w^0)
DATA ·avx512F32Size64CrossTw+776(SB)/4, $0x3f45e403 // Re(w^7)
DATA ·avx512F32Size64CrossTw+780(SB)/4, $0x3f45e403 // Re(w^7)
DATA ·avx512F32Size64CrossTw+784(SB)/4, $0x3e47c5c2 // Re(w^14)
DATA ·avx512F32Size64CrossTw+788(SB)/4, $0x3e47c5c2 // Re(w^14)
DATA ·avx512F32Size64CrossTw+792(SB)/4, $0xbef15aea // Re(w^21)
DATA ·avx512F32Size64CrossTw+796(SB)/4, $0xbef15aea // Re(w^21)
DATA ·avx512F32Size64CrossTw+800(SB)/4, $0xbf6c835e // Re(w^28)
DATA ·avx512F32Size64CrossTw+804(SB)/4, $0xbf6c835e // Re(w^28)
DATA ·avx512F32Size64CrossTw+808(SB)/4, $0xbf74fa0b // Re(w^35)
DATA ·avx512F32Size64CrossTw+812(SB)/4, $0xbf74fa0b // Re(w^35)
DATA ·avx512F32Size64CrossTw+816(SB)/4, $0xbf0e39da // Re(w^42)
DATA ·avx512F32Size64CrossTw+820(SB)/4, $0xbf0e39da // Re(w^42)
DATA ·avx512F32Size64CrossTw+824(SB)/4, $0x3dc8bd36 // Re(w^49)
DATA ·avx512F32Size64CrossTw+828(SB)/4, $0x3dc8bd36 // Re(w^49)
DATA ·avx512F32Size64CrossTw+832(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+836(SB)/4, $0x80000000 // Im(w^0)
DATA ·avx512F32Size64CrossTw+840(SB)/4, $0xbf226799 // Im(w^7)
DATA ·avx512F32Size64CrossTw+844(SB)/4, $0xbf226799 // Im(w^7)
DATA ·avx512F32Size64CrossTw+848(SB)/4, $0xbf7b14be // Im(w^14)
DATA ·avx512F32Size64CrossTw+852(SB)/4, $0xbf7b14be // Im(w^14)
DATA ·avx512F32Size64CrossTw+856(SB)/4, $0xbf61c598 // Im(w^21)
DATA ·avx512F32Size64CrossTw+860(SB)/4, $0xbf61c598 // Im(w^21)
DATA ·avx512F32Size64CrossTw+864(SB)/4, $0xbec3ef15 // Im(w^28)
DATA ·avx512F32Size64CrossTw+868(SB)/4, $0xbec3ef15 // Im(w^28)
DATA ·avx512F32Size64CrossTw+872(SB)/4, $0x3e94a031 // Im(w^35)
DATA ·avx512F32Size64CrossTw+876(SB)/4, $0x3e94a031 // Im(w^35)
DATA ·avx512F32Size64CrossTw+880(SB)/4, $0x3f54db31 // Im(w^42)
DATA ·avx512F32Size64CrossTw+884(SB)/4, $0x3f54db31 // Im(w^42)
DATA ·avx512F32Size64CrossTw+888(SB)/4, $0x3f7ec46d // Im(w^49)
DATA ·avx512F32Size64CrossTw+892(SB)/4, $0x3f7ec46d // Im(w^49)
GLOBL ·avx512F32Size64CrossTw(SB), RODATA|NOPTR, $896
