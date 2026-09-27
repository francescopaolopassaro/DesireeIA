// Interleaved int8 GEMM, AVX512-VNNI copy at 256 bits (vpdpbusd, EVEX). See
// i8gemm.h. The instruction set is enabled per function (I8_TARGET), never
// for the file.
#include "i8gemm.h"

#if defined(__AVX2__) && (defined(_MSC_VER) || defined(__GNUC__))
#define I8_NS i8_vnni512
#define I8_DOT 2
#if defined(_MSC_VER) && !defined(__clang__)
#define I8_TARGET
#else
#define I8_TARGET __attribute__((target("avx512vnni,avx512vl")))
#endif
#include "i8gemm_impl.h"
namespace desireeia { namespace i8_vnni512 { bool available() { return true; } } }
#else
namespace desireeia { namespace i8_vnni512 {
bool available() { return false; }
void run(const I8GemmJob&) {}
} }
#endif
