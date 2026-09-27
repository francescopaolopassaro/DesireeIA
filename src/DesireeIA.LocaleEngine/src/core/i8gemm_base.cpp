// Interleaved int8 GEMM, baseline copy (AVX2 on x86, NEON sdot on ARM), plus
// the run-time choice among the copies. See i8gemm.h.
#include "i8gemm.h"
#include <cstdlib>
#include <string>

#if defined(__AVX2__)
#define I8_NS i8_base
#define I8_DOT 0
#define I8_TARGET
#include "i8gemm_impl.h"
namespace desireeia { namespace i8_base { bool available() { return true; } } }
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define I8_NS i8_base
#define I8_DOT 3
#define I8_TARGET
#include "i8gemm_impl.h"
namespace desireeia { namespace i8_base { bool available() { return true; } } }
#else
namespace desireeia { namespace i8_base {
bool available() { return false; }
void run(const I8GemmJob&) {}
} }
#endif

#if defined(__AVX2__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace desireeia {
namespace {

#if defined(__AVX2__)
void cpuid(int leaf, int sub, int r[4]) {
#if defined(_MSC_VER)
    __cpuidex(r, leaf, sub);
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(leaf, sub, a, b, c, d);
    r[0] = (int) a; r[1] = (int) b; r[2] = (int) c; r[3] = (int) d;
#endif
}

unsigned long long xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
#endif
}

bool has_avx_vnni() {
    int r[4];
    cpuid(0, 0, r);
    if (r[0] < 7) return false;
    cpuid(7, 1, r);
    return (r[0] >> 4) & 1;                      // CPUID.(7,1):EAX[4]
}

bool has_avx512_vnni_vl() {
    int r[4];
    cpuid(0, 0, r);
    if (r[0] < 7) return false;
    cpuid(7, 0, r);
    const bool vnni = (r[2] >> 11) & 1;          // ECX[11] AVX512_VNNI
    const bool vl = (r[1] >> 31) & 1;            // EBX[31] AVX512VL
    const bool f = (r[1] >> 16) & 1;             // EBX[16] AVX512F
    // EVEX encodings need the OS to save the AVX-512 state (opmask + ZMM).
    const bool os = (xcr0() & 0xE6) == 0xE6;
    return vnni && vl && f && os;
}
#endif

enum class Isa { None, Base, Vnni, Vnni512 };

Isa pick() {
    // DESIREEIA_CPU_ISA=avx2|none forces a lower level (benchmarks, bisecting).
    const char* force = std::getenv("DESIREEIA_CPU_ISA");
    if (force && std::string(force) == "none") return Isa::None;
    const bool base_only = force && std::string(force) == "avx2";
#if defined(__AVX2__)
    if (!base_only && has_avx_vnni() && i8_vnni::available()) return Isa::Vnni;
    if (!base_only && has_avx512_vnni_vl() && i8_vnni512::available()) return Isa::Vnni512;
#endif
    (void) base_only;
    return i8_base::available() ? Isa::Base : Isa::None;
}

Isa isa() {
    static const Isa v = pick();
    return v;
}

} // namespace

bool i8_gemm_interleaved(const I8GemmJob& job) {
    switch (isa()) {
        case Isa::Vnni: i8_vnni::run(job); return true;
        case Isa::Vnni512: i8_vnni512::run(job); return true;
        case Isa::Base: i8_base::run(job); return true;
        default: return false;
    }
}

const char* i8_gemm_isa() {
    switch (isa()) {
        case Isa::Vnni: return "avx-vnni";
        case Isa::Vnni512: return "avx512-vnni";
#if defined(__ARM_NEON)
        case Isa::Base: return "neon-dotprod";
#else
        case Isa::Base: return "avx2";
#endif
        default: return "none";
    }
}

} // namespace desireeia
