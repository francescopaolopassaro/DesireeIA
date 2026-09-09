#include "engine.h"
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#include <thread>
#include <unistd.h>
#include <dlfcn.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
#include <cstdio>
#include <set>
#include <utility>
#endif

namespace desireeia {

namespace {

// Physical core count, distinct from the LOGICAL processor count
// std::thread::hardware_concurrency()/GetSystemInfo() report. Sizing the
// thread pool from the logical count means one thread per hyperthread
// sibling, and for the AVX2 memory-bound kernels here two threads sharing a
// physical core's L1/L2 contend for the same cache lines and same load/store
// ports instead of adding throughput — measured on this machine as the 20-
// vs-22-thread cliff already documented in thread_pool.cpp. That cap works
// around the symptom; this fixes what the pool is told to begin with.
//
// Returns 0 on any failure to detect, so the caller can fall back to the
// logical count rather than sizing the pool to zero.
int32_t physical_core_count() {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<uint8_t> buf(len);
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return 0;
    int32_t cores = 0;
    size_t offset = 0;
    while (offset < len) {
        auto* rec = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
        if (rec->Relationship == RelationProcessorCore) ++cores;
        if (rec->Size == 0) break; // malformed record: stop rather than loop forever
        offset += rec->Size;
    }
    return cores;
#elif defined(__APPLE__)
    int32_t cores = 0;
    size_t sz = sizeof(cores);
    if (sysctlbyname("hw.physicalcpu", &cores, &sz, nullptr, 0) != 0) return 0;
    return cores;
#else
    // Linux: no single syscall gives this. /proc/cpuinfo lists one block per
    // LOGICAL processor, each carrying "physical id" (socket) and "core id"
    // (core within that socket) — hyperthread siblings share both. Counting
    // distinct (physical id, core id) pairs counts physical cores, including
    // correctly across multiple sockets.
    std::FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return 0;
    std::set<std::pair<int, int>> seen;
    int phys = 0, core = -1;
    bool have_phys = false, have_core = false;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        int v;
        if (std::sscanf(line, "physical id : %d", &v) == 1) { phys = v; have_phys = true; }
        else if (std::sscanf(line, "core id : %d", &v) == 1) { core = v; have_core = true; }
        else if (line[0] == '\n') {
            // Blank line ends one logical processor's block.
            if (have_core) seen.insert({have_phys ? phys : 0, core});
            have_phys = false; have_core = false;
        }
    }
    if (have_core) seen.insert({have_phys ? phys : 0, core});
    std::fclose(f);
    return (int32_t) seen.size();
#endif
}

#if defined(__x86_64__) || defined(_M_X64)
// Real AVX-512F support: CPUID leaf 7 feature bit AND the OS's XCR0 saying
// it has enabled the zmm/opmask register state (checked via XGETBV, which
// is itself gated on CPUID leaf 1's OSXSAVE bit — asking the OS whether it
// exposed XGETBV to user mode before calling it). Both are required: a CPU
// that reports the feature bit with the state left disabled by the OS still
// faults on the first AVX-512 instruction.
bool cpu_supports_avx512() {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    if (!osxsave) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    const bool os_avx512_state = (xcr0 & 0xE0) == 0xE0; // bits 5,6,7: opmask, zmm_hi256, hi16_zmm
    if (!os_avx512_state) return false;
    int regs7[4] = {0, 0, 0, 0};
    __cpuidex(regs7, 7, 0);
    return (regs7[1] & (1 << 16)) != 0; // EBX bit 16: AVX512F
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1), "c"(0));
    const bool osxsave = (ecx & (1u << 27)) != 0;
    if (!osxsave) return false;
    unsigned int xcr0_lo, xcr0_hi;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    const bool os_avx512_state = (xcr0_lo & 0xE0) == 0xE0;
    if (!os_avx512_state) return false;
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
    return (ebx & (1u << 16)) != 0;
#else
    return false; // Unknown compiler: refuse rather than guess.
#endif
}
#endif

bool lib_present(const char* name) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(name);
    if (h) {
        FreeLibrary(h);
        return true;
    }
    return false;
#else
    void* h = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (h) {
        dlclose(h);
        return true;
    }
    return false;
#endif
}

// La sola presenza di nvcuda.dll/nvml.dll NON implica una GPU NVIDIA
// funzionante: puo' restare installata da un driver rimosso solo in parte,
// o da altro software che la porta con se'. Bug reale scoperto dall'utente
// (macchina senza GPU che risultava "cuda=1" nel probe): serve interrogare
// davvero il driver (cuInit + cuDeviceGetCount) e contare i device
// riportati, non fidarsi del solo LoadLibrary/dlopen.
int cuda_real_device_count() {
#if defined(_WIN32)
    HMODULE h = LoadLibraryA("nvcuda.dll");
#else
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
#endif
    if (!h) return 0;

    using cuInit_t = int (*)(unsigned int);
    using cuDeviceGetCount_t = int (*)(int*);
#if defined(_WIN32)
    auto cuInit_fn = reinterpret_cast<cuInit_t>(GetProcAddress(h, "cuInit"));
    auto cuDeviceGetCount_fn = reinterpret_cast<cuDeviceGetCount_t>(GetProcAddress(h, "cuDeviceGetCount"));
#else
    auto cuInit_fn = reinterpret_cast<cuInit_t>(dlsym(h, "cuInit"));
    auto cuDeviceGetCount_fn = reinterpret_cast<cuDeviceGetCount_t>(dlsym(h, "cuDeviceGetCount"));
#endif

    int count = 0;
    if (cuInit_fn && cuDeviceGetCount_fn) {
        if (cuInit_fn(0) == 0 /* CUDA_SUCCESS */) {
            cuDeviceGetCount_fn(&count);
        }
    }

#if defined(_WIN32)
    FreeLibrary(h);
#else
    dlclose(h);
#endif
    return count;
}

void detect_gpu(desireeia_hw_info& out) {
#if defined(_WIN32)
    out.cuda_device_count = static_cast<int32_t>(cuda_real_device_count());
    if (lib_present("ze_loader.dll")) {
        out.intel_gpu_count = 1;
    }
    if (lib_present("axsdk.dll") || lib_present("axelera.dll")) {
        out.axelera_device_count = 1;
    }
#elif defined(__linux__) || defined(__APPLE__)
    out.cuda_device_count = static_cast<int32_t>(cuda_real_device_count());
    if (lib_present("libze_loader.so.1") || lib_present("libze_loader.so")) {
        out.intel_gpu_count = 1;
    }
    if (lib_present("libaxelera.so") || lib_present("libaxsdk.so")) {
        out.axelera_device_count = 1;
    }
#endif
}

}

desireeia_error probe_hardware(desireeia_hw_info& out) {
    std::memset(&out, 0, sizeof(out));

#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const int32_t logical = static_cast<int32_t>(si.dwNumberOfProcessors);
    const int32_t physical = physical_core_count();
    // Physical when it could be measured; the logical count only as a
    // fallback, on the same reasoning as the Linux/macOS branch below.
    out.cpu_threads = physical > 0 ? physical : logical;

    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        out.ram_total_mb = ms.ullTotalPhys / (1024 * 1024);
        out.ram_free_mb = ms.ullAvailPhys / (1024 * 1024);
    }
#else
    {
        const int32_t physical = physical_core_count();
        const int32_t logical = static_cast<int32_t>(std::thread::hardware_concurrency());
        out.cpu_threads = physical > 0 ? physical : logical;
    }
    long pages = sysconf(_SC_PHYS_PAGES);
    long avail = sysconf(_SC_AVPHYS_PAGES);
    long page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) {
        out.ram_total_mb = static_cast<uint64_t>(pages) * page / (1024 * 1024);
    }
    if (avail > 0 && page > 0) {
        out.ram_free_mb = static_cast<uint64_t>(avail) * page / (1024 * 1024);
    }
#endif

#if defined(__x86_64__) || defined(_M_X64)
    // AVX/AVX2 are safe to assume: this engine's x86_64 build already
    // requires them (the AVX2 kernels are not behind a runtime check
    // anywhere else in the codebase, so a CPU without them can't run this
    // build regardless of what probe_hardware reports).
    //
    // AVX-512 is NOT safe to assume, and until now that's exactly what this
    // did — unconditionally set to 1 for any x86_64 build, no CPUID check at
    // all. That's not "detected but unused", it's a wrong answer sitting
    // ready for the first future caller that trusts it to pick an AVX-512
    // kernel: illegal-instruction crash on any CPU without it. That
    // includes plenty of current hardware — hybrid performance/efficiency
    // designs commonly fuse AVX-512 off entirely (including, per this exact
    // function once fixed, this development machine) specifically because
    // the two core types would otherwise disagree on the instruction set.
    //
    // Real detection needs two things, not one: CPUID leaf 7 reporting the
    // AVX-512F feature bit, AND the OS having enabled the extended (zmm/k-mask)
    // register state via XSAVE — a CPU can report the feature while an OS
    // that predates it leaves that state disabled, and executing an
    // AVX-512 instruction in that case still faults.
    out.cpu_has_avx = 2;
    out.cpu_has_avx2 = 1;
    out.cpu_has_avx512 = cpu_supports_avx512() ? 1 : 0;
#elif defined(__aarch64__) || defined(_M_ARM64)
    out.cpu_has_neon = 1;
#endif

    detect_gpu(out);

    return DESIREEIA_OK;
}

}