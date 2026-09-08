#include "engine.h"
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <thread>
#include <unistd.h>
#include <dlfcn.h>
#endif

namespace desireeia {

namespace {

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
    out.cpu_threads = static_cast<int32_t>(si.dwNumberOfProcessors);

    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        out.ram_total_mb = ms.ullTotalPhys / (1024 * 1024);
        out.ram_free_mb = ms.ullAvailPhys / (1024 * 1024);
    }
#else
    out.cpu_threads = static_cast<int32_t>(std::thread::hardware_concurrency());
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
    out.cpu_has_avx = 2;
    out.cpu_has_avx2 = 1;
    out.cpu_has_avx512 = 1;
#elif defined(__aarch64__) || defined(_M_ARM64)
    out.cpu_has_neon = 1;
#endif

    detect_gpu(out);

    return DESIREEIA_OK;
}

}