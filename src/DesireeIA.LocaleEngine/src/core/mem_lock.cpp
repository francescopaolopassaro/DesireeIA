#include "mem_lock.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#endif

namespace desireeia {

#if defined(_WIN32)

bool mlock_range(const void* ptr, size_t len) {
    if (!ptr || len == 0) return false;

    // VirtualLock fails outright if the range doesn't fit the process's
    // current working-set quota (a few hundred KB by default) — locking a
    // model's weights needs that quota raised first, not just attempted.
    // Grown to fit this call plus headroom for the rest of the process, not
    // set to an unbounded maximum: an unbounded quota just moves the same
    // failure to whatever tries to lock next.
    SIZE_T minSize = 0, maxSize = 0;
    if (GetProcessWorkingSetSize(GetCurrentProcess(), &minSize, &maxSize)) {
        const SIZE_T wanted = static_cast<SIZE_T>(len) + (64ull * 1024 * 1024);
        if (wanted > maxSize) {
            SetProcessWorkingSetSize(GetCurrentProcess(),
                                     minSize + wanted, maxSize + wanted);
        }
    }

    return VirtualLock(const_cast<void*>(ptr), len) != 0;
}

void munlock_range(const void* ptr, size_t len) {
    if (!ptr || len == 0) return;
    VirtualUnlock(const_cast<void*>(ptr), len);
}

#else

bool mlock_range(const void* ptr, size_t len) {
    if (!ptr || len == 0) return false;

    // Try to raise RLIMIT_MEMLOCK to fit, same reasoning as the Windows
    // working-set bump: the default limit (commonly 64 KiB or 8 MiB
    // depending on the distribution) is nowhere near a model's weights.
    // Raising past the hard limit requires a privilege this process may not
    // have — attempted, not required: mlock() below is what actually tells
    // us whether it worked.
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        const rlim_t wanted = static_cast<rlim_t>(len) + (64ull * 1024 * 1024);
        if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < wanted) {
            struct rlimit want = rl;
            want.rlim_cur = (rl.rlim_max == RLIM_INFINITY) ? wanted
                           : (wanted < rl.rlim_max ? wanted : rl.rlim_max);
            setrlimit(RLIMIT_MEMLOCK, &want); // best-effort; ignored on failure
        }
    }

    return mlock(ptr, len) == 0;
}

void munlock_range(const void* ptr, size_t len) {
    if (!ptr || len == 0) return;
    munlock(ptr, len);
}

#endif

}
