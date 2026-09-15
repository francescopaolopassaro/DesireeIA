// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_MEM_LOCK_H
#define DESIREEIA_MEM_LOCK_H

#include <cstddef>

namespace desireeia {

// Best-effort request that the OS never page [ptr, ptr+len) out to disk.
//
// "Best-effort" is the honest description, not a hedge: every platform caps
// how much a process may lock (Linux's RLIMIT_MEMLOCK, Windows' per-process
// working-set quota), and a multi-gigabyte model routinely exceeds the
// default. This function raises what it can (Windows: bumps the working-set
// quota to fit before locking; Linux: tries to raise RLIMIT_MEMLOCK first)
// and returns false without throwing or logging on failure — a caller
// running unprivileged, or simply out of quota, should keep the weights it
// already has rather than treat "couldn't lock" as fatal. Locking is a
// latency guarantee (no swap stall mid-decode under memory pressure), never
// a correctness one: unlocked weights are still perfectly correct, just not
// guaranteed to stay resident.
bool mlock_range(const void* ptr, size_t len);

// Releases a range locked by mlock_range. Safe to call on a range that
// failed to lock (a no-op then, not an error).
void munlock_range(const void* ptr, size_t len);

}

#endif
