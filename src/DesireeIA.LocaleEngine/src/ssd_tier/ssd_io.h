// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_SSD_IO_H
#define DESIREEIA_SSD_IO_H

// Low-level SSD I/O layer using direct positional reads.
//   - Direct I/O (O_DIRECT / FILE_FLAG_NO_BUFFERING) to bypass the page cache
//   - Aligned buffers (4096-byte) required for direct I/O
//   - Positional reads (pread-style) for concurrent access without seek
//   - posix_fadvise(DONTNEED) to evict streamed pages from the page cache
//   - macOS: fcntl(F_NOCACHE) as equivalent of O_DIRECT

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <fcntl.h>
#else
#include <sys/mman.h>
#endif
#endif

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace desireeia {

static constexpr size_t SSD_ALIGN = 4096;

inline void* ssd_aligned_alloc(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, SSD_ALIGN);
#else
    void* p = nullptr;
    if (posix_memalign(&p, SSD_ALIGN, size) != 0) return nullptr;
    return p;
#endif
}

inline void ssd_aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

inline size_t ssd_aligned_size(size_t n) {
    return (n + SSD_ALIGN - 1) & ~(SSD_ALIGN - 1);
}

// Direct-I/O file handle for reading tensor data from a GGUF file,
// bypassing the kernel page cache.
//
//   Linux:   open() with O_DIRECT, pread() for positional reads
//   Windows: CreateFileA with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
//            ReadFile for positional reads
//   macOS:   open() with O_RDONLY, fcntl(F_NOCACHE) to advise against caching,
//            pread() for positional reads
//
// All read offsets and buffer addresses must be sector-aligned (4096 bytes)
// when direct I/O is active. Use aligned buffers from ssd_aligned_alloc().
class SsdFile {
public:
    SsdFile() = default;
    ~SsdFile() { close(); }

    SsdFile(const SsdFile&) = delete;
    SsdFile& operator=(const SsdFile&) = delete;

    bool open(const std::string& path) {
        close();
#ifdef _WIN32
        handle_ = CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            handle_ = CreateFileA(
                path.c_str(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                nullptr,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                nullptr);
            direct_ = false;
            return handle_ != INVALID_HANDLE_VALUE;
        }
        direct_ = true;
        return true;
#elif defined(__APPLE__)
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;
        fcntl(fd_, F_NOCACHE, 1);
        direct_ = true;
        return true;
#else
        int flags = O_RDONLY;
#ifdef O_DIRECT
        flags |= O_DIRECT;
        direct_ = true;
#else
        direct_ = false;
#endif
        fd_ = ::open(path.c_str(), flags);
        return fd_ >= 0;
#endif
    }

    void close() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }

    bool is_open() const {
#ifdef _WIN32
        return handle_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }

    bool is_direct() const { return direct_; }

    // Positional read at byte offset `off`, reading `n` bytes into `buf`.
    // `buf` MUST be aligned to SSD_ALIGN when direct I/O is active.
    // Retries on EINTR (Linux/macOS) and ERROR_IO_PENDING (Windows).
    size_t pread(void* buf, size_t n, uint64_t off) const {
        if (!is_open()) return 0;
#ifdef _WIN32
        OVERLAPPED ov = {};
        ov.Offset = static_cast<DWORD>(off & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>(off >> 32);
        DWORD bytesRead = 0;
        BOOL ok = ReadFile(handle_, buf, static_cast<DWORD>(n), &bytesRead, &ov);
        if (!ok) {
            DWORD err = GetLastError();
            if (err == ERROR_IO_PENDING) {
                ok = GetOverlappedResult(handle_, &ov, &bytesRead, TRUE);
            }
            if (!ok) return 0;
        }
        return static_cast<size_t>(bytesRead);
#else
        ssize_t total = 0;
        char* p = static_cast<char*>(buf);
        uint64_t pos = off;
        while (static_cast<size_t>(total) < n) {
            ssize_t r = ::pread(fd_, p + total, n - total, pos + total);
            if (r < 0) {
                if (errno == EINTR) continue;
                return 0;
            }
            if (r == 0) return static_cast<size_t>(total);
            total += r;
        }
        return static_cast<size_t>(total);
#endif
    }

    // Evict pages from the page cache after a streaming read.
    //   Linux:  posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED)
    //   macOS:  no equivalent; F_NOCACHE at open time prevents caching
    //   Windows: FILE_FLAG_NO_BUFFERING avoids the page cache entirely
    void fadvise_dontneed(uint64_t off, size_t len) const {
#ifdef __linux__
        if (fd_ >= 0) {
            posix_fadvise(fd_, static_cast<off_t>(off),
                          static_cast<off_t>(len), POSIX_FADV_DONTNEED);
        }
#else
        (void)off; (void)len;
#endif
    }

    // Hint the kernel that sequential access is expected.
    //   Linux:  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)
    //   macOS/Windows: no-op (behavior is the default)
    void fadvise_sequential() const {
#ifdef __linux__
        if (fd_ >= 0) {
            posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
        }
#endif
    }

private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    bool direct_ = false;
};

// Pre-allocated aligned slab for reading tensor data.
// Avoids malloc/free per read.
struct AlignedSlab {
    void* data = nullptr;
    size_t capacity = 0;

    bool alloc(size_t min_capacity) {
        free();
        capacity = ssd_aligned_size(min_capacity);
        data = ssd_aligned_alloc(capacity);
        return data != nullptr;
    }

    void free() {
        if (data) {
            ssd_aligned_free(data);
            data = nullptr;
            capacity = 0;
        }
    }

    ~AlignedSlab() { free(); }
};

} // namespace desireeia

#endif // DESIREEIA_SSD_IO_H
