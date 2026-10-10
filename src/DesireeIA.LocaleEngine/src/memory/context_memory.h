// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#pragma once

// Context memory: an on-board RAG store that keeps big tool payloads
// (files, uploads, command output, streams) on SSD instead of in the
// prompt.
//
// A file pasted whole into the conversation is paid for on every later turn
// in prompt tokens and in KV cache - which lives in VRAM next to the
// weights, so on a local GPU it is the first thing to run out. Here a large
// payload is streamed to disk and the model only sees a short stub (handle,
// size, outline, first lines); it then pulls just what it needs with a line
// window read or a hybrid BM25 + embedding search.
//
// On-disk layout (our own format, no external database):
//
//   <root>/<session>/<handle>/
//     data.br      blocks of CHUNK_LINES lines, each Brotli-compressed on
//                  its own -> random access decompresses only touched blocks
//     chunks.bin   fixed-size ChunkRec per block (line range, offsets)
//     terms.bin    per block: (term hash, tf) pairs - an append-only forward
//                  index, so a stream append only writes the new tail
//     vectors.f32  one L2-normalized embedding row per block (optional)
//     meta.bin     name, version, totals, content digest
//
// An item is addressed by a stable name (file path, upload name, stream
// id): the handle derives from it, so a new upload with the same name
// replaces the content (version + 1) and a stream appends to it, reopening
// only the trailing partial block. Re-putting identical content is a no-op
// returning a byte-identical stub, which keeps the prompt prefix - and the
// engine's KV reuse of it - stable across turns.
//
// A background thread deletes a session once it has been idle longer than
// the TTL, so nothing piles up on disk after a conversation is gone.

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace desireeia {

// text -> L2-normalized vector; empty when the embedder can't embed.
using MemoryEmbedder = std::function<std::vector<float>(const std::string&)>;

class ContextMemory {
public:
    ContextMemory(std::filesystem::path root, double ttl_seconds);
    ~ContextMemory();

    ContextMemory(const ContextMemory&) = delete;
    ContextMemory& operator=(const ContextMemory&) = delete;

    void set_embedder(MemoryEmbedder embedder);

    // All results are JSON documents (UTF-8), ready to hand to the model.
    std::string put_file(const std::string& session, const std::string& name, const std::filesystem::path& path);
    std::string put_text(const std::string& session, const std::string& name, const std::string& text);
    std::string append(const std::string& session, const std::string& name, const std::string& text);
    std::string read(const std::string& session, const std::string& handle, uint32_t offset, uint32_t limit);
    std::string search(const std::string& session, const std::string& query, const std::string& handle, uint32_t k);
    std::string list(const std::string& session);
    // Replace lines [first, last] (1-based, inclusive) with `text`; first =
    // last + 1 inserts before `first`, first = total + 1 appends at the end.
    std::string replace_lines(const std::string& session, const std::string& handle, uint32_t first, uint32_t last,
                              const std::string& text);
    // Stream the item to another file (written atomically). File-backed
    // items don't need it: their edits are already written through.
    std::string export_to(const std::string& session, const std::string& handle, const std::filesystem::path& path);

    void touch(const std::string& session);
    void drop_session(const std::string& session);
    int sweep();

private:
    struct Session;
    std::shared_ptr<Session> session(const std::string& id);
    std::vector<float> embed(const std::string& text);
    std::string put_file_locked(const std::filesystem::path& dir, const std::string& name,
                                const std::filesystem::path& path);
    bool refresh_locked(const std::filesystem::path& dir, const std::string& handle);
    void sweeper_loop();

    std::filesystem::path root_;
    double ttl_seconds_;
    std::mutex guard_;
    std::map<std::string, std::shared_ptr<Session>> sessions_;
    std::mutex embed_mu_;
    MemoryEmbedder embedder_;
    bool stop_ = false;
    std::condition_variable stop_cv_;
    std::thread sweeper_;
};

std::string memory_handle_for(const std::string& name);

} // namespace desireeia
