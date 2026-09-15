cmake_minimum_required(VERSION 3.20)
project(DesireeIALocaleEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DESIREEIA_BUILD_TESTS "Build native self tests" OFF)

# Backend CUDA: OFF di default per non richiedere il CUDA Toolkit sulle
# macchine CPU-only (vedi docs/CUDAPiano.md). Quando ON, richiede
# CUDAToolkit installato e visibile a CMake (CUDAToolkit_ROOT o rilevamento
# automatico) — non ancora verificato su questa macchina di sviluppo
# (nvcc non trovato al momento in cui questo blocco e' stato scritto,
# 2026-09-14): il flag esiste e la build CPU-only resta invariata a OFF,
# ma la compilazione/misura del kernel CUDA resta da fare su una macchina
# col toolkit disponibile.
option(DESIREEIA_ENABLE_CUDA "Build the CUDA backend" OFF)
if(DESIREEIA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(DesireeIALocaleEngine SHARED
    src/abi.cpp
    src/core/probe.cpp
    src/core/plan.cpp
    src/core/mem_lock.cpp
    src/core/matmul.cpp
    src/core/thread_pool.cpp
    src/core/quant.cpp
    src/core/ctx.cpp
    src/core/sampler.cpp
    src/core/chat_template.cpp
    src/core/arch_tags.cpp
    src/models/dense_forward.cpp
    src/models/moe_route.cpp
    src/models/prerouter.cpp
    src/models/ssm_forward.cpp
    src/models/bert_forward.cpp
    src/tokenizer/spm_tokenizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/tokenizer/tokenizer.cpp
    src/gguf/gguf_reader.cpp
    src/quant/quant.cpp
    src/safetensors/st_reader.cpp
    src/kv/kv_cache.cpp
    src/kv/kv_quant.cpp
    src/expert/expert_store.cpp
    src/ssd_tier/ssd_tier.cpp
    src/ssd_tier/mirror_manager.cpp
    src/ssd_tier/tiered_expert_store.cpp
    src/ssd_tier/hybrid_tier.cpp
    src/vision/vision_image.cpp
    src/vision/vision_clip.cpp
    src/vision/vision_gguf.cpp
)

if(DESIREEIA_ENABLE_CUDA)
    target_sources(DesireeIALocaleEngine PRIVATE src/cuda/backend_cuda.cu)
    set_target_properties(DesireeIALocaleEngine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(DesireeIALocaleEngine PRIVATE CUDA::cudart)
    target_compile_definitions(DesireeIALocaleEngine PRIVATE DESIREEIA_CUDA_ENABLED)
endif()

target_include_directories(DesireeIALocaleEngine PUBLIC include)
target_include_directories(DesireeIALocaleEngine PRIVATE src)
target_compile_definitions(DesireeIALocaleEngine PRIVATE DESIREEIA_BUILD)

if(MSVC)
    # Con CUDA abilitato, questo target compila anche file .cu con nvcc: le
    # flag MSVC-style (/O2 /arch:AVX2) non sono flag nvcc valide e, non
    # prefissate da -Xcompiler, vengono scambiate per ulteriori file di
    # input ("a single input file is required"). $<COMPILE_LANGUAGE:CXX>
    # le limita ai soli sorgenti C++, lasciando nvcc libero di applicare le
    # proprie flag di ottimizzazione (gia' specificate a parte, vedi sopra).
    target_compile_options(DesireeIALocaleEngine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /arch:AVX2>)
    # Equivalente MSVC di -Wl,--export-all-symbols (ramo MinGW sotto): senza
    # questo, MSVC non esporta nulla da una DLL a meno di __declspec(dllexport)
    # esplicito ovunque, e desireeia_selftest (che chiama direttamente classi
    # interne come ExpertStore/SsdTier, non solo l'ABI C pubblica) fallisce a
    # linkare con LNK2019 su ogni simbolo interno usato dai test.
    set_target_properties(DesireeIALocaleEngine PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    target_compile_options(DesireeIALocaleEngine PRIVATE -O3 -march=native)
    set_target_properties(DesireeIALocaleEngine PROPERTIES OUTPUT_NAME "DesireeIALocaleEngine" PREFIX "")

    if(MINGW)
        # --export-all-symbols is a MinGW-flavored binutils flag (it mimics
        # a Windows DLL's default of exporting nothing without it). Plain
        # GNU ld on Linux/macOS does not recognize it at all — passing it
        # there is a hard link error, not a no-op, which this branch used
        # to do unconditionally for every non-MSVC toolchain including a
        # native Linux gcc build. ELF shared objects already export every
        # non-static symbol by default, which is the same effect this flag
        # achieves on Windows, so Linux/macOS need nothing extra here.
        target_link_options(DesireeIALocaleEngine PRIVATE -Wl,--export-all-symbols)
        get_filename_component(MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        add_custom_command(TARGET DesireeIALocaleEngine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MINGW_BIN_DIR}/libstdc++-6.dll"
                "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
                "${MINGW_BIN_DIR}/libwinpthread-1.dll"
                $<TARGET_FILE_DIR:DesireeIALocaleEngine>)
    endif()
endif()

if(DESIREEIA_BUILD_TESTS)
    enable_testing()
    add_executable(desireeia_selftest tests/selftest.cpp)
    target_include_directories(desireeia_selftest PRIVATE src)
    target_link_libraries(desireeia_selftest PRIVATE DesireeIALocaleEngine)
    if(DESIREEIA_ENABLE_CUDA)
        target_compile_definitions(desireeia_selftest PRIVATE DESIREEIA_CUDA_ENABLED)
    endif()
    add_test(NAME desireeia_selftest COMMAND desireeia_selftest)

    # Microbenchmark dei soli kernel: serve a confrontare due varianti in
    # secondi invece che minuti, perche' le misure end-to-end su portatile
    # derivano del 30% per throttling termico (vedi bench/kernel_bench.cpp).
    add_executable(desireeia_kernel_bench ${CMAKE_CURRENT_SOURCE_DIR}/../../bench/kernel_bench.cpp)
    target_include_directories(desireeia_kernel_bench PRIVATE src)
    target_link_libraries(desireeia_kernel_bench PRIVATE DesireeIALocaleEngine)
    if(DESIREEIA_ENABLE_CUDA)
        target_compile_definitions(desireeia_kernel_bench PRIVATE DESIREEIA_CUDA_ENABLED)
    endif()
endif()
