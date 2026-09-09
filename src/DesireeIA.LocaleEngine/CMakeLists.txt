cmake_minimum_required(VERSION 3.20)
project(DesireeIALocaleEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DESIREEIA_BUILD_TESTS "Build native self tests" OFF)

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

target_include_directories(DesireeIALocaleEngine PUBLIC include)
target_include_directories(DesireeIALocaleEngine PRIVATE src)
target_compile_definitions(DesireeIALocaleEngine PRIVATE DESIREEIA_BUILD)

if(MSVC)
    target_compile_options(DesireeIALocaleEngine PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(DesireeIALocaleEngine PRIVATE -O3 -march=native)
    set_target_properties(DesireeIALocaleEngine PROPERTIES OUTPUT_NAME "DesireeIALocaleEngine" PREFIX "")
    target_link_options(DesireeIALocaleEngine PRIVATE -Wl,--export-all-symbols)

    if(MINGW)
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
    add_test(NAME desireeia_selftest COMMAND desireeia_selftest)

    # Microbenchmark dei soli kernel: serve a confrontare due varianti in
    # secondi invece che minuti, perche' le misure end-to-end su portatile
    # derivano del 30% per throttling termico (vedi bench/kernel_bench.cpp).
    add_executable(desireeia_kernel_bench ${CMAKE_CURRENT_SOURCE_DIR}/../../bench/kernel_bench.cpp)
    target_include_directories(desireeia_kernel_bench PRIVATE src)
    target_link_libraries(desireeia_kernel_bench PRIVATE DesireeIALocaleEngine)
endif()
