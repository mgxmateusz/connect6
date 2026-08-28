#include "native_bot_hybrid_liveroad_pair_variants_kernel.cu"

extern "C" cudaError_t launch_tactical_bot_hybrid_liveroad_pair32_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_hybrid_liveroad_pairk_kernel<32><<<batch, v4_detail::THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}
