#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cstdint>

extern "C" cudaError_t launch_tactical_bot_cuda(
    const int8_t*, const int8_t*, const int8_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_v2_cuda(
    const int8_t*, const int8_t*, const int8_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_v3_top16_cuda(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_small_top12_cuda(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_v4_top12_replypair6_cuda(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_full_pair_cuda(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_pairfirst_p128_cuda(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_liveroad_cuda(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);
extern "C" cudaError_t launch_tactical_bot_hybrid_liveroad_pair128_cuda(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);

using BotLaunchFn = cudaError_t (*)(
    const int8_t*, const int8_t*, const int8_t*, int64_t*, int, cudaStream_t);
using SearchBotLaunchFn = cudaError_t (*)(
    const int8_t*, const int8_t*, const int8_t*, int16_t*, int64_t*, int, cudaStream_t);

void validate_common_inputs(
    const torch::Tensor& boards,
    const torch::Tensor& current_player,
    const torch::Tensor& stones_left) {
    TORCH_CHECK(boards.is_cuda(), "boards must be a CUDA tensor");
    TORCH_CHECK(current_player.is_cuda(), "current_player must be a CUDA tensor");
    TORCH_CHECK(stones_left.is_cuda(), "stones_left must be a CUDA tensor");
    TORCH_CHECK(boards.scalar_type() == torch::kInt8, "boards must have dtype torch.int8");
    TORCH_CHECK(current_player.scalar_type() == torch::kInt8, "current_player must have dtype torch.int8");
    TORCH_CHECK(stones_left.scalar_type() == torch::kInt8, "stones_left must have dtype torch.int8");
    TORCH_CHECK(boards.dim() == 3 && boards.size(1) == 19 && boards.size(2) == 19,
                "boards must have shape [B, 19, 19]");
    TORCH_CHECK(current_player.dim() == 1 && current_player.size(0) == boards.size(0),
                "current_player must have shape [B]");
    TORCH_CHECK(stones_left.dim() == 1 && stones_left.size(0) == boards.size(0),
                "stones_left must have shape [B]");
    TORCH_CHECK(boards.device() == current_player.device() && boards.device() == stones_left.device(),
                "all inputs must be on the same CUDA device");
}

torch::Tensor tactical_bot_actions_impl(
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left,
    BotLaunchFn launch,
    const char* label) {
    validate_common_inputs(boards, current_player, stones_left);
    auto boards_c = boards.contiguous();
    auto player_c = current_player.contiguous();
    auto left_c = stones_left.contiguous();
    const int batch = static_cast<int>(boards_c.size(0));
    auto actions = torch::empty({batch}, boards_c.options().dtype(torch::kInt64));
    if (batch == 0) return actions;
    const int device = boards_c.get_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    const cudaError_t err = launch(
        boards_c.data_ptr<int8_t>(), player_c.data_ptr<int8_t>(),
        left_c.data_ptr<int8_t>(), actions.data_ptr<int64_t>(), batch, stream);
    TORCH_CHECK(err == cudaSuccess, label, " kernel launch failed: ", cudaGetErrorString(err));
    return actions;
}

torch::Tensor tactical_search_actions_impl(
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left,
    torch::Tensor pending_second,
    SearchBotLaunchFn launch,
    const char* label) {
    validate_common_inputs(boards, current_player, stones_left);
    TORCH_CHECK(pending_second.is_cuda(), "pending_second must be a CUDA tensor");
    TORCH_CHECK(pending_second.scalar_type() == torch::kInt16,
                "pending_second must have dtype torch.int16");
    TORCH_CHECK(pending_second.dim() == 1 && pending_second.size(0) == boards.size(0),
                "pending_second must have shape [B]");
    TORCH_CHECK(pending_second.device() == boards.device(),
                "pending_second and boards must be on the same CUDA device");
    TORCH_CHECK(pending_second.is_contiguous(), "pending_second must be contiguous");

    auto boards_c = boards.contiguous();
    auto player_c = current_player.contiguous();
    auto left_c = stones_left.contiguous();
    const int batch = static_cast<int>(boards_c.size(0));
    auto actions = torch::empty({batch}, boards_c.options().dtype(torch::kInt64));
    if (batch == 0) return actions;
    const int device = boards_c.get_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    const cudaError_t err = launch(
        boards_c.data_ptr<int8_t>(), player_c.data_ptr<int8_t>(), left_c.data_ptr<int8_t>(),
        pending_second.data_ptr<int16_t>(), actions.data_ptr<int64_t>(), batch, stream);
    TORCH_CHECK(err == cudaSuccess, label, " kernel launch failed: ", cudaGetErrorString(err));
    return actions;
}

torch::Tensor tactical_bot_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l) {
    return tactical_bot_actions_impl(b, p, l, launch_tactical_bot_cuda, "GPU Tactical Bot V1");
}
torch::Tensor tactical_bot_v2_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l) {
    return tactical_bot_actions_impl(b, p, l, launch_tactical_bot_v2_cuda, "GPU Tactical Bot V2");
}
torch::Tensor tactical_bot_v3_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l, torch::Tensor pending) {
    return tactical_search_actions_impl(b, p, l, pending, launch_tactical_bot_v3_top16_cuda,
                                        "GPU Tactical Bot V3 Top16 Pair-State");
}
torch::Tensor tactical_bot_small_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l, torch::Tensor pending) {
    return tactical_search_actions_impl(b, p, l, pending, launch_tactical_bot_small_top12_cuda,
                                        "GPU Tactical Bot Small Top12 Pair-State");
}
torch::Tensor tactical_bot_v4_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l, torch::Tensor pending) {
    return tactical_search_actions_impl(b, p, l, pending, launch_tactical_bot_v4_top12_replypair6_cuda,
                                        "GPU Tactical Bot V4 Top12 ReplyPair6");
}
torch::Tensor tactical_bot_full_pair_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l, torch::Tensor pending) {
    return tactical_search_actions_impl(b, p, l, pending, launch_tactical_bot_full_pair_cuda,
                                        "GPU Tactical Bot Full Pair Brute Force");
}
torch::Tensor tactical_bot_pairfirst_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l, torch::Tensor pending) {
    return tactical_search_actions_impl(b, p, l, pending, launch_tactical_bot_pairfirst_p128_cuda,
                                        "GPU Tactical Bot PairFirst AllPairs P128");
}
torch::Tensor tactical_bot_liveroad_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l, torch::Tensor pending) {
    return tactical_search_actions_impl(b, p, l, pending, launch_tactical_bot_liveroad_cuda,
                                        "GPU Tactical Bot LiveRoad Brute Force");
}
torch::Tensor tactical_bot_hybrid_actions(torch::Tensor b, torch::Tensor p, torch::Tensor l, torch::Tensor pending) {
    return tactical_search_actions_impl(b, p, l, pending, launch_tactical_bot_hybrid_liveroad_pair128_cuda,
                                        "GPU Tactical Bot Hybrid LiveRoad Pair128");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tactical_bot_actions", &tactical_bot_actions, "Connect6 GPU Tactical Bot V1 actions");
    m.def("tactical_bot_v2_actions", &tactical_bot_v2_actions, "Connect6 GPU Tactical Bot V2 actions");
    m.def("tactical_bot_v3_actions", &tactical_bot_v3_actions, "Connect6 GPU Tactical Bot V3 top16 actions");
    m.def("tactical_bot_small_actions", &tactical_bot_small_actions, "Connect6 GPU Tactical Bot Small top12 actions");
    m.def("tactical_bot_v4_actions", &tactical_bot_v4_actions, "Connect6 GPU Tactical Bot V4 reply-pair6 actions");
    m.def("tactical_bot_full_pair_actions", &tactical_bot_full_pair_actions,
          "Connect6 GPU Tactical Bot exhaustive full-pair actions");
    m.def("tactical_bot_pairfirst_actions", &tactical_bot_pairfirst_actions,
          "Connect6 GPU Tactical Bot all-pairs cheap prefilter then TOP128 exact actions");
    m.def("tactical_bot_liveroad_actions", &tactical_bot_liveroad_actions,
          "Connect6 GPU Tactical Bot live-road restricted exact brute-force actions");
    m.def("tactical_bot_hybrid_actions", &tactical_bot_hybrid_actions,
          "Connect6 GPU Tactical Bot pure LiveRoad pool then cheap TOP128 exact actions");
}
