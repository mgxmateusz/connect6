#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cstdint>


extern "C" cudaError_t launch_tactical_bot_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int64_t* actions,
    int batch,
    cudaStream_t stream);

extern "C" cudaError_t launch_tactical_bot_v2_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int64_t* actions,
    int batch,
    cudaStream_t stream);

using BotLaunchFn = cudaError_t (*)(
    const int8_t*,
    const int8_t*,
    const int8_t*,
    int64_t*,
    int,
    cudaStream_t);


torch::Tensor tactical_bot_actions_impl(
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left,
    BotLaunchFn launch,
    const char* label) {
    TORCH_CHECK(boards.is_cuda(), "boards must be a CUDA tensor");
    TORCH_CHECK(current_player.is_cuda(), "current_player must be a CUDA tensor");
    TORCH_CHECK(stones_left.is_cuda(), "stones_left must be a CUDA tensor");
    TORCH_CHECK(boards.scalar_type() == torch::kInt8, "boards must have dtype torch.int8");
    TORCH_CHECK(current_player.scalar_type() == torch::kInt8, "current_player must have dtype torch.int8");
    TORCH_CHECK(stones_left.scalar_type() == torch::kInt8, "stones_left must have dtype torch.int8");
    TORCH_CHECK(boards.dim() == 3, "boards must have shape [B, 19, 19]");
    TORCH_CHECK(
        boards.size(1) == 19 && boards.size(2) == 19,
        "GPU tactical bot currently supports only 19x19 boards");
    TORCH_CHECK(
        current_player.dim() == 1 && current_player.size(0) == boards.size(0),
        "current_player must have shape [B]");
    TORCH_CHECK(
        stones_left.dim() == 1 && stones_left.size(0) == boards.size(0),
        "stones_left must have shape [B]");
    TORCH_CHECK(
        boards.device() == current_player.device(),
        "boards and current_player must be on the same CUDA device");
    TORCH_CHECK(
        boards.device() == stones_left.device(),
        "boards and stones_left must be on the same CUDA device");

    auto boards_c = boards.contiguous();
    auto player_c = current_player.contiguous();
    auto left_c = stones_left.contiguous();
    const int batch = static_cast<int>(boards_c.size(0));
    auto actions = torch::empty(
        {batch},
        boards_c.options().dtype(torch::kInt64));
    if (batch == 0) {
        return actions;
    }

    const int device = boards_c.get_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    const cudaError_t err = launch(
        boards_c.data_ptr<int8_t>(),
        player_c.data_ptr<int8_t>(),
        left_c.data_ptr<int8_t>(),
        actions.data_ptr<int64_t>(),
        batch,
        stream);
    TORCH_CHECK(
        err == cudaSuccess,
        label,
        " kernel launch failed: ",
        cudaGetErrorString(err));
    return actions;
}


torch::Tensor tactical_bot_actions(
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left) {
    return tactical_bot_actions_impl(
        boards,
        current_player,
        stones_left,
        launch_tactical_bot_cuda,
        "GPU Tactical Bot V1");
}


torch::Tensor tactical_bot_v2_actions(
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left) {
    return tactical_bot_actions_impl(
        boards,
        current_player,
        stones_left,
        launch_tactical_bot_v2_cuda,
        "GPU Tactical Bot V2");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "tactical_bot_actions",
        &tactical_bot_actions,
        "Connect6 GPU Tactical Bot V1 actions");
    m.def(
        "tactical_bot_v2_actions",
        &tactical_bot_v2_actions,
        "Connect6 GPU Tactical Bot V2 actions");
}
