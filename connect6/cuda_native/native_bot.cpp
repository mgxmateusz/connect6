#include <torch/extension.h>


torch::Tensor tactical_bot_actions_cuda(
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left);


torch::Tensor tactical_bot_actions(
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left) {
    TORCH_CHECK(boards.is_cuda(), "boards must be a CUDA tensor");
    TORCH_CHECK(current_player.is_cuda(), "current_player must be a CUDA tensor");
    TORCH_CHECK(stones_left.is_cuda(), "stones_left must be a CUDA tensor");
    TORCH_CHECK(boards.scalar_type() == torch::kInt8, "boards must have dtype torch.int8");
    TORCH_CHECK(current_player.scalar_type() == torch::kInt8, "current_player must have dtype torch.int8");
    TORCH_CHECK(stones_left.scalar_type() == torch::kInt8, "stones_left must have dtype torch.int8");
    TORCH_CHECK(boards.dim() == 3, "boards must have shape [B, 19, 19]");
    TORCH_CHECK(boards.size(1) == 19 && boards.size(2) == 19, "GPU tactical bot currently supports only 19x19 boards");
    TORCH_CHECK(current_player.dim() == 1 && current_player.size(0) == boards.size(0), "current_player must have shape [B]");
    TORCH_CHECK(stones_left.dim() == 1 && stones_left.size(0) == boards.size(0), "stones_left must have shape [B]");
    TORCH_CHECK(boards.device() == current_player.device(), "boards and current_player must be on the same CUDA device");
    TORCH_CHECK(boards.device() == stones_left.device(), "boards and stones_left must be on the same CUDA device");

    return tactical_bot_actions_cuda(
        boards.contiguous(),
        current_player.contiguous(),
        stones_left.contiguous());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "tactical_bot_actions",
        &tactical_bot_actions,
        "Connect6 one-pass tactical bot actions on CUDA");
}
