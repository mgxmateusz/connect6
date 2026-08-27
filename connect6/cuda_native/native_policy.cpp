#include <torch/extension.h>
#include <vector>


torch::Tensor policy_actions_dense_cuda(
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> norm_weights,
    std::vector<torch::Tensor> norm_biases,
    torch::Tensor policy_weight,
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left,
    torch::Tensor model_ids);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "policy_actions_dense",
        &policy_actions_dense_cuda,
        "Connect6 V6 dense-board native policy argmax (SM120 FP16 WMMA + GroupNorm)"
    );
}
