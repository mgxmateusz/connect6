#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> run_championship_cuda(
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases,
    torch::Tensor policy_weight,
    torch::Tensor game_ids,
    int64_t num_models,
    int64_t slots);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_championship",
        &run_championship_cuda,
        "Connect6 fully device-side championship (SM120, FP16 WMMA)"
    );
}
