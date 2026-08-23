#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> run_rollout_cuda(
    std::vector<torch::Tensor> conv_weights,
    std::vector<torch::Tensor> norm_weights,
    std::vector<torch::Tensor> norm_biases,
    torch::Tensor policy_weight,
    torch::Tensor policy_bias,
    torch::Tensor value_weight,
    torch::Tensor value_bias,
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left,
    torch::Tensor empty_count,
    torch::Tensor move_count,
    torch::Tensor rng_counter,
    torch::Tensor table_kind,
    torch::Tensor opponent_model,
    torch::Tensor current_color,
    torch::Tensor bot_version,
    torch::Tensor buffer_boards,
    torch::Tensor buffer_players,
    torch::Tensor buffer_stones_left,
    torch::Tensor buffer_move_counts,
    torch::Tensor buffer_actions,
    torch::Tensor buffer_logprobs,
    torch::Tensor buffer_values,
    torch::Tensor buffer_episode_ids,
    torch::Tensor episode_results,
    torch::Tensor episode_terminal_moves,
    int64_t target_completed_positions,
    double temperature,
    double random_black_opening_fraction,
    int64_t seed,
    bool symmetry_augmentation,
    int64_t symmetry_phase_start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_rollout",
        &run_rollout_cuda,
        "GPU-native Connect6 training rollout (CUDA graph conditional loop)"
    );
}
