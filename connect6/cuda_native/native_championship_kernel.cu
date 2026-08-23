#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>

using namespace nvcuda;

namespace {

constexpr int BOARD = 19;
constexpr int HW = BOARD * BOARD;
constexpr int WORDS = 6;
constexpr int STORAGE_C = 96;
constexpr int WARP = 32;
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

#define CUDA_CHECK(EXPR) do { \
    cudaError_t _e = (EXPR); \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e)); \
} while (0)

__device__ __forceinline__ unsigned long long* board_ptr(
    int64_t* boards, int color, int word, int slots) {
    return reinterpret_cast<unsigned long long*>(boards) + (color * WORDS + word) * slots;
}

__device__ __forceinline__ const unsigned long long* board_ptr_const(
    const int64_t* boards, int color, int word, int slots) {
    return reinterpret_cast<const unsigned long long*>(boards) + (color * WORDS + word) * slots;
}

__device__ __forceinline__ bool bit_at(
    const int64_t* boards, int color, int slot, int pos, int slots) {
    int word = pos >> 6;
    int bit = pos & 63;
    unsigned long long value = *board_ptr_const(boards, color, word, slots);
    return ((value >> bit) & 1ULL) != 0ULL;
}

__device__ __forceinline__ bool occupied_at(
    const int64_t* boards, int slot, int pos, int slots) {
    return bit_at(boards, 0, slot, pos, slots) || bit_at(boards, 1, slot, pos, slots);
}

__device__ __forceinline__ int actor_model(
    int slot,
    const int8_t* current_player,
    const int32_t* black_model,
    const int32_t* white_model) {
    return current_player[slot] == 1 ? black_model[slot] : white_model[slot];
}

__device__ __forceinline__ half canonical_input_value(
    const int64_t* boards,
    int slot,
    int pos,
    int channel,
    int slots,
    const int8_t* current_player,
    const int8_t* stones_left) {
    if (channel == 2) {
        return __float2half(stones_left[slot] == 1 ? 1.0f : 0.0f);
    }
    const bool black_to_move = current_player[slot] == 1;
    const int me_color = black_to_move ? 0 : 1;
    const int opp_color = black_to_move ? 1 : 0;
    const bool value = bit_at(boards, channel == 0 ? me_color : opp_color, slot, pos, slots);
    return __float2half(value ? 1.0f : 0.0f);
}

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + __expf(-x));
}

// First convolution: canonical checkpoint input is generated lazily from bitboards.
template<int COUT, int KH, int KPAD, int OPAD>
__global__ void conv_first_kernel(
    const int64_t* __restrict__ boards,
    const uint8_t* __restrict__ active,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    const int32_t* __restrict__ black_model,
    const int32_t* __restrict__ white_model,
    const half* __restrict__ weights,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int slots) {

    constexpr int NT = (COUT + WN - 1) / WN;
    int q = static_cast<int>(blockIdx.x);
    int tile_n = q % NT; q /= NT;
    int tile_m = q % ((HW + WM - 1) / WM); q /= ((HW + WM - 1) / WM);
    int slot = q;
    if (slot >= slots || !active[slot]) return;

    const int lane = threadIdx.x & 31;
    const int model = actor_model(slot, current_player, black_model, white_model);
    const int m0 = tile_m * WM;
    const int n0 = tile_n * WN;

    __shared__ half a_smem[WM * WK];
    __shared__ float c_smem[WM * WN];

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    constexpr int KREAL = 3 * KH * KH;
    constexpr int PAD = KH / 2;

    for (int k0 = 0; k0 < KPAD; k0 += WK) {
        for (int t = lane; t < WM * WK; t += WARP) {
            int mr = t / WK;
            int kk = t - mr * WK;
            int pos = m0 + mr;
            int k = k0 + kk;
            half v = __float2half(0.0f);
            if (pos < HW && k < KREAL) {
                int channel = k / (KH * KH);
                int rem = k - channel * KH * KH;
                int kr = rem / KH;
                int kc = rem - kr * KH;
                int orow = pos / BOARD;
                int ocol = pos - orow * BOARD;
                int irow = orow + kr - PAD;
                int icol = ocol + kc - PAD;
                if ((unsigned)irow < BOARD && (unsigned)icol < BOARD) {
                    int ipos = irow * BOARD + icol;
                    v = canonical_input_value(
                        boards, slot, ipos, channel, slots, current_player, stones_left);
                }
            }
            a_smem[t] = v;
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, a_smem, WK);
        const half* bptr = weights + (static_cast<int64_t>(model) * OPAD + n0) * KPAD + k0;
        wmma::load_matrix_sync(b_frag, bptr, KPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(c_smem, c_frag, WN, wmma::mem_row_major);
    __syncwarp();
    for (int t = lane; t < WM * WN; t += WARP) {
        int mr = t / WN;
        int nc = t - mr * WN;
        int pos = m0 + mr;
        int oc = n0 + nc;
        if (pos < HW && oc < COUT) {
            float y = c_smem[t] + __half2float(bias[static_cast<int64_t>(model) * OPAD + oc]);
            y = silu_fast(y);
            output[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + oc] = __float2half_rn(y);
        }
    }
}

template<int CIN, int COUT, int KH, int KPAD, int OPAD>
__global__ void conv_hidden_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ active,
    const int8_t* __restrict__ current_player,
    const int32_t* __restrict__ black_model,
    const int32_t* __restrict__ white_model,
    const half* __restrict__ weights,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int slots) {

    constexpr int NT = (COUT + WN - 1) / WN;
    int q = static_cast<int>(blockIdx.x);
    int tile_n = q % NT; q /= NT;
    int tile_m = q % ((HW + WM - 1) / WM); q /= ((HW + WM - 1) / WM);
    int slot = q;
    if (slot >= slots || !active[slot]) return;

    const int lane = threadIdx.x & 31;
    const int model = actor_model(slot, current_player, black_model, white_model);
    const int m0 = tile_m * WM;
    const int n0 = tile_n * WN;

    __shared__ half a_smem[WM * WK];
    __shared__ float c_smem[WM * WN];

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    constexpr int KREAL = CIN * KH * KH;
    constexpr int PAD = KH / 2;

    for (int k0 = 0; k0 < KPAD; k0 += WK) {
        for (int t = lane; t < WM * WK; t += WARP) {
            int mr = t / WK;
            int kk = t - mr * WK;
            int pos = m0 + mr;
            int k = k0 + kk;
            half v = __float2half(0.0f);
            if (pos < HW && k < KREAL) {
                int channel = k / (KH * KH);
                int rem = k - channel * KH * KH;
                int kr = rem / KH;
                int kc = rem - kr * KH;
                int orow = pos / BOARD;
                int ocol = pos - orow * BOARD;
                int irow = orow + kr - PAD;
                int icol = ocol + kc - PAD;
                if ((unsigned)irow < BOARD && (unsigned)icol < BOARD) {
                    int ipos = irow * BOARD + icol;
                    v = input[(static_cast<int64_t>(slot) * HW + ipos) * STORAGE_C + channel];
                }
            }
            a_smem[t] = v;
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, a_smem, WK);
        const half* bptr = weights + (static_cast<int64_t>(model) * OPAD + n0) * KPAD + k0;
        wmma::load_matrix_sync(b_frag, bptr, KPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(c_smem, c_frag, WN, wmma::mem_row_major);
    __syncwarp();
    for (int t = lane; t < WM * WN; t += WARP) {
        int mr = t / WN;
        int nc = t - mr * WN;
        int pos = m0 + mr;
        int oc = n0 + nc;
        if (pos < HW && oc < COUT) {
            float y = c_smem[t] + __half2float(bias[static_cast<int64_t>(model) * OPAD + oc]);
            y = silu_fast(y);
            output[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + oc] = __float2half_rn(y);
        }
    }
}

__device__ __forceinline__ void choose_better(float candidate_v, int candidate_i, float& best_v, int& best_i) {
    if (candidate_v > best_v || (candidate_v == best_v && candidate_i < best_i)) {
        best_v = candidate_v;
        best_i = candidate_i;
    }
}

__global__ void policy_argmax_kernel(
    const half* __restrict__ features,
    const half* __restrict__ policy_weight,
    const int64_t* __restrict__ boards,
    const uint8_t* __restrict__ active,
    const int8_t* __restrict__ current_player,
    const int32_t* __restrict__ black_model,
    const int32_t* __restrict__ white_model,
    int16_t* __restrict__ actions,
    int slots) {

    int slot = static_cast<int>(blockIdx.x);
    if (slot >= slots || !active[slot]) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int model = actor_model(slot, current_player, black_model, white_model);

    const half* w = policy_weight + static_cast<int64_t>(model) * STORAGE_C;
    const half2* w2 = reinterpret_cast<const half2*>(w);
    float best_v = -FLT_MAX;
    int best_i = HW;

    for (int pos = tid; pos < HW; pos += blockDim.x) {
        if (occupied_at(boards, slot, pos, slots)) continue;
        const half* fp = features + (static_cast<int64_t>(slot) * HW + pos) * STORAGE_C;
        const half2* f2 = reinterpret_cast<const half2*>(fp);
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < STORAGE_C / 2; ++k) {
            float2 a = __half22float2(f2[k]);
            float2 b = __half22float2(w2[k]);
            acc = fmaf(a.x, b.x, acc);
            acc = fmaf(a.y, b.y, acc);
        }
        choose_better(acc, pos, best_v, best_i);
    }

    for (int d = 16; d > 0; d >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, best_v, d);
        int oi = __shfl_down_sync(0xffffffffu, best_i, d);
        choose_better(ov, oi, best_v, best_i);
    }

    __shared__ float warp_v[8];
    __shared__ int warp_i[8];
    if (lane == 0) {
        warp_v[warp] = best_v;
        warp_i[warp] = best_i;
    }
    __syncthreads();

    if (warp == 0) {
        float v = lane < (blockDim.x >> 5) ? warp_v[lane] : -FLT_MAX;
        int i = lane < (blockDim.x >> 5) ? warp_i[lane] : HW;
        for (int d = 16; d > 0; d >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, d);
            int oi = __shfl_down_sync(0xffffffffu, i, d);
            choose_better(ov, oi, v, i);
        }
        if (lane == 0) actions[slot] = static_cast<int16_t>(i < HW ? i : 0);
    }
}

__device__ __forceinline__ int64_t pair_prefix(int i, int n) {
    return static_cast<int64_t>(i) * (2LL * n - i - 1) / 2;
}

__device__ __forceinline__ void pair_from_index(int pair_idx, int n, int& a, int& b) {
    double x = static_cast<double>(2 * n - 1);
    double disc = x * x - 8.0 * static_cast<double>(pair_idx);
    int i = static_cast<int>(floor((x - sqrt(disc)) * 0.5));
    if (i < 0) i = 0;
    if (i > n - 2) i = n - 2;
    while (i > 0 && pair_prefix(i, n) > pair_idx) --i;
    while (i + 1 < n - 1 && pair_prefix(i + 1, n) <= pair_idx) ++i;
    int64_t base = pair_prefix(i, n);
    a = i;
    b = i + 1 + static_cast<int>(pair_idx - base);
}

__device__ __forceinline__ void decode_game_models(int game_id, int n, int& black, int& white) {
    int pair_idx = game_id >> 1;
    int a, b;
    pair_from_index(pair_idx, n, a, b);
    if ((game_id & 1) == 0) {
        black = a; white = b;
    } else {
        black = b; white = a;
    }
}

__device__ __forceinline__ void clear_slot_boards(int64_t* boards, int slot, int slots) {
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int w = 0; w < WORDS; ++w) {
            *board_ptr(boards, c, w, slots) = *board_ptr(boards, c, w, slots); // keep compiler address math hot
            board_ptr(boards, c, w, slots)[slot] = 0ULL;
        }
    }
}

__device__ __forceinline__ void load_queue_item(
    int slot,
    int queue_idx,
    const int32_t* game_ids,
    int total,
    int num_models,
    int64_t* boards,
    uint8_t* active,
    int8_t* current_player,
    int8_t* stones_left,
    int16_t* move_count,
    int32_t* black_model,
    int32_t* white_model,
    int32_t* current_queue,
    int slots) {
    if (queue_idx >= total) {
        active[slot] = 0;
        return;
    }
    int game_id = game_ids[queue_idx];
    int black, white;
    decode_game_models(game_id, num_models, black, white);
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int w = 0; w < WORDS; ++w) {
            board_ptr(boards, c, w, slots)[slot] = 0ULL;
        }
    }
    current_player[slot] = 1;
    stones_left[slot] = 1;
    move_count[slot] = 0;
    black_model[slot] = black;
    white_model[slot] = white;
    current_queue[slot] = queue_idx;
    active[slot] = 1;
}

__device__ __forceinline__ bool has_actor_stone(
    const int64_t* boards, int actor, int slot, int row, int col, int slots) {
    if ((unsigned)row >= BOARD || (unsigned)col >= BOARD) return false;
    int pos = row * BOARD + col;
    int color = actor == 1 ? 0 : 1;
    return bit_at(boards, color, slot, pos, slots);
}

__device__ __forceinline__ bool check_win_after_move(
    const int64_t* boards, int actor, int slot, int action, int slots) {
    int row = action / BOARD;
    int col = action - row * BOARD;
    const int dr[4] = {1, 0, 1, 1};
    const int dc[4] = {0, 1, 1, -1};
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        int count = 1;
        #pragma unroll
        for (int s = 1; s <= 5; ++s) {
            if (!has_actor_stone(boards, actor, slot, row + dr[d] * s, col + dc[d] * s, slots)) break;
            ++count;
        }
        #pragma unroll
        for (int s = 1; s <= 5; ++s) {
            if (!has_actor_stone(boards, actor, slot, row - dr[d] * s, col - dc[d] * s, slots)) break;
            ++count;
        }
        if (count >= 6) return true;
    }
    return false;
}

__global__ void init_counters_kernel(int32_t* counters, int initial_active) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        counters[0] = initial_active; // next queue index
        counters[1] = initial_active; // active slots
        counters[2] = 0;              // completed games
    }
}

__global__ void init_slots_kernel(
    const int32_t* game_ids,
    int total,
    int num_models,
    int64_t* boards,
    uint8_t* active,
    int8_t* current_player,
    int8_t* stones_left,
    int16_t* move_count,
    int32_t* black_model,
    int32_t* white_model,
    int32_t* current_queue,
    int slots) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;
    if (slot < total) {
        load_queue_item(
            slot, slot, game_ids, total, num_models, boards, active, current_player,
            stones_left, move_count, black_model, white_model, current_queue, slots);
    } else {
        active[slot] = 0;
    }
}

__global__ void game_step_kernel(
    const int16_t* __restrict__ actions,
    const int32_t* __restrict__ game_ids,
    int total,
    int num_models,
    int64_t* __restrict__ boards,
    uint8_t* __restrict__ active,
    int8_t* __restrict__ current_player,
    int8_t* __restrict__ stones_left,
    int16_t* __restrict__ move_count,
    int32_t* __restrict__ black_model,
    int32_t* __restrict__ white_model,
    int32_t* __restrict__ current_queue,
    int8_t* __restrict__ results,
    int32_t* __restrict__ counters,
    int slots) {

    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots || !active[slot]) return;

    int action = static_cast<int>(actions[slot]);
    int actor = static_cast<int>(current_player[slot]);
    int color = actor == 1 ? 0 : 1;
    int word = action >> 6;
    int bit = action & 63;
    board_ptr(boards, color, word, slots)[slot] |= (1ULL << bit);

    int moves = static_cast<int>(move_count[slot]) + 1;
    move_count[slot] = static_cast<int16_t>(moves);
    bool won = check_win_after_move(boards, actor, slot, action, slots);
    bool draw = !won && moves >= HW;

    if (won || draw) {
        int qidx = current_queue[slot];
        results[qidx] = won ? static_cast<int8_t>(actor) : static_cast<int8_t>(0);
        atomicAdd(counters + 2, 1);
        int next = atomicAdd(counters + 0, 1);
        if (next < total) {
            load_queue_item(
                slot, next, game_ids, total, num_models, boards, active, current_player,
                stones_left, move_count, black_model, white_model, current_queue, slots);
        } else {
            active[slot] = 0;
            atomicSub(counters + 1, 1);
        }
        return;
    }

    int left = static_cast<int>(stones_left[slot]) - 1;
    if (left > 0) {
        stones_left[slot] = static_cast<int8_t>(left);
    } else {
        current_player[slot] = static_cast<int8_t>(-actor);
        stones_left[slot] = 2;
    }
}

__global__ void set_condition_kernel(cudaGraphConditionalHandle handle, const int32_t* counters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cudaGraphSetConditional(handle, counters[1] > 0 ? 1U : 0U);
    }
}

cudaGraphNode_t add_kernel_node(
    cudaGraph_t graph,
    cudaGraphNode_t dependency,
    void* func,
    dim3 grid,
    dim3 block,
    void** args) {
    cudaKernelNodeParams p{};
    p.func = func;
    p.gridDim = grid;
    p.blockDim = block;
    p.sharedMemBytes = 0;
    p.kernelParams = args;
    p.extra = nullptr;
    cudaGraphNode_t node{};
    if (dependency) {
        CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, &dependency, 1, &p));
    } else {
        CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p));
    }
    return node;
}

void check_half_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " musi być CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " musi być FP16");
    TORCH_CHECK(t.is_contiguous(), name, " musi być contiguous");
}

} // namespace

std::vector<torch::Tensor> run_championship_cuda(
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases,
    torch::Tensor policy_weight,
    torch::Tensor game_ids,
    int64_t num_models,
    int64_t slots64) {

    TORCH_CHECK(weights.size() == 8, "Native engine wymaga dokładnie 8 warstw conv");
    TORCH_CHECK(biases.size() == 8, "Native engine wymaga bias dla 8 warstw conv");
    TORCH_CHECK(game_ids.is_cuda() && game_ids.scalar_type() == torch::kInt32 && game_ids.is_contiguous(),
                "game_ids musi być CUDA int32 contiguous");
    TORCH_CHECK(num_models >= 2 && num_models <= 65535, "Niepoprawna liczba modeli");
    TORCH_CHECK(slots64 >= 32 && slots64 <= 32768, "slots musi być w zakresie 32..32768");
    for (int i = 0; i < 8; ++i) {
        check_half_cuda_contiguous(weights[i], "weights");
        check_half_cuda_contiguous(biases[i], "biases");
    }
    check_half_cuda_contiguous(policy_weight, "policy_weight");

    const int slots = static_cast<int>(slots64);
    const int total = static_cast<int>(game_ids.numel());
    const int models = static_cast<int>(num_models);
    TORCH_CHECK(total > 0, "Brak gier do rozegrania");
    TORCH_CHECK(weights[0].size(0) == models, "Liczba modeli w wagach != num_models");

    // Exact architecture this engine is specialized for.
    const int expected_o[8] = {32, 32, 64, 64, 64, 96, 96, 96};
    const int expected_k[8] = {1600, 288, 288, 576, 576, 576, 864, 864};
    for (int i = 0; i < 8; ++i) {
        TORCH_CHECK(weights[i].dim() == 3, "Packed weight musi mieć [M,O,K]");
        TORCH_CHECK(weights[i].size(1) == expected_o[i], "Niepoprawne O w warstwie ", i);
        TORCH_CHECK(weights[i].size(2) == expected_k[i], "Niepoprawne KPAD w warstwie ", i);
        TORCH_CHECK(biases[i].sizes() == torch::IntArrayRef({models, expected_o[i]}),
                    "Niepoprawny bias w warstwie ", i);
    }
    TORCH_CHECK(policy_weight.sizes() == torch::IntArrayRef({models, STORAGE_C}),
                "policy_weight musi mieć [M,96]");

    const int device = game_ids.get_device();
    CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(device);

    auto dev = game_ids.device();
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opts_i8  = torch::TensorOptions().dtype(torch::kInt8).device(dev);
    auto opts_i16 = torch::TensorOptions().dtype(torch::kInt16).device(dev);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opts_f16 = torch::TensorOptions().dtype(torch::kFloat16).device(dev);

    torch::Tensor boards = torch::zeros({2, WORDS, slots}, opts_i64);
    torch::Tensor active = torch::zeros({slots}, opts_u8);
    torch::Tensor current_player = torch::empty({slots}, opts_i8);
    torch::Tensor stones_left = torch::empty({slots}, opts_i8);
    torch::Tensor move_count = torch::empty({slots}, opts_i16);
    torch::Tensor black_model = torch::empty({slots}, opts_i32);
    torch::Tensor white_model = torch::empty({slots}, opts_i32);
    torch::Tensor current_queue = torch::empty({slots}, opts_i32);
    torch::Tensor actions = torch::empty({slots}, opts_i16);
    torch::Tensor counters = torch::zeros({3}, opts_i32);
    torch::Tensor results = torch::empty({total}, opts_i8);
    torch::Tensor feat_a = torch::empty({slots, HW, STORAGE_C}, opts_f16);
    torch::Tensor feat_b = torch::empty({slots, HW, STORAGE_C}, opts_f16);

    int32_t* game_p = game_ids.data_ptr<int32_t>();
    int64_t* boards_p = boards.data_ptr<int64_t>();
    uint8_t* active_p = active.data_ptr<uint8_t>();
    int8_t* player_p = current_player.data_ptr<int8_t>();
    int8_t* stones_p = stones_left.data_ptr<int8_t>();
    int16_t* moves_p = move_count.data_ptr<int16_t>();
    int32_t* black_p = black_model.data_ptr<int32_t>();
    int32_t* white_p = white_model.data_ptr<int32_t>();
    int32_t* queue_p = current_queue.data_ptr<int32_t>();
    int16_t* actions_p = actions.data_ptr<int16_t>();
    int32_t* counters_p = counters.data_ptr<int32_t>();
    int8_t* results_p = results.data_ptr<int8_t>();
    half* a_p = reinterpret_cast<half*>(feat_a.data_ptr<at::Half>());
    half* b_p = reinterpret_cast<half*>(feat_b.data_ptr<at::Half>());

    std::vector<const half*> w(8), bs(8);
    for (int i = 0; i < 8; ++i) {
        w[i] = reinterpret_cast<const half*>(weights[i].data_ptr<at::Half>());
        bs[i] = reinterpret_cast<const half*>(biases[i].data_ptr<at::Half>());
    }
    const half* policy_p = reinterpret_cast<const half*>(policy_weight.data_ptr<at::Half>());

    int initial_active = total < slots ? total : slots;
    init_counters_kernel<<<1, 1, 0, stream>>>(counters_p, initial_active);
    init_slots_kernel<<<(slots + 255) / 256, 256, 0, stream>>>(
        game_p, total, models, boards_p, active_p, player_p, stones_p, moves_p,
        black_p, white_p, queue_p, slots);
    CUDA_CHECK(cudaGetLastError());

    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    cudaGraphConditionalHandle handle{};
    CUDA_CHECK(cudaGraphConditionalHandleCreate(
        &handle, graph, 1, cudaGraphCondAssignDefault));

    cudaGraphNodeParams cp{};
    cp.type = cudaGraphNodeTypeConditional;
    cp.conditional.handle = handle;
    cp.conditional.type = cudaGraphCondTypeWhile;
    cp.conditional.size = 1;
    cudaGraphNode_t conditional_node{};
    CUDA_CHECK(cudaGraphAddNode(&conditional_node, graph, nullptr, 0, &cp));
    cudaGraph_t body = cp.conditional.phGraph_out[0];

    cudaGraphNode_t dep{};
    const int MTILES = (HW + WM - 1) / WM;

    // Layer 0: 3 -> 32, 23x23, K padded 1587 -> 1600.
    {
        void* args[] = {&boards_p, &active_p, &player_p, &stones_p, &black_p, &white_p,
                        &w[0], &bs[0], &a_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_first_kernel<32,23,1600,32>,
                              dim3(slots * MTILES * 2), dim3(32), args);
    }
    // 32 -> 32
    {
        void* args[] = {&a_p, &active_p, &player_p, &black_p, &white_p,
                        &w[1], &bs[1], &b_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_hidden_kernel<32,32,3,288,32>,
                              dim3(slots * MTILES * 2), dim3(32), args);
    }
    // 32 -> 64
    {
        void* args[] = {&b_p, &active_p, &player_p, &black_p, &white_p,
                        &w[2], &bs[2], &a_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_hidden_kernel<32,64,3,288,64>,
                              dim3(slots * MTILES * 4), dim3(32), args);
    }
    // 64 -> 64
    {
        void* args[] = {&a_p, &active_p, &player_p, &black_p, &white_p,
                        &w[3], &bs[3], &b_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_hidden_kernel<64,64,3,576,64>,
                              dim3(slots * MTILES * 4), dim3(32), args);
    }
    // 64 -> 64
    {
        void* args[] = {&b_p, &active_p, &player_p, &black_p, &white_p,
                        &w[4], &bs[4], &a_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_hidden_kernel<64,64,3,576,64>,
                              dim3(slots * MTILES * 4), dim3(32), args);
    }
    // 64 -> 96
    {
        void* args[] = {&a_p, &active_p, &player_p, &black_p, &white_p,
                        &w[5], &bs[5], &b_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_hidden_kernel<64,96,3,576,96>,
                              dim3(slots * MTILES * 6), dim3(32), args);
    }
    // 96 -> 96
    {
        void* args[] = {&b_p, &active_p, &player_p, &black_p, &white_p,
                        &w[6], &bs[6], &a_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_hidden_kernel<96,96,3,864,96>,
                              dim3(slots * MTILES * 6), dim3(32), args);
    }
    // 96 -> 96
    {
        void* args[] = {&a_p, &active_p, &player_p, &black_p, &white_p,
                        &w[7], &bs[7], &b_p, &slots};
        dep = add_kernel_node(body, dep, (void*)conv_hidden_kernel<96,96,3,864,96>,
                              dim3(slots * MTILES * 6), dim3(32), args);
    }
    // Policy 1x1 + legal mask + argmax fused. Bias omitted: scalar per model cancels in argmax.
    {
        void* args[] = {&b_p, &policy_p, &boards_p, &active_p, &player_p, &black_p, &white_p,
                        &actions_p, &slots};
        dep = add_kernel_node(body, dep, (void*)policy_argmax_kernel,
                              dim3(slots), dim3(256), args);
    }
    // Game step + win/draw + device-side refill + result store.
    {
        void* args[] = {&actions_p, &game_p, &total, &models, &boards_p, &active_p,
                        &player_p, &stones_p, &moves_p, &black_p, &white_p, &queue_p,
                        &results_p, &counters_p, &slots};
        dep = add_kernel_node(body, dep, (void*)game_step_kernel,
                              dim3((slots + 255) / 256), dim3(256), args);
    }
    // Device decides whether the graph performs another move. No host round-trip.
    {
        void* args[] = {&handle, &counters_p};
        dep = add_kernel_node(body, dep, (void*)set_condition_kernel,
                              dim3(1), dim3(1), args);
    }

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));

    return {results, counters};
}
