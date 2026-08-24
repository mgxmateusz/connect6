#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>

#include "native_rollout_bot.cuh"

namespace wmma = nvcuda::wmma;

namespace {

constexpr int BOARD = 19;
constexpr int HW = BOARD * BOARD;
constexpr int STORAGE_C = 96;
constexpr int GROUP_CHANNELS = 8;
constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int CONV_THREADS = WARP * WARPS_PER_BLOCK;
constexpr int POLICY_THREADS = 256;
constexpr int NORM_THREADS = 256;
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
constexpr int MTILES = (HW + WM - 1) / WM;
constexpr int MGROUPS = (MTILES + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
constexpr int UNKNOWN_EPISODE_RESULT = 2;
constexpr int UNKNOWN_TERMINAL_MOVE = -1;

constexpr uint8_t TABLE_SELF = 0;
constexpr uint8_t TABLE_HISTORY = 1;
constexpr uint8_t TABLE_BOT = 2;

constexpr uint8_t MODE_CURRENT = 0;
constexpr uint8_t MODE_HISTORY = 1;
constexpr uint8_t MODE_BOT_V1 = 2;
constexpr uint8_t MODE_BOT_V2 = 3;
constexpr uint8_t MODE_RANDOM_OPENING = 4;

enum CounterIndex : int {
    C_COMPLETED_POSITIONS = 0,
    C_BUFFER_COUNT = 1,
    C_NEXT_EPISODE_ID = 2,
    C_GAMES = 3,
    C_BLACK_WINS = 4,
    C_WHITE_WINS = 5,
    C_DRAWS = 6,
    C_GAME_LENGTH_SUM = 7,
    C_HISTORY_GAMES = 8,
    C_HISTORY_WINS = 9,
    C_HISTORY_LOSSES = 10,
    C_HISTORY_DRAWS = 11,
    C_BOT_GAMES = 12,
    C_BOT_WINS = 13,
    C_BOT_LOSSES = 14,
    C_BOT_DRAWS = 15,
    C_GRAPH_STEPS = 16,
    C_ERROR = 17,
    C_COUNT = 18,
};

#define CUDA_CHECK(EXPR) do { \
    cudaError_t _e = (EXPR); \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e)); \
} while (0)

__device__ __forceinline__ unsigned long long atomic_add_i64(
    int64_t* address, unsigned long long value) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address), value);
}

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float random_uniform01(
    unsigned long long seed,
    int slot,
    int64_t counter,
    unsigned long long tag) {
    unsigned long long x = seed;
    x ^= static_cast<unsigned long long>(slot + 1) * 0xd2b74407b1ce6e93ULL;
    x ^= static_cast<unsigned long long>(counter + 1) * 0xca5a826395121157ULL;
    x ^= tag * 0x9e3779b97f4a7c15ULL;
    const unsigned long long h = splitmix64(x);
    const unsigned int mantissa = static_cast<unsigned int>((h >> 40) & 0xFFFFFFULL);
    return (static_cast<float>(mantissa) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ int view_to_canonical_pos(int pos, int phase) {
    int r = pos / BOARD;
    int c = pos - r * BOARD;
    const int k = phase & 3;
    const bool flip = phase >= 4;
    if (flip) c = BOARD - 1 - c;

    const int old_r = r;
    const int old_c = c;
    if (k == 1) {
        r = old_c;
        c = BOARD - 1 - old_r;
    } else if (k == 2) {
        r = BOARD - 1 - old_r;
        c = BOARD - 1 - old_c;
    } else if (k == 3) {
        r = BOARD - 1 - old_c;
        c = old_r;
    }
    return r * BOARD + c;
}

__device__ __forceinline__ half canonical_input_value(
    const int8_t* boards,
    int slot,
    int view_pos,
    int channel,
    const int8_t* current_player,
    const int8_t* stones_left,
    const int32_t* symmetry_phase,
    int symmetry_enabled) {
    if (channel == 2) return __float2half(1.0f);
    if (channel == 3)
        return __float2half(stones_left[slot] == 1 ? 1.0f : 0.0f);

    const int phase = symmetry_enabled ? symmetry_phase[0] : 0;
    const int canonical_pos = view_to_canonical_pos(view_pos, phase);
    const int8_t value = boards[static_cast<int64_t>(slot) * HW + canonical_pos];
    const int8_t player = current_player[slot];
    const bool set = channel == 0 ? value == player : value == -player;
    return __float2half(set ? 1.0f : 0.0f);
}

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void prepare_actor_kernel(
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    const int16_t* __restrict__ move_count,
    const int64_t* __restrict__ rng_counter,
    const uint8_t* __restrict__ table_kind,
    const int32_t* __restrict__ opponent_model,
    const int8_t* __restrict__ current_color,
    const uint8_t* __restrict__ bot_version,
    uint8_t* __restrict__ actor_mode,
    uint8_t* __restrict__ cnn_active,
    int32_t* __restrict__ actor_model,
    unsigned long long seed,
    float random_opening_fraction,
    int slots) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;

    const bool fresh_black = move_count[slot] == 0
        && current_player[slot] == 1
        && stones_left[slot] == 1;
    bool forced_random = false;
    if (fresh_black && random_opening_fraction > 0.0f) {
        if (random_opening_fraction >= 1.0f) {
            forced_random = true;
        } else {
            forced_random = random_uniform01(
                seed, slot, rng_counter[slot], 0x4f50454eULL) < random_opening_fraction;
        }
    }

    uint8_t mode = MODE_CURRENT;
    int32_t model = 0;
    if (forced_random) {
        mode = MODE_RANDOM_OPENING;
    } else {
        const uint8_t kind = table_kind[slot];
        if (kind == TABLE_SELF || current_player[slot] == current_color[slot]) {
            mode = MODE_CURRENT;
            model = 0;
        } else if (kind == TABLE_HISTORY) {
            mode = MODE_HISTORY;
            model = opponent_model[slot];
        } else if (kind == TABLE_BOT) {
            mode = bot_version[slot] == 2 ? MODE_BOT_V2 : MODE_BOT_V1;
        }
    }

    actor_mode[slot] = mode;
    actor_model[slot] = model;
    cnn_active[slot] = (mode == MODE_CURRENT || mode == MODE_HISTORY) ? 1 : 0;
}

template<int COUT, int KH, int KPAD, int OPAD>
__global__ void conv_first_kernel(
    const int8_t* __restrict__ boards,
    const uint8_t* __restrict__ cnn_active,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    const int32_t* __restrict__ actor_model,
    const int32_t* __restrict__ symmetry_phase,
    int symmetry_enabled,
    const half* __restrict__ weights,
    half* __restrict__ output,
    int slots) {
    constexpr int NT = (COUT + WN - 1) / WN;
    int q = static_cast<int>(blockIdx.x);
    const int tile_n = q % NT; q /= NT;
    const int group_m = q % MGROUPS; q /= MGROUPS;
    const int slot = q;
    if (slot >= slots || !cnn_active[slot]) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int tile_m = group_m * WARPS_PER_BLOCK + warp;
    const bool valid_warp = tile_m < MTILES;
    const int model = actor_model[slot];
    const int m0 = tile_m * WM;
    const int n0 = tile_n * WN;

    __shared__ half a_smem[WARPS_PER_BLOCK][WM * WK];
    __shared__ half b_smem[WK * WN];
    __shared__ float c_smem[WARPS_PER_BLOCK][WM * WN];

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    constexpr int KREAL = 4 * KH * KH;
    constexpr int PAD = KH / 2;
    for (int k0 = 0; k0 < KPAD; k0 += WK) {
        for (int t = tid; t < WK * WN; t += CONV_THREADS) {
            const int row = t & 15;
            const int col = t >> 4;
            const int out_channel = n0 + col;
            b_smem[t] = out_channel < COUT
                ? weights[(static_cast<int64_t>(model) * OPAD + out_channel) * KPAD + k0 + row]
                : __float2half(0.0f);
        }
        if (valid_warp) {
            for (int t = lane; t < WM * WK; t += WARP) {
                const int mr = t >> 4;
                const int kk = t & 15;
                const int pos = m0 + mr;
                const int k = k0 + kk;
                half v = __float2half(0.0f);
                if (pos < HW && k < KREAL) {
                    const int channel = k / (KH * KH);
                    const int rem = k - channel * KH * KH;
                    const int kr = rem / KH;
                    const int kc = rem - kr * KH;
                    const int orow = pos / BOARD;
                    const int ocol = pos - orow * BOARD;
                    const int irow = orow + kr - PAD;
                    const int icol = ocol + kc - PAD;
                    if ((unsigned)irow < BOARD && (unsigned)icol < BOARD) {
                        v = canonical_input_value(
                            boards, slot, irow * BOARD + icol, channel,
                            current_player, stones_left, symmetry_phase, symmetry_enabled);
                    }
                }
                a_smem[warp][t] = v;
            }
        }
        __syncthreads();
        if (valid_warp) {
            wmma::load_matrix_sync(a_frag, a_smem[warp], WK);
            wmma::load_matrix_sync(b_frag, b_smem, WK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    if (valid_warp) {
        wmma::store_matrix_sync(c_smem[warp], c_frag, WN, wmma::mem_row_major);
        __syncwarp();
        for (int t = lane; t < WM * WN; t += WARP) {
            const int mr = t >> 4;
            const int nc = t & 15;
            const int pos = m0 + mr;
            const int oc = n0 + nc;
            if (pos < HW && oc < COUT) {
                output[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + oc]
                    = __float2half_rn(c_smem[warp][t]);
            }
        }
    }
}

template<int CIN, int COUT, int KH, int KPAD, int OPAD>
__global__ void conv_hidden_kernel(
    const half* __restrict__ input,
    const uint8_t* __restrict__ cnn_active,
    const int32_t* __restrict__ actor_model,
    const half* __restrict__ weights,
    half* __restrict__ output,
    int slots) {
    constexpr int NT = (COUT + WN - 1) / WN;
    int q = static_cast<int>(blockIdx.x);
    const int tile_n = q % NT; q /= NT;
    const int group_m = q % MGROUPS; q /= MGROUPS;
    const int slot = q;
    if (slot >= slots || !cnn_active[slot]) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int tile_m = group_m * WARPS_PER_BLOCK + warp;
    const bool valid_warp = tile_m < MTILES;
    const int model = actor_model[slot];
    const int m0 = tile_m * WM;
    const int n0 = tile_n * WN;

    __shared__ half a_smem[WARPS_PER_BLOCK][WM * WK];
    __shared__ half b_smem[WK * WN];
    __shared__ float c_smem[WARPS_PER_BLOCK][WM * WN];
    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    constexpr int KREAL = CIN * KH * KH;
    constexpr int PAD = KH / 2;
    for (int k0 = 0; k0 < KPAD; k0 += WK) {
        for (int t = tid; t < WK * WN; t += CONV_THREADS) {
            const int row = t & 15;
            const int col = t >> 4;
            const int out_channel = n0 + col;
            b_smem[t] = out_channel < COUT
                ? weights[(static_cast<int64_t>(model) * OPAD + out_channel) * KPAD + k0 + row]
                : __float2half(0.0f);
        }
        if (valid_warp) {
            for (int t = lane; t < WM * WK; t += WARP) {
                const int mr = t >> 4;
                const int kk = t & 15;
                const int pos = m0 + mr;
                const int k = k0 + kk;
                half v = __float2half(0.0f);
                if (pos < HW && k < KREAL) {
                    const int channel = k / (KH * KH);
                    const int rem = k - channel * KH * KH;
                    const int kr = rem / KH;
                    const int kc = rem - kr * KH;
                    const int orow = pos / BOARD;
                    const int ocol = pos - orow * BOARD;
                    const int irow = orow + kr - PAD;
                    const int icol = ocol + kc - PAD;
                    if ((unsigned)irow < BOARD && (unsigned)icol < BOARD) {
                        const int ipos = irow * BOARD + icol;
                        v = input[(static_cast<int64_t>(slot) * HW + ipos) * STORAGE_C + channel];
                    }
                }
                a_smem[warp][t] = v;
            }
        }
        __syncthreads();
        if (valid_warp) {
            wmma::load_matrix_sync(a_frag, a_smem[warp], WK);
            wmma::load_matrix_sync(b_frag, b_smem, WK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    if (valid_warp) {
        wmma::store_matrix_sync(c_smem[warp], c_frag, WN, wmma::mem_row_major);
        __syncwarp();
        for (int t = lane; t < WM * WN; t += WARP) {
            const int mr = t >> 4;
            const int nc = t & 15;
            const int pos = m0 + mr;
            const int oc = n0 + nc;
            if (pos < HW && oc < COUT) {
                output[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + oc]
                    = __float2half_rn(c_smem[warp][t]);
            }
        }
    }
}

template<int CHANNELS>
__global__ void groupnorm_silu_kernel(
    half* __restrict__ features,
    const uint8_t* __restrict__ cnn_active,
    const int32_t* __restrict__ actor_model,
    const half* __restrict__ norm_weight,
    const half* __restrict__ norm_bias,
    int slots) {
    constexpr int GROUPS = CHANNELS / GROUP_CHANNELS;
    constexpr int ITEMS = HW * GROUP_CHANNELS;
    const int q = static_cast<int>(blockIdx.x);
    const int group = q % GROUPS;
    const int slot = q / GROUPS;
    if (slot >= slots || !cnn_active[slot]) return;

    const int tid = threadIdx.x;
    const int model = actor_model[slot];
    const int c0 = group * GROUP_CHANNELS;
    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int i = tid; i < ITEMS; i += blockDim.x) {
        const int pos = i / GROUP_CHANNELS;
        const int c = c0 + (i - pos * GROUP_CHANNELS);
        const float x = __half2float(
            features[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + c]);
        local_sum += x;
        local_sq = fmaf(x, x, local_sq);
    }
    __shared__ float sums[NORM_THREADS];
    __shared__ float sqs[NORM_THREADS];
    sums[tid] = local_sum;
    sqs[tid] = local_sq;
    __syncthreads();
    for (int stride = NORM_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            sqs[tid] += sqs[tid + stride];
        }
        __syncthreads();
    }
    const float mean = sums[0] / static_cast<float>(ITEMS);
    const float second = sqs[0] / static_cast<float>(ITEMS);
    const float variance = fmaxf(0.0f, second - mean * mean);
    const float inv_std = rsqrtf(variance + 1e-5f);
    for (int i = tid; i < ITEMS; i += blockDim.x) {
        const int pos = i / GROUP_CHANNELS;
        const int c = c0 + (i - pos * GROUP_CHANNELS);
        const int64_t index = (static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + c;
        const float x = __half2float(features[index]);
        const float gamma = __half2float(
            norm_weight[static_cast<int64_t>(model) * CHANNELS + c]);
        const float beta = __half2float(
            norm_bias[static_cast<int64_t>(model) * CHANNELS + c]);
        features[index] = __float2half_rn(silu_fast((x - mean) * inv_std * gamma + beta));
    }
}

__device__ __forceinline__ float head_dot(
    const half* features, const half* weight, int slot, int pos) {
    const half2* f2 = reinterpret_cast<const half2*>(
        features + (static_cast<int64_t>(slot) * HW + pos) * STORAGE_C);
    const half2* w2 = reinterpret_cast<const half2*>(weight);
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < STORAGE_C / 2; ++k) {
        const float2 a = __half22float2(f2[k]);
        const float2 b = __half22float2(w2[k]);
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
    }
    return acc;
}

__device__ __forceinline__ void choose_gumbel(
    float candidate_key, int candidate_action, float& best_key, int& best_action) {
    if (candidate_key > best_key ||
        (candidate_key == best_key && candidate_action < best_action)) {
        best_key = candidate_key;
        best_action = candidate_action;
    }
}

__global__ void policy_sample_kernel(
    const half* __restrict__ features,
    const half* __restrict__ policy_weight,
    const half* __restrict__ policy_bias,
    const half* __restrict__ value_weight,
    const half* __restrict__ value_bias,
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    const int16_t* __restrict__ move_count,
    const int64_t* __restrict__ rng_counter,
    const uint8_t* __restrict__ actor_mode,
    const int32_t* __restrict__ actor_model,
    const int32_t* __restrict__ symmetry_phase,
    int symmetry_enabled,
    int16_t* __restrict__ actions_view,
    int16_t* __restrict__ actions_canonical,
    int32_t* __restrict__ current_episode_ids,
    int32_t* __restrict__ segment_positions,
    int8_t* __restrict__ buffer_boards,
    int8_t* __restrict__ buffer_players,
    int8_t* __restrict__ buffer_stones_left,
    int16_t* __restrict__ buffer_move_counts,
    int16_t* __restrict__ buffer_actions,
    float* __restrict__ buffer_logprobs,
    float* __restrict__ buffer_values,
    int32_t* __restrict__ buffer_episode_ids,
    int64_t* __restrict__ counters,
    int64_t buffer_capacity,
    float temperature,
    unsigned long long seed,
    int slots) {
    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= slots) return;
    const uint8_t mode = actor_mode[slot];
    if (mode != MODE_CURRENT && mode != MODE_HISTORY) return;

    const int tid = threadIdx.x;
    const int model = actor_model[slot];
    const int phase = symmetry_enabled ? symmetry_phase[0] : 0;
    const float inv_temp = 1.0f / fmaxf(temperature, 1e-4f);
    const half* pw = policy_weight + static_cast<int64_t>(model) * STORAGE_C;
    const half* vw = value_weight + static_cast<int64_t>(model) * STORAGE_C;
    const float pb = __half2float(policy_bias[model]);
    const float vb = __half2float(value_bias[model]);

    float local_best_key = -FLT_MAX;
    int local_best_action = HW;
    float local_max_logit = -FLT_MAX;
    float local_value_sum = 0.0f;
    for (int view_pos = tid; view_pos < HW; view_pos += blockDim.x) {
        const int canonical_pos = view_to_canonical_pos(view_pos, phase);
        if (boards[static_cast<int64_t>(slot) * HW + canonical_pos] == 0) {
            const float logit = (head_dot(features, pw, slot, view_pos) + pb) * inv_temp;
            const float u = random_uniform01(
                seed, slot, rng_counter[slot],
                static_cast<unsigned long long>(view_pos) + 0x504f4c49ULL);
            const float safe_u = fminf(fmaxf(u, 1e-7f), 1.0f - 1e-7f);
            choose_gumbel(-logf(-logf(safe_u)) + logit, view_pos,
                          local_best_key, local_best_action);
            local_max_logit = fmaxf(local_max_logit, logit);
        }
        if (mode == MODE_CURRENT)
            local_value_sum += head_dot(features, vw, slot, view_pos) + vb;
    }

    __shared__ float best_keys[POLICY_THREADS];
    __shared__ int best_actions[POLICY_THREADS];
    __shared__ float max_logits[POLICY_THREADS];
    __shared__ float value_sums[POLICY_THREADS];
    __shared__ float exp_sums[POLICY_THREADS];
    __shared__ int64_t transition_idx;
    best_keys[tid] = local_best_key;
    best_actions[tid] = local_best_action;
    max_logits[tid] = local_max_logit;
    value_sums[tid] = local_value_sum;
    __syncthreads();
    for (int stride = POLICY_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            choose_gumbel(best_keys[tid + stride], best_actions[tid + stride],
                          best_keys[tid], best_actions[tid]);
            max_logits[tid] = fmaxf(max_logits[tid], max_logits[tid + stride]);
            value_sums[tid] += value_sums[tid + stride];
        }
        __syncthreads();
    }

    const int chosen_view = best_actions[0] < HW ? best_actions[0] : 0;
    if (tid == 0) {
        actions_view[slot] = static_cast<int16_t>(chosen_view);
        actions_canonical[slot] = static_cast<int16_t>(view_to_canonical_pos(chosen_view, phase));
        transition_idx = -1;
    }
    __syncthreads();
    if (mode != MODE_CURRENT) return;

    float local_exp_sum = 0.0f;
    const float max_logit = max_logits[0];
    for (int view_pos = tid; view_pos < HW; view_pos += blockDim.x) {
        const int canonical_pos = view_to_canonical_pos(view_pos, phase);
        if (boards[static_cast<int64_t>(slot) * HW + canonical_pos] == 0) {
            const float logit = (head_dot(features, pw, slot, view_pos) + pb) * inv_temp;
            local_exp_sum += __expf(logit - max_logit);
        }
    }
    exp_sums[tid] = local_exp_sum;
    __syncthreads();
    for (int stride = POLICY_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) exp_sums[tid] += exp_sums[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        const int64_t idx = static_cast<int64_t>(atomic_add_i64(counters + C_BUFFER_COUNT, 1ULL));
        transition_idx = idx;
        if (idx >= buffer_capacity) {
            counters[C_ERROR] = 1;
        } else {
            const float selected_logit =
                (head_dot(features, pw, slot, chosen_view) + pb) * inv_temp;
            const float logsumexp = max_logit + logf(fmaxf(exp_sums[0], 1e-30f));
            buffer_players[idx] = current_player[slot];
            buffer_stones_left[idx] = stones_left[slot];
            buffer_move_counts[idx] = move_count[slot];
            buffer_actions[idx] = static_cast<int16_t>(chosen_view);
            buffer_logprobs[idx] = selected_logit - logsumexp;
            buffer_values[idx] = tanhf(value_sums[0] / static_cast<float>(HW));
            buffer_episode_ids[idx] = current_episode_ids[slot];
            segment_positions[slot] += 1;
        }
    }
    __syncthreads();
    if (transition_idx >= 0 && transition_idx < buffer_capacity) {
        for (int view_pos = tid; view_pos < HW; view_pos += blockDim.x) {
            const int canonical_pos = view_to_canonical_pos(view_pos, phase);
            buffer_boards[transition_idx * HW + view_pos] =
                boards[static_cast<int64_t>(slot) * HW + canonical_pos];
        }
    }
}

__global__ void random_opening_kernel(
    const uint8_t* __restrict__ actor_mode,
    const int64_t* __restrict__ rng_counter,
    int16_t* __restrict__ actions_canonical,
    unsigned long long seed,
    int slots) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots || actor_mode[slot] != MODE_RANDOM_OPENING) return;
    const float u = random_uniform01(seed, slot, rng_counter[slot], 0x52414e44ULL);
    int action = static_cast<int>(u * static_cast<float>(HW));
    if (action >= HW) action = HW - 1;
    actions_canonical[slot] = static_cast<int16_t>(action);
}

__device__ __forceinline__ bool has_stone(
    const int8_t* board, int row, int col, int8_t actor) {
    if ((unsigned)row >= BOARD || (unsigned)col >= BOARD) return false;
    return board[row * BOARD + col] == actor;
}

__device__ __forceinline__ bool check_win_after_move(
    const int8_t* board, int action, int8_t actor) {
    const int row = action / BOARD;
    const int col = action - row * BOARD;
    const int dr[4] = {1, 0, 1, 1};
    const int dc[4] = {0, 1, 1, -1};
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        int count = 1;
        #pragma unroll
        for (int s = 1; s <= 5; ++s) {
            if (!has_stone(board, row + dr[d] * s, col + dc[d] * s, actor)) break;
            ++count;
        }
        #pragma unroll
        for (int s = 1; s <= 5; ++s) {
            if (!has_stone(board, row - dr[d] * s, col - dc[d] * s, actor)) break;
            ++count;
        }
        if (count >= 6) return true;
    }
    return false;
}

__global__ void game_step_kernel(
    int8_t* __restrict__ boards,
    int8_t* __restrict__ current_player,
    int8_t* __restrict__ stones_left,
    int16_t* __restrict__ empty_count,
    int16_t* __restrict__ move_count,
    int64_t* __restrict__ rng_counter,
    const uint8_t* __restrict__ table_kind,
    int8_t* __restrict__ current_color,
    const int16_t* __restrict__ actions_canonical,
    int32_t* __restrict__ current_episode_ids,
    int32_t* __restrict__ segment_positions,
    int8_t* __restrict__ episode_results,
    int16_t* __restrict__ episode_terminal_moves,
    int64_t episode_capacity,
    int64_t* __restrict__ counters,
    int slots) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slots) return;
    int action = static_cast<int>(actions_canonical[slot]);
    int8_t* board = boards + static_cast<int64_t>(slot) * HW;
    if (action < 0 || action >= HW || board[action] != 0) {
        counters[C_ERROR] = 3;
        action = -1;
        for (int pos = 0; pos < HW; ++pos) {
            if (board[pos] == 0) { action = pos; break; }
        }
        if (action < 0) return;
    }

    const int8_t actor = current_player[slot];
    board[action] = actor;
    empty_count[slot] -= 1;
    move_count[slot] += 1;
    rng_counter[slot] += 1;
    const bool won = check_win_after_move(board, action, actor);
    const bool draw = !won && empty_count[slot] == 0;
    if (won || draw) {
        const int8_t winner = won ? actor : static_cast<int8_t>(0);
        const int32_t episode = current_episode_ids[slot];
        if (episode >= 0 && episode < episode_capacity) {
            episode_results[episode] = winner;
            episode_terminal_moves[episode] = move_count[slot];
        } else counters[C_ERROR] = 4;

        atomic_add_i64(counters + C_COMPLETED_POSITIONS,
                       static_cast<unsigned long long>(segment_positions[slot]));
        atomic_add_i64(counters + C_GAMES, 1ULL);
        atomic_add_i64(counters + C_GAME_LENGTH_SUM,
                       static_cast<unsigned long long>(move_count[slot]));
        if (winner == 1) atomic_add_i64(counters + C_BLACK_WINS, 1ULL);
        else if (winner == -1) atomic_add_i64(counters + C_WHITE_WINS, 1ULL);
        else atomic_add_i64(counters + C_DRAWS, 1ULL);

        const uint8_t kind = table_kind[slot];
        if (kind == TABLE_HISTORY) {
            atomic_add_i64(counters + C_HISTORY_GAMES, 1ULL);
            if (winner == 0) atomic_add_i64(counters + C_HISTORY_DRAWS, 1ULL);
            else if (winner == current_color[slot]) atomic_add_i64(counters + C_HISTORY_WINS, 1ULL);
            else atomic_add_i64(counters + C_HISTORY_LOSSES, 1ULL);
        } else if (kind == TABLE_BOT) {
            atomic_add_i64(counters + C_BOT_GAMES, 1ULL);
            if (winner == 0) atomic_add_i64(counters + C_BOT_DRAWS, 1ULL);
            else if (winner == current_color[slot]) atomic_add_i64(counters + C_BOT_WINS, 1ULL);
            else atomic_add_i64(counters + C_BOT_LOSSES, 1ULL);
        }

        for (int pos = 0; pos < HW; ++pos) board[pos] = 0;
        current_player[slot] = 1;
        stones_left[slot] = 1;
        empty_count[slot] = HW;
        move_count[slot] = 0;
        segment_positions[slot] = 0;
        const int64_t next_episode = static_cast<int64_t>(
            atomic_add_i64(counters + C_NEXT_EPISODE_ID, 1ULL));
        if (next_episode >= episode_capacity) counters[C_ERROR] = 5;
        current_episode_ids[slot] = static_cast<int32_t>(next_episode);
        if (kind == TABLE_HISTORY || kind == TABLE_BOT)
            current_color[slot] = static_cast<int8_t>(-current_color[slot]);
        return;
    }

    const int left = static_cast<int>(stones_left[slot]) - 1;
    if (left > 0) stones_left[slot] = static_cast<int8_t>(left);
    else {
        current_player[slot] = static_cast<int8_t>(-actor);
        stones_left[slot] = 2;
    }
}

__global__ void init_rollout_kernel(
    int32_t* current_episode_ids,
    int32_t* segment_positions,
    int64_t* counters,
    int32_t* symmetry_phase,
    int slots,
    int symmetry_phase_start) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < slots) {
        current_episode_ids[idx] = idx;
        segment_positions[idx] = 0;
    }
    if (idx < C_COUNT) counters[idx] = 0;
    if (idx == 0) {
        counters[C_NEXT_EPISODE_ID] = slots;
        symmetry_phase[0] = symmetry_phase_start & 7;
    }
}

__global__ void advance_rollout_kernel(
    int64_t* counters, int32_t* symmetry_phase, int symmetry_enabled) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        counters[C_GRAPH_STEPS] += 1;
        if (symmetry_enabled) symmetry_phase[0] = (symmetry_phase[0] + 1) & 7;
    }
}

__global__ void set_condition_kernel(
    cudaGraphConditionalHandle handle,
    int64_t* counters,
    int64_t target,
    int64_t max_graph_steps) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const bool step_limit = counters[C_GRAPH_STEPS] >= max_graph_steps;
        if (step_limit && counters[C_COMPLETED_POSITIONS] < target && counters[C_ERROR] == 0)
            counters[C_ERROR] = 6;
        const bool keep_going = counters[C_COMPLETED_POSITIONS] < target
            && counters[C_ERROR] == 0 && !step_limit;
        cudaGraphSetConditional(handle, keep_going ? 1U : 0U);
    }
}

cudaGraphNode_t add_kernel_node(
    cudaGraph_t graph, cudaGraphNode_t dependency, void* func,
    dim3 grid, dim3 block, void** args) {
    cudaKernelNodeParams p{};
    p.func = func; p.gridDim = grid; p.blockDim = block;
    p.sharedMemBytes = 0; p.kernelParams = args; p.extra = nullptr;
    cudaGraphNode_t node{};
    if (dependency) CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, &dependency, 1, &p));
    else CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p));
    return node;
}

void check_cuda_contiguous(
    const torch::Tensor& t, torch::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " musi być CUDA");
    TORCH_CHECK(t.scalar_type() == dtype, name, " ma zły dtype");
    TORCH_CHECK(t.is_contiguous(), name, " musi być contiguous");
}

void check_half_cuda_contiguous(const torch::Tensor& t, const char* name) {
    check_cuda_contiguous(t, torch::kFloat16, name);
}

}  // namespace

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
    int64_t symmetry_phase_start) {

    TORCH_CHECK(conv_weights.size() == 8, "Native rollout wymaga 8 conv weights");
    TORCH_CHECK(norm_weights.size() == 8 && norm_biases.size() == 8,
                "Native rollout wymaga 8 GroupNorm affine pairs");
    TORCH_CHECK(target_completed_positions > 0, "target_completed_positions musi być > 0");
    TORCH_CHECK(temperature > 0.0, "temperature musi być > 0");
    TORCH_CHECK(random_black_opening_fraction >= 0.0 && random_black_opening_fraction <= 1.0,
                "random_black_opening_fraction musi być w [0,1]");

    int slots = static_cast<int>(boards.size(0));
    TORCH_CHECK(slots > 0 && slots <= 32768, "Niepoprawna liczba rollout slots");
    TORCH_CHECK(boards.dim() == 3 && boards.size(1) == BOARD && boards.size(2) == BOARD,
                "boards musi mieć [N,19,19]");
    check_cuda_contiguous(boards, torch::kInt8, "boards");
    check_cuda_contiguous(current_player, torch::kInt8, "current_player");
    check_cuda_contiguous(stones_left, torch::kInt8, "stones_left");
    check_cuda_contiguous(empty_count, torch::kInt16, "empty_count");
    check_cuda_contiguous(move_count, torch::kInt16, "move_count");
    check_cuda_contiguous(rng_counter, torch::kInt64, "rng_counter");
    check_cuda_contiguous(table_kind, torch::kUInt8, "table_kind");
    check_cuda_contiguous(opponent_model, torch::kInt32, "opponent_model");
    check_cuda_contiguous(current_color, torch::kInt8, "current_color");
    check_cuda_contiguous(bot_version, torch::kUInt8, "bot_version");

    int64_t buffer_capacity = buffer_boards.size(0);
    TORCH_CHECK(buffer_boards.dim() == 3 && buffer_boards.size(1) == BOARD && buffer_boards.size(2) == BOARD,
                "buffer_boards musi mieć [CAP,19,19]");
    check_cuda_contiguous(buffer_boards, torch::kInt8, "buffer_boards");
    check_cuda_contiguous(buffer_players, torch::kInt8, "buffer_players");
    check_cuda_contiguous(buffer_stones_left, torch::kInt8, "buffer_stones_left");
    check_cuda_contiguous(buffer_move_counts, torch::kInt16, "buffer_move_counts");
    check_cuda_contiguous(buffer_actions, torch::kInt16, "buffer_actions");
    check_cuda_contiguous(buffer_logprobs, torch::kFloat32, "buffer_logprobs");
    check_cuda_contiguous(buffer_values, torch::kFloat32, "buffer_values");
    check_cuda_contiguous(buffer_episode_ids, torch::kInt32, "buffer_episode_ids");
    check_cuda_contiguous(episode_results, torch::kInt8, "episode_results");
    check_cuda_contiguous(episode_terminal_moves, torch::kInt16, "episode_terminal_moves");
    TORCH_CHECK(episode_results.numel() == episode_terminal_moves.numel(),
                "episode result arrays mają różne rozmiary");

    const int models = static_cast<int>(policy_weight.size(0));
    TORCH_CHECK(models >= 1 && models <= 65535, "Niepoprawna liczba packed models");
    const int expected_o[8] = {32, 32, 64, 64, 64, 96, 96, 96};
    const int expected_k[8] = {2128, 288, 288, 576, 576, 576, 864, 864};
    for (int i = 0; i < 8; ++i) {
        check_half_cuda_contiguous(conv_weights[i], "conv_weights");
        check_half_cuda_contiguous(norm_weights[i], "norm_weights");
        check_half_cuda_contiguous(norm_biases[i], "norm_biases");
        TORCH_CHECK(conv_weights[i].dim() == 3 && conv_weights[i].size(0) == models &&
                    conv_weights[i].size(1) == expected_o[i] && conv_weights[i].size(2) == expected_k[i],
                    "Niepoprawny packed conv weight w warstwie ", i);
        TORCH_CHECK(norm_weights[i].dim() == 2 && norm_weights[i].size(0) == models &&
                    norm_weights[i].size(1) == expected_o[i], "Niepoprawny norm weight w warstwie ", i);
        TORCH_CHECK(norm_biases[i].dim() == 2 && norm_biases[i].size(0) == models &&
                    norm_biases[i].size(1) == expected_o[i], "Niepoprawny norm bias w warstwie ", i);
    }
    check_half_cuda_contiguous(policy_weight, "policy_weight");
    check_half_cuda_contiguous(policy_bias, "policy_bias");
    check_half_cuda_contiguous(value_weight, "value_weight");
    check_half_cuda_contiguous(value_bias, "value_bias");
    TORCH_CHECK(policy_weight.dim() == 2 && policy_weight.size(0) == models && policy_weight.size(1) == STORAGE_C,
                "policy_weight musi mieć [M,96]");
    TORCH_CHECK(value_weight.dim() == 2 && value_weight.size(0) == models && value_weight.size(1) == STORAGE_C,
                "value_weight musi mieć [M,96]");
    TORCH_CHECK(policy_bias.numel() == models && value_bias.numel() == models,
                "head biases muszą mieć [M]");

    const int device = boards.get_device();
    CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    const auto dev = boards.device();
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opts_i16 = torch::TensorOptions().dtype(torch::kInt16).device(dev);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);
    auto opts_f16 = torch::TensorOptions().dtype(torch::kFloat16).device(dev);

    torch::Tensor actor_mode = torch::empty({slots}, opts_u8);
    torch::Tensor cnn_active = torch::empty({slots}, opts_u8);
    torch::Tensor actor_model = torch::empty({slots}, opts_i32);
    torch::Tensor actions_view = torch::empty({slots}, opts_i16);
    torch::Tensor actions_canonical = torch::empty({slots}, opts_i16);
    torch::Tensor current_episode_ids = torch::empty({slots}, opts_i32);
    torch::Tensor segment_positions = torch::zeros({slots}, opts_i32);
    torch::Tensor counters = torch::zeros({C_COUNT}, opts_i64);
    torch::Tensor symmetry_phase = torch::empty({1}, opts_i32);
    torch::Tensor feat_a = torch::empty({slots, HW, STORAGE_C}, opts_f16);
    torch::Tensor feat_b = torch::empty({slots, HW, STORAGE_C}, opts_f16);

    episode_results.fill_(UNKNOWN_EPISODE_RESULT);
    episode_terminal_moves.fill_(UNKNOWN_TERMINAL_MOVE);

    uint8_t* mode_p = actor_mode.data_ptr<uint8_t>();
    uint8_t* active_p = cnn_active.data_ptr<uint8_t>();
    int32_t* model_p = actor_model.data_ptr<int32_t>();
    int16_t* action_view_p = actions_view.data_ptr<int16_t>();
    int16_t* action_p = actions_canonical.data_ptr<int16_t>();
    int32_t* episode_id_p = current_episode_ids.data_ptr<int32_t>();
    int32_t* segment_p = segment_positions.data_ptr<int32_t>();
    int64_t* counters_p = counters.data_ptr<int64_t>();
    int32_t* phase_p = symmetry_phase.data_ptr<int32_t>();
    half* a_p = reinterpret_cast<half*>(feat_a.data_ptr<at::Half>());
    half* b_p = reinterpret_cast<half*>(feat_b.data_ptr<at::Half>());
    int8_t* boards_p = boards.data_ptr<int8_t>();
    int8_t* player_p = current_player.data_ptr<int8_t>();
    int8_t* stones_p = stones_left.data_ptr<int8_t>();
    int16_t* empty_p = empty_count.data_ptr<int16_t>();
    int16_t* move_p = move_count.data_ptr<int16_t>();
    int64_t* rng_p = rng_counter.data_ptr<int64_t>();
    uint8_t* kind_p = table_kind.data_ptr<uint8_t>();
    int32_t* opp_model_p = opponent_model.data_ptr<int32_t>();
    int8_t* color_p = current_color.data_ptr<int8_t>();
    uint8_t* bot_p = bot_version.data_ptr<uint8_t>();
    int8_t* bb_p = buffer_boards.data_ptr<int8_t>();
    int8_t* bp_p = buffer_players.data_ptr<int8_t>();
    int8_t* bs_p = buffer_stones_left.data_ptr<int8_t>();
    int16_t* bm_p = buffer_move_counts.data_ptr<int16_t>();
    int16_t* ba_p = buffer_actions.data_ptr<int16_t>();
    float* bl_p = buffer_logprobs.data_ptr<float>();
    float* bv_p = buffer_values.data_ptr<float>();
    int32_t* be_p = buffer_episode_ids.data_ptr<int32_t>();
    int8_t* er_p = episode_results.data_ptr<int8_t>();
    int16_t* etm_p = episode_terminal_moves.data_ptr<int16_t>();

    std::vector<const half*> w(8), nw(8), nb(8);
    for (int i = 0; i < 8; ++i) {
        w[i] = reinterpret_cast<const half*>(conv_weights[i].data_ptr<at::Half>());
        nw[i] = reinterpret_cast<const half*>(norm_weights[i].data_ptr<at::Half>());
        nb[i] = reinterpret_cast<const half*>(norm_biases[i].data_ptr<at::Half>());
    }
    const half* pw_p = reinterpret_cast<const half*>(policy_weight.data_ptr<at::Half>());
    const half* pbias_p = reinterpret_cast<const half*>(policy_bias.data_ptr<at::Half>());
    const half* vw_p = reinterpret_cast<const half*>(value_weight.data_ptr<at::Half>());
    const half* vbias_p = reinterpret_cast<const half*>(value_bias.data_ptr<at::Half>());
    unsigned long long seed_u = static_cast<unsigned long long>(seed);
    float temp_f = static_cast<float>(temperature);
    float opening_f = static_cast<float>(random_black_opening_fraction);
    int symmetry_enabled = symmetry_augmentation ? 1 : 0;
    int phase_start = static_cast<int>(symmetry_phase_start) & 7;
    int64_t episode_capacity = episode_results.numel();
    int64_t target = target_completed_positions;
    int64_t max_graph_steps = 1000000;

    const int init_count = slots > C_COUNT ? slots : C_COUNT;
    init_rollout_kernel<<<(init_count + 255) / 256, 256, 0, stream>>>(
        episode_id_p, segment_p, counters_p, phase_p, slots, phase_start);
    CUDA_CHECK(cudaGetLastError());

    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle{};
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams cp{};
    cp.type = cudaGraphNodeTypeConditional;
    cp.conditional.handle = handle;
    cp.conditional.type = cudaGraphCondTypeWhile;
    cp.conditional.size = 1;
    cudaGraphNode_t conditional_node{};
    CUDA_CHECK(cudaGraphAddNode(&conditional_node, graph, nullptr, 0, &cp));
    cudaGraph_t body = cp.conditional.phGraph_out[0];
    cudaGraphNode_t dep{};

    { void* args[] = {&player_p,&stones_p,&move_p,&rng_p,&kind_p,&opp_model_p,&color_p,&bot_p,
                      &mode_p,&active_p,&model_p,&seed_u,&opening_f,&slots};
      dep = add_kernel_node(body,dep,(void*)prepare_actor_kernel,dim3((slots+255)/256),dim3(256),args); }
    { void* args[] = {&boards_p,&active_p,&player_p,&stones_p,&model_p,&phase_p,&symmetry_enabled,&w[0],&a_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_first_kernel<32,23,2128,32>,dim3(slots*MGROUPS*2),dim3(CONV_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&nw[0],&nb[0],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<32>,dim3(slots*4),dim3(NORM_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&w[1],&b_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_hidden_kernel<32,32,3,288,32>,dim3(slots*MGROUPS*2),dim3(CONV_THREADS),args); }
    { void* args[] = {&b_p,&active_p,&model_p,&nw[1],&nb[1],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<32>,dim3(slots*4),dim3(NORM_THREADS),args); }
    { void* args[] = {&b_p,&active_p,&model_p,&w[2],&a_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_hidden_kernel<32,64,3,288,64>,dim3(slots*MGROUPS*4),dim3(CONV_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&nw[2],&nb[2],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<64>,dim3(slots*8),dim3(NORM_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&w[3],&b_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_hidden_kernel<64,64,3,576,64>,dim3(slots*MGROUPS*4),dim3(CONV_THREADS),args); }
    { void* args[] = {&b_p,&active_p,&model_p,&nw[3],&nb[3],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<64>,dim3(slots*8),dim3(NORM_THREADS),args); }
    { void* args[] = {&b_p,&active_p,&model_p,&w[4],&a_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_hidden_kernel<64,64,3,576,64>,dim3(slots*MGROUPS*4),dim3(CONV_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&nw[4],&nb[4],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<64>,dim3(slots*8),dim3(NORM_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&w[5],&b_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_hidden_kernel<64,96,3,576,96>,dim3(slots*MGROUPS*6),dim3(CONV_THREADS),args); }
    { void* args[] = {&b_p,&active_p,&model_p,&nw[5],&nb[5],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<96>,dim3(slots*12),dim3(NORM_THREADS),args); }
    { void* args[] = {&b_p,&active_p,&model_p,&w[6],&a_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_hidden_kernel<96,96,3,864,96>,dim3(slots*MGROUPS*6),dim3(CONV_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&nw[6],&nb[6],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<96>,dim3(slots*12),dim3(NORM_THREADS),args); }
    { void* args[] = {&a_p,&active_p,&model_p,&w[7],&b_p,&slots};
      dep = add_kernel_node(body,dep,(void*)conv_hidden_kernel<96,96,3,864,96>,dim3(slots*MGROUPS*6),dim3(CONV_THREADS),args); }
    { void* args[] = {&b_p,&active_p,&model_p,&nw[7],&nb[7],&slots};
      dep = add_kernel_node(body,dep,(void*)groupnorm_silu_kernel<96>,dim3(slots*12),dim3(NORM_THREADS),args); }
    { void* args[] = {&b_p,&pw_p,&pbias_p,&vw_p,&vbias_p,&boards_p,&player_p,&stones_p,&move_p,&rng_p,
                      &mode_p,&model_p,&phase_p,&symmetry_enabled,&action_view_p,&action_p,&episode_id_p,&segment_p,
                      &bb_p,&bp_p,&bs_p,&bm_p,&ba_p,&bl_p,&bv_p,&be_p,&counters_p,&buffer_capacity,&temp_f,&seed_u,&slots};
      dep = add_kernel_node(body,dep,(void*)policy_sample_kernel,dim3(slots),dim3(POLICY_THREADS),args); }
    { void* args[] = {&boards_p,&player_p,&stones_p,&mode_p,&action_p,&slots};
      dep = add_kernel_node(body,dep,(void*)connect6_rollout_bot::tactical_bot_kernel<false, MODE_BOT_V1>,
                            dim3(slots),dim3(connect6_rollout_bot::THREADS),args); }
    { void* args[] = {&boards_p,&player_p,&stones_p,&mode_p,&action_p,&slots};
      dep = add_kernel_node(body,dep,(void*)connect6_rollout_bot::tactical_bot_kernel<true, MODE_BOT_V2>,
                            dim3(slots),dim3(connect6_rollout_bot::THREADS),args); }
    { void* args[] = {&mode_p,&rng_p,&action_p,&seed_u,&slots};
      dep = add_kernel_node(body,dep,(void*)random_opening_kernel,dim3((slots+255)/256),dim3(256),args); }
    { void* args[] = {&boards_p,&player_p,&stones_p,&empty_p,&move_p,&rng_p,&kind_p,&color_p,&action_p,&episode_id_p,
                      &segment_p,&er_p,&etm_p,&episode_capacity,&counters_p,&slots};
      dep = add_kernel_node(body,dep,(void*)game_step_kernel,dim3((slots+255)/256),dim3(256),args); }
    { void* args[] = {&counters_p,&phase_p,&symmetry_enabled};
      dep = add_kernel_node(body,dep,(void*)advance_rollout_kernel,dim3(1),dim3(1),args); }
    { void* args[] = {&handle,&counters_p,&target,&max_graph_steps};
      dep = add_kernel_node(body,dep,(void*)set_condition_kernel,dim3(1),dim3(1),args); }

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return {counters, symmetry_phase};
}
