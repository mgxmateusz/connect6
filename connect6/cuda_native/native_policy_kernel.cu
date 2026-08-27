#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int BOARD = 19;
constexpr int HW = BOARD * BOARD;
constexpr int STORAGE_C = 96;
constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int CONV_THREADS = WARP * WARPS_PER_BLOCK;
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
constexpr int MTILES = (HW + WM - 1) / WM;
constexpr int MGROUPS = (MTILES + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
constexpr int GN_GROUP_C = 8;
constexpr int GN_THREADS = 256;

#define CUDA_CHECK(EXPR) do { \
    cudaError_t _e = (EXPR); \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e)); \
} while (0)

__device__ __forceinline__ half canonical_input_value_dense(
    const int8_t* boards,
    int slot,
    int pos,
    int channel,
    const int8_t* current_player,
    const int8_t* stones_left) {
    if (channel == 2) return __float2half(1.0f);
    if (channel == 3) return __float2half(stones_left[slot] == 1 ? 1.0f : 0.0f);
    const int8_t v = boards[static_cast<int64_t>(slot) * HW + pos];
    const int8_t p = current_player[slot];
    if (channel == 0) return __float2half(v == p ? 1.0f : 0.0f);
    return __float2half(v == -p ? 1.0f : 0.0f);
}

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + __expf(-x));
}

template<int COUT, int KH, int KPAD, int OPAD, bool ACTIVATE>
__global__ void conv_first_dense_kernel(
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    const int32_t* __restrict__ model_ids,
    const half* __restrict__ weights,
    half* __restrict__ output,
    int slots) {

    constexpr int NT = (COUT + WN - 1) / WN;
    int q = static_cast<int>(blockIdx.x);
    const int tile_n = q % NT; q /= NT;
    const int group_m = q % MGROUPS; q /= MGROUPS;
    const int slot = q;
    if (slot >= slots) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int tile_m = group_m * WARPS_PER_BLOCK + warp;
    const bool valid_warp = tile_m < MTILES;
    const int model = model_ids[slot];
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
            b_smem[t] = weights[
                (static_cast<int64_t>(model) * OPAD + n0 + col) * KPAD + k0 + row];
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
                        v = canonical_input_value_dense(
                            boards, slot, irow * BOARD + icol, channel,
                            current_player, stones_left);
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
                const float y = c_smem[warp][t];
                output[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + oc]
                    = __float2half_rn(ACTIVATE ? silu_fast(y) : y);
            }
        }
    }
}

template<int CIN, int COUT, int KH, int KPAD, int OPAD, bool ACTIVATE>
__global__ void conv_hidden_dense_kernel(
    const half* __restrict__ input,
    const int32_t* __restrict__ model_ids,
    const half* __restrict__ weights,
    half* __restrict__ output,
    int slots) {

    constexpr int NT = (COUT + WN - 1) / WN;
    int q = static_cast<int>(blockIdx.x);
    const int tile_n = q % NT; q /= NT;
    const int group_m = q % MGROUPS; q /= MGROUPS;
    const int slot = q;
    if (slot >= slots) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int tile_m = group_m * WARPS_PER_BLOCK + warp;
    const bool valid_warp = tile_m < MTILES;
    const int model = model_ids[slot];
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
            b_smem[t] = weights[
                (static_cast<int64_t>(model) * OPAD + n0 + col) * KPAD + k0 + row];
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
                        v = input[
                            (static_cast<int64_t>(slot) * HW + ipos) * STORAGE_C + channel];
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
                const float y = c_smem[warp][t];
                output[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + oc]
                    = __float2half_rn(ACTIVATE ? silu_fast(y) : y);
            }
        }
    }
}

template<int C, int OPAD>
__global__ void group_norm_silu_dense_kernel(
    half* __restrict__ features,
    const int32_t* __restrict__ model_ids,
    const half* __restrict__ norm_weight,
    const half* __restrict__ norm_bias,
    int slots) {

    constexpr int GROUPS = C / GN_GROUP_C;
    constexpr int N = HW * GN_GROUP_C;
    const int q = static_cast<int>(blockIdx.x);
    const int group = q % GROUPS;
    const int slot = q / GROUPS;
    if (slot >= slots) return;

    const int tid = threadIdx.x;
    const int model = model_ids[slot];
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int e = tid; e < N; e += blockDim.x) {
        const int pos = e / GN_GROUP_C;
        const int c = group * GN_GROUP_C + (e % GN_GROUP_C);
        const float v = __half2float(
            features[(static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + c]);
        sum += v;
        sumsq = fmaf(v, v, sumsq);
    }

    __shared__ float red_sum[GN_THREADS];
    __shared__ float red_sumsq[GN_THREADS];
    red_sum[tid] = sum;
    red_sumsq[tid] = sumsq;
    __syncthreads();
    for (int stride = GN_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_sum[tid] += red_sum[tid + stride];
            red_sumsq[tid] += red_sumsq[tid + stride];
        }
        __syncthreads();
    }

    const float mean = red_sum[0] / static_cast<float>(N);
    const float variance = fmaxf(
        red_sumsq[0] / static_cast<float>(N) - mean * mean, 0.0f);
    const float inv_std = rsqrtf(variance + 1.0e-5f);

    for (int e = tid; e < N; e += blockDim.x) {
        const int pos = e / GN_GROUP_C;
        const int c = group * GN_GROUP_C + (e % GN_GROUP_C);
        const int64_t offset = (static_cast<int64_t>(slot) * HW + pos) * STORAGE_C + c;
        const float v = __half2float(features[offset]);
        const float gamma = __half2float(norm_weight[static_cast<int64_t>(model) * OPAD + c]);
        const float beta = __half2float(norm_bias[static_cast<int64_t>(model) * OPAD + c]);
        const float y = (v - mean) * inv_std * gamma + beta;
        features[offset] = __float2half_rn(silu_fast(y));
    }
}

__device__ __forceinline__ void choose_better(
    float candidate_v, int candidate_i, float& best_v, int& best_i) {
    if (candidate_v > best_v || (candidate_v == best_v && candidate_i < best_i)) {
        best_v = candidate_v;
        best_i = candidate_i;
    }
}

__global__ void policy_argmax_dense_kernel(
    const half* __restrict__ features,
    const half* __restrict__ policy_weight,
    const int8_t* __restrict__ boards,
    const int32_t* __restrict__ model_ids,
    int64_t* __restrict__ actions,
    int slots) {

    const int slot = static_cast<int>(blockIdx.x);
    if (slot >= slots) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int model = model_ids[slot];

    const half2* w2 = reinterpret_cast<const half2*>(
        policy_weight + static_cast<int64_t>(model) * STORAGE_C);
    float best_v = -FLT_MAX;
    int best_i = HW;

    for (int pos = tid; pos < HW; pos += blockDim.x) {
        if (boards[static_cast<int64_t>(slot) * HW + pos] != 0) continue;
        const half2* f2 = reinterpret_cast<const half2*>(
            features + (static_cast<int64_t>(slot) * HW + pos) * STORAGE_C);
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < STORAGE_C / 2; ++k) {
            const float2 a = __half22float2(f2[k]);
            const float2 b = __half22float2(w2[k]);
            acc = fmaf(a.x, b.x, acc);
            acc = fmaf(a.y, b.y, acc);
        }
        choose_better(acc, pos, best_v, best_i);
    }

    for (int d = 16; d > 0; d >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best_v, d);
        const int oi = __shfl_down_sync(0xffffffffu, best_i, d);
        choose_better(ov, oi, best_v, best_i);
    }

    __shared__ float warp_v[WARPS_PER_BLOCK];
    __shared__ int warp_i[WARPS_PER_BLOCK];
    if (lane == 0) {
        warp_v[warp] = best_v;
        warp_i[warp] = best_i;
    }
    __syncthreads();

    if (warp == 0) {
        float v = lane < WARPS_PER_BLOCK ? warp_v[lane] : -FLT_MAX;
        int i = lane < WARPS_PER_BLOCK ? warp_i[lane] : HW;
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, d);
            const int oi = __shfl_down_sync(0xffffffffu, i, d);
            choose_better(ov, oi, v, i);
        }
        if (lane == 0) actions[slot] = static_cast<int64_t>(i < HW ? i : 0);
    }
}

void check_half_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " musi być CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " musi być FP16");
    TORCH_CHECK(t.is_contiguous(), name, " musi być contiguous");
}

}  // namespace


torch::Tensor policy_actions_dense_cuda(
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> norm_weights,
    std::vector<torch::Tensor> norm_biases,
    torch::Tensor policy_weight,
    torch::Tensor boards,
    torch::Tensor current_player,
    torch::Tensor stones_left,
    torch::Tensor model_ids) {

    TORCH_CHECK(weights.size() == 8, "Native policy wymaga 8 warstw conv");
    TORCH_CHECK(norm_weights.size() == 8 && norm_biases.size() == 8,
                "Native policy wymaga 8 slotów parametrów norm");
    TORCH_CHECK(boards.is_cuda() && boards.scalar_type() == torch::kInt8 && boards.is_contiguous(),
                "boards musi być CUDA int8 contiguous");
    TORCH_CHECK(boards.dim() == 3 && boards.size(1) == BOARD && boards.size(2) == BOARD,
                "boards musi mieć [B,19,19]");
    TORCH_CHECK(current_player.is_cuda() && current_player.scalar_type() == torch::kInt8 && current_player.is_contiguous(),
                "current_player musi być CUDA int8 contiguous");
    TORCH_CHECK(stones_left.is_cuda() && stones_left.scalar_type() == torch::kInt8 && stones_left.is_contiguous(),
                "stones_left musi być CUDA int8 contiguous");
    TORCH_CHECK(model_ids.is_cuda() && model_ids.scalar_type() == torch::kInt32 && model_ids.is_contiguous(),
                "model_ids musi być CUDA int32 contiguous");

    const int slots = static_cast<int>(boards.size(0));
    TORCH_CHECK(slots > 0 && slots <= 32768, "batch musi być w zakresie 1..32768");
    TORCH_CHECK(current_player.dim() == 1 && current_player.size(0) == slots,
                "current_player musi mieć [B]");
    TORCH_CHECK(stones_left.dim() == 1 && stones_left.size(0) == slots,
                "stones_left musi mieć [B]");
    TORCH_CHECK(model_ids.dim() == 1 && model_ids.size(0) == slots,
                "model_ids musi mieć [B]");

    const int64_t models = policy_weight.size(0);
    TORCH_CHECK(models >= 1 && models <= 65535, "Niepoprawna liczba modeli");
    const int expected_o[8] = {32, 32, 64, 64, 64, 96, 96, 96};
    const int expected_k[8] = {2128, 288, 288, 576, 576, 576, 864, 864};
    for (int i = 0; i < 8; ++i) {
        check_half_cuda_contiguous(weights[i], "weights");
        check_half_cuda_contiguous(norm_weights[i], "norm_weights");
        check_half_cuda_contiguous(norm_biases[i], "norm_biases");
        TORCH_CHECK(weights[i].dim() == 3 && weights[i].size(0) == models &&
                    weights[i].size(1) == expected_o[i] && weights[i].size(2) == expected_k[i],
                    "Niepoprawny packed weight w warstwie ", i);
        TORCH_CHECK(norm_weights[i].dim() == 2 && norm_weights[i].size(0) == models &&
                    norm_weights[i].size(1) == expected_o[i], "Niepoprawny norm weight w warstwie ", i);
        TORCH_CHECK(norm_biases[i].dim() == 2 && norm_biases[i].size(0) == models &&
                    norm_biases[i].size(1) == expected_o[i], "Niepoprawny norm bias w warstwie ", i);
    }
    check_half_cuda_contiguous(policy_weight, "policy_weight");
    TORCH_CHECK(policy_weight.dim() == 2 && policy_weight.size(1) == STORAGE_C,
                "policy_weight musi mieć [M,96]");

    const int device = boards.get_device();
    TORCH_CHECK(current_player.get_device() == device && stones_left.get_device() == device &&
                model_ids.get_device() == device && policy_weight.get_device() == device,
                "Wszystkie tensory muszą być na tym samym GPU");
    CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();

    auto feat_opts = torch::TensorOptions().dtype(torch::kFloat16).device(boards.device());
    auto action_opts = torch::TensorOptions().dtype(torch::kInt64).device(boards.device());
    torch::Tensor feat_a = torch::empty({slots, HW, STORAGE_C}, feat_opts);
    torch::Tensor feat_b = torch::empty({slots, HW, STORAGE_C}, feat_opts);
    torch::Tensor actions = torch::empty({slots}, action_opts);

    const int8_t* board_p = boards.data_ptr<int8_t>();
    const int8_t* player_p = current_player.data_ptr<int8_t>();
    const int8_t* stones_p = stones_left.data_ptr<int8_t>();
    const int32_t* ids_p = model_ids.data_ptr<int32_t>();
    half* a_p = reinterpret_cast<half*>(feat_a.data_ptr<at::Half>());
    half* b_p = reinterpret_cast<half*>(feat_b.data_ptr<at::Half>());
    int64_t* actions_p = actions.data_ptr<int64_t>();

    std::vector<const half*> w(8), nw(8), nb(8);
    for (int i = 0; i < 8; ++i) {
        w[i] = reinterpret_cast<const half*>(weights[i].data_ptr<at::Half>());
        nw[i] = reinterpret_cast<const half*>(norm_weights[i].data_ptr<at::Half>());
        nb[i] = reinterpret_cast<const half*>(norm_biases[i].data_ptr<at::Half>());
    }
    const half* policy_p = reinterpret_cast<const half*>(policy_weight.data_ptr<at::Half>());

    conv_first_dense_kernel<32,23,2128,32,false><<<slots * MGROUPS * 2, CONV_THREADS, 0, stream>>>(
        board_p, player_p, stones_p, ids_p, w[0], a_p, slots);
    group_norm_silu_dense_kernel<32,32><<<slots * (32 / GN_GROUP_C), GN_THREADS, 0, stream>>>(
        a_p, ids_p, nw[0], nb[0], slots);
    conv_hidden_dense_kernel<32,32,3,288,32,true><<<slots * MGROUPS * 2, CONV_THREADS, 0, stream>>>(
        a_p, ids_p, w[1], b_p, slots);
    conv_hidden_dense_kernel<32,64,3,288,64,false><<<slots * MGROUPS * 4, CONV_THREADS, 0, stream>>>(
        b_p, ids_p, w[2], a_p, slots);
    group_norm_silu_dense_kernel<64,64><<<slots * (64 / GN_GROUP_C), GN_THREADS, 0, stream>>>(
        a_p, ids_p, nw[2], nb[2], slots);
    conv_hidden_dense_kernel<64,64,3,576,64,true><<<slots * MGROUPS * 4, CONV_THREADS, 0, stream>>>(
        a_p, ids_p, w[3], b_p, slots);
    conv_hidden_dense_kernel<64,64,3,576,64,true><<<slots * MGROUPS * 4, CONV_THREADS, 0, stream>>>(
        b_p, ids_p, w[4], a_p, slots);
    conv_hidden_dense_kernel<64,96,3,576,96,false><<<slots * MGROUPS * 6, CONV_THREADS, 0, stream>>>(
        a_p, ids_p, w[5], b_p, slots);
    group_norm_silu_dense_kernel<96,96><<<slots * (96 / GN_GROUP_C), GN_THREADS, 0, stream>>>(
        b_p, ids_p, nw[5], nb[5], slots);
    conv_hidden_dense_kernel<96,96,3,864,96,true><<<slots * MGROUPS * 6, CONV_THREADS, 0, stream>>>(
        b_p, ids_p, w[6], a_p, slots);
    conv_hidden_dense_kernel<96,96,3,864,96,false><<<slots * MGROUPS * 6, CONV_THREADS, 0, stream>>>(
        a_p, ids_p, w[7], b_p, slots);
    group_norm_silu_dense_kernel<96,96><<<slots * (96 / GN_GROUP_C), GN_THREADS, 0, stream>>>(
        b_p, ids_p, nw[7], nb[7], slots);
    policy_argmax_dense_kernel<<<slots, CONV_THREADS, 0, stream>>>(
        b_p, policy_p, board_p, ids_p, actions_p, slots);
    CUDA_CHECK(cudaGetLastError());

    return actions;
}
