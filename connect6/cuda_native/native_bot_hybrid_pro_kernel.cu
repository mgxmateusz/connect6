#include "native_bot_hybrid_liveroad_pair_variants32_kernel.cu"
#include "native_bot_v2_pro_score.cuh"

namespace {

using namespace v4_detail;
using v2pro_detail::score_all_pro;

__device__ __forceinline__ int64_t hybrid_pair_score_pro_prior(
    const int8_t* board,
    int first,
    int second,
    int8_t player,
    const int16_t* base_opp_ta,
    const int16_t* base_opp_tb,
    int base_opp_threat_count,
    int base_opp_overflow,
    const int* pro_scores) {
    return cheap_pair_score(
        board, first, second, player,
        base_opp_ta, base_opp_tb, base_opp_threat_count, base_opp_overflow) +
        static_cast<int64_t>(pro_scores[first]) +
        static_cast<int64_t>(pro_scores[second]);
}

template <int PRO_HYBRID_K>
__global__ void tactical_search_hybrid_pro_kernel(
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    int16_t* __restrict__ pending_second,
    int64_t* __restrict__ actions,
    int batch) {
    const int board_id = blockIdx.x;
    if (board_id >= batch) return;
    const int tid = threadIdx.x;
    const int8_t* src = boards + static_cast<int64_t>(board_id) * CELLS;
    const int8_t player = current_player[board_id];
    const int8_t left = stones_left[board_id];

    __shared__ int8_t board[CELLS];
    __shared__ int scores[CELLS];
    __shared__ int active[CELLS];
    __shared__ int candidate_actions[CELLS];
    __shared__ int candidate_count;
    __shared__ int64_t row_scores[CELLS];
    __shared__ int selected_first[PRO_HYBRID_K];
    __shared__ int selected_second[PRO_HYBRID_K];
    __shared__ int64_t selected_scores[PRO_HYBRID_K];
    __shared__ int reduce[THREADS];
    __shared__ int16_t my_ta[MAX_THREATS];
    __shared__ int16_t my_tb[MAX_THREATS];
    __shared__ int16_t opp_ta[MAX_THREATS];
    __shared__ int16_t opp_tb[MAX_THREATS];
    __shared__ int my_threat_count;
    __shared__ int opp_threat_count;
    __shared__ int my_overflow;
    __shared__ int opp_overflow;
    __shared__ int my_win;
    __shared__ int opp_win;
    __shared__ int eval_result;
    __shared__ int base_opp_threat_count;
    __shared__ int base_opp_overflow;

    if (left <= 1) {
        const int cached = static_cast<int>(pending_second[board_id]);
        if (cached >= 0 && cached < CELLS && src[cached] == 0) {
            if (tid == 0) {
                actions[board_id] = static_cast<int64_t>(cached);
                pending_second[board_id] = static_cast<int16_t>(-1);
            }
            return;
        }
        if (tid == 0) pending_second[board_id] = static_cast<int16_t>(-1);
    } else if (tid == 0) {
        pending_second[board_id] = static_cast<int16_t>(-1);
    }

    for (int i = tid; i < CELLS; i += blockDim.x) {
        board[i] = src[i];
        active[i] = 0;
    }
    __syncthreads();

    if (left <= 1) {
        score_all_pro(board, player, left, scores);
        __syncthreads();
        if (tid == 0) {
            int a[1]; int s[1];
            select_top_k_serial<1>(scores, a, s);
            actions[board_id] = static_cast<int64_t>(a[0]);
        }
        return;
    }

    score_all_pro(board, player, 2, scores);
    __syncthreads();
    if (tid == 0) {
        int a[1]; int s[1];
        select_top_k_serial<1>(scores, a, s);
        if (a[0] >= 0 && s[0] >= WIN_SCORE) {
            actions[board_id] = static_cast<int64_t>(a[0]);
            pending_second[board_id] = static_cast<int16_t>(-1);
            candidate_count = -1;
        } else {
            candidate_count = 0;
            base_opp_threat_count = 0;
            base_opp_overflow = 0;
        }
    }
    __syncthreads();
    if (candidate_count < 0) return;

    // EXACTLY the baseline Hybrid structural pool: clean own/opp roads with >=2
    // stones. V2Pro does not add cells to the pool.
    for (int road = tid; road < ROADS; road += blockDim.x) {
        int start, dr, dc;
        decode_road(road, start, dr, dc);
        int mine = 0, opp = 0;
        int empties[WIN];
        int empty_n = 0;
        const int r = start / BOARD;
        const int c = start - r * BOARD;
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            const int idx = (r + k * dr) * BOARD + (c + k * dc);
            const int8_t v = board[idx];
            if (v == player) ++mine;
            else if (v == -player) ++opp;
            else empties[empty_n++] = idx;
        }

        const bool live_own = opp == 0 && mine >= 2 && mine < WIN;
        const bool live_opp = mine == 0 && opp >= 2 && opp < WIN;
        if (live_own || live_opp) {
            for (int e = 0; e < empty_n; ++e) atomicExch(&active[empties[e]], 1);
        }

        if (mine == 0 && (opp == 4 || opp == 5) && empty_n >= 1 && empty_n <= 2) {
            const int slot = atomicAdd(&base_opp_threat_count, 1);
            if (slot < MAX_THREATS) {
                opp_ta[slot] = static_cast<int16_t>(empties[0]);
                opp_tb[slot] = static_cast<int16_t>(empty_n == 2 ? empties[1] : -1);
            } else {
                atomicExch(&base_opp_overflow, 1);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        candidate_count = 0;
        for (int cell = 0; cell < CELLS; ++cell) {
            if (active[cell] != 0 && board[cell] == 0)
                candidate_actions[candidate_count++] = cell;
        }

        // Same legality-only fallback as baseline Hybrid; still no min16 floor.
        if (candidate_count < 2) {
            int a[2]; int s[2];
            select_top_k_serial<2>(scores, a, s);
            actions[board_id] = static_cast<int64_t>(a[0]);
            pending_second[board_id] = static_cast<int16_t>(a[1]);
            candidate_count = -1;
        }
    }
    __syncthreads();
    if (candidate_count < 0) return;

    if (tid == 0) {
        for (int q = 0; q < PRO_HYBRID_K; ++q) {
            selected_scores[q] = static_cast<int64_t>(LLONG_MIN);
            selected_first[q] = -1;
            selected_second[q] = -1;
        }
    }
    __syncthreads();

    for (int i = 0; i + 1 < candidate_count; ++i) {
        const int first = candidate_actions[i];
        for (int j = i + 1 + tid; j < candidate_count; j += blockDim.x) {
            const int second = candidate_actions[j];
            row_scores[j] = hybrid_pair_score_pro_prior(
                board, first, second, player,
                opp_ta, opp_tb, base_opp_threat_count, base_opp_overflow, scores);
        }
        __syncthreads();

        if (tid == 0) {
            for (int j = i + 1; j < candidate_count; ++j) {
                const int second = candidate_actions[j];
                const int64_t pair_score = row_scores[j];
                if (selected_first[PRO_HYBRID_K - 1] >= 0 &&
                    !cheap_pair_better(pair_score, first, second,
                        selected_scores[PRO_HYBRID_K - 1],
                        selected_first[PRO_HYBRID_K - 1],
                        selected_second[PRO_HYBRID_K - 1])) {
                    continue;
                }
                for (int pos = 0; pos < PRO_HYBRID_K; ++pos) {
                    if (cheap_pair_better(pair_score, first, second,
                            selected_scores[pos], selected_first[pos], selected_second[pos])) {
                        for (int s = PRO_HYBRID_K - 1; s > pos; --s) {
                            selected_scores[s] = selected_scores[s - 1];
                            selected_first[s] = selected_first[s - 1];
                            selected_second[s] = selected_second[s - 1];
                        }
                        selected_scores[pos] = pair_score;
                        selected_first[pos] = first;
                        selected_second[pos] = second;
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }

    int best_value = INVALID_SCORE;
    int64_t best_order = static_cast<int64_t>(LLONG_MIN);
    int best_first = -1;
    int best_second = -1;

    #pragma unroll 1
    for (int q = 0; q < PRO_HYBRID_K; ++q) {
        const int first = selected_first[q];
        const int second = selected_second[q];
        if (first < 0 || second < 0) continue;
        if (tid == 0) {
            board[first] = player;
            board[second] = player;
        }
        __syncthreads();
        const int value = evaluate_state_parallel(
            board, player, static_cast<int8_t>(-player), reduce,
            my_ta, my_tb, opp_ta, opp_tb,
            &my_threat_count, &opp_threat_count,
            &my_overflow, &opp_overflow, &my_win, &opp_win, &eval_result);
        if (tid == 0) {
            if (best_first < 0 ||
                pair_better(value, selected_scores[q], first, second,
                            best_value, best_order, best_first, best_second)) {
                best_value = value;
                best_order = selected_scores[q];
                best_first = first;
                best_second = second;
            }
            board[first] = 0;
            board[second] = 0;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (best_first < 0) {
            best_first = candidate_actions[0];
            best_second = candidate_actions[1];
        }
        actions[board_id] = static_cast<int64_t>(best_first);
        pending_second[board_id] = static_cast<int16_t>(best_second);
    }
}

}  // namespace

extern "C" cudaError_t launch_tactical_bot_hybrid_pro_pair128_cuda(
    const int8_t* boards, const int8_t* current_player, const int8_t* stones_left,
    int16_t* pending_second, int64_t* actions, int batch, cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_hybrid_pro_kernel<128><<<batch, v4_detail::THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_tactical_bot_hybrid_pro_pair32_cuda(
    const int8_t* boards, const int8_t* current_player, const int8_t* stones_left,
    int16_t* pending_second, int64_t* actions, int batch, cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_hybrid_pro_kernel<32><<<batch, v4_detail::THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}
