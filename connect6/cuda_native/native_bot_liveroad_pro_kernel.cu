#include "native_bot_liveroad_kernel.cu"
#include "native_bot_v2_pro_score.cuh"

namespace {

using namespace v4_detail;
using v2pro_detail::score_all_pro;

constexpr int PRO_LIVE_MIN_POOL = 16;

// Same LiveRoad structural pool and exhaustive restricted pair search. V2Pro
// replaces V2 only for one-stone play, immediate win, the sparse-opening TOP16
// floor and exact-value tie ordering.
__global__ void tactical_search_liveroad_pro_kernel(
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
    __shared__ int seed_actions[PRO_LIVE_MIN_POOL];
    __shared__ int seed_scores[PRO_LIVE_MIN_POOL];
    __shared__ int candidate_count;
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
        select_top_k_serial<PRO_LIVE_MIN_POOL>(scores, seed_actions, seed_scores);
        if (seed_actions[0] >= 0 && seed_scores[0] >= WIN_SCORE) {
            actions[board_id] = static_cast<int64_t>(seed_actions[0]);
            pending_second[board_id] = static_cast<int16_t>(-1);
            candidate_count = -1;
        } else {
            candidate_count = 0;
        }
    }
    __syncthreads();
    if (candidate_count < 0) return;

    for (int road = tid; road < ROADS; road += blockDim.x) {
        int start, dr, dc;
        decode_road(road, start, dr, dc);
        int mine = 0;
        int opp = 0;
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
    }
    __syncthreads();

    if (tid == 0) {
        int count = 0;
        for (int cell = 0; cell < CELLS; ++cell) count += active[cell] != 0;
        if (count < PRO_LIVE_MIN_POOL) {
            for (int k = 0; k < PRO_LIVE_MIN_POOL && count < PRO_LIVE_MIN_POOL; ++k) {
                const int cell = seed_actions[k];
                if (cell >= 0 && board[cell] == 0 && active[cell] == 0) {
                    active[cell] = 1;
                    ++count;
                }
            }
        }
        candidate_count = 0;
        for (int cell = 0; cell < CELLS; ++cell) {
            if (active[cell] != 0 && board[cell] == 0)
                candidate_actions[candidate_count++] = cell;
        }
    }
    __syncthreads();

    int best_value = INVALID_SCORE;
    int64_t best_order = INT64_MIN;
    int best_first = -1;
    int best_second = -1;

    #pragma unroll 1
    for (int i = 0; i < candidate_count; ++i) {
        const int first = candidate_actions[i];
        #pragma unroll 1
        for (int j = i + 1; j < candidate_count; ++j) {
            const int second = candidate_actions[j];
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
                const int64_t order = static_cast<int64_t>(scores[first]) +
                                      static_cast<int64_t>(scores[second]);
                if (best_first < 0 ||
                    pair_better(value, order, first, second,
                                best_value, best_order, best_first, best_second)) {
                    best_value = value;
                    best_order = order;
                    best_first = first;
                    best_second = second;
                }
                board[first] = 0;
                board[second] = 0;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        if (best_first < 0) {
            best_first = candidate_count > 0 ? candidate_actions[0] : seed_actions[0];
            best_second = candidate_count > 1 ? candidate_actions[1] : -1;
        }
        actions[board_id] = static_cast<int64_t>(best_first);
        pending_second[board_id] = static_cast<int16_t>(best_second);
    }
}

}  // namespace

extern "C" cudaError_t launch_tactical_bot_liveroad_pro_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_liveroad_pro_kernel<<<batch, v4_detail::THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}
