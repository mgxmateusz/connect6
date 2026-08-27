#include "native_bot_pair_eval_v4.cuh"

namespace {

using namespace v4_detail;

constexpr int CANDIDATE_K = 12;

__global__ void tactical_search_small_top12_kernel(
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
    __shared__ int candidate_actions[CANDIDATE_K];
    __shared__ int candidate_scores[CANDIDATE_K];

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

    __shared__ int selected_first;
    __shared__ int selected_second;

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

    for (int i = tid; i < CELLS; i += blockDim.x) board[i] = src[i];
    __syncthreads();

    if (left <= 1) {
        score_all(board, player, left, scores);
        __syncthreads();
        if (tid == 0) {
            int a[1];
            int s[1];
            select_top_k_serial<1>(scores, a, s);
            actions[board_id] = static_cast<int64_t>(a[0]);
        }
        return;
    }

    score_all(board, player, 2, scores);
    __syncthreads();
    if (tid == 0) {
        select_top_k_serial<CANDIDATE_K>(
            scores, candidate_actions, candidate_scores);
        if (candidate_actions[0] >= 0 && candidate_scores[0] >= WIN_SCORE) {
            selected_first = candidate_actions[0];
            selected_second = -1;
        } else {
            selected_first = -1;
            selected_second = -1;
        }
    }
    __syncthreads();

    if (selected_first >= 0) {
        if (tid == 0) {
            actions[board_id] = static_cast<int64_t>(selected_first);
            pending_second[board_id] = static_cast<int16_t>(-1);
        }
        return;
    }

    int best_value = INVALID_SCORE;
    int64_t best_order = static_cast<int64_t>(LLONG_MIN);
    int best_first = -1;
    int best_second = -1;

    #pragma unroll 1
    for (int i = 0; i < CANDIDATE_K; ++i) {
        const int first = candidate_actions[i];
        if (first < 0) continue;

        #pragma unroll 1
        for (int j = i + 1; j < CANDIDATE_K; ++j) {
            const int second = candidate_actions[j];
            if (second < 0) continue;

            if (tid == 0) {
                board[first] = player;
                board[second] = player;
            }
            __syncthreads();

            const int value = evaluate_state_parallel(
                board,
                player,
                static_cast<int8_t>(-player),
                reduce,
                my_ta,
                my_tb,
                opp_ta,
                opp_tb,
                &my_threat_count,
                &opp_threat_count,
                &my_overflow,
                &opp_overflow,
                &my_win,
                &opp_win,
                &eval_result);

            if (tid == 0) {
                const int64_t order =
                    static_cast<int64_t>(candidate_scores[i]) +
                    static_cast<int64_t>(candidate_scores[j]);
                if (best_first < 0 ||
                    pair_better(
                        value,
                        order,
                        first,
                        second,
                        best_value,
                        best_order,
                        best_first,
                        best_second)) {
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
            selected_first = candidate_actions[0];
            selected_second = -1;
        } else {
            selected_first = best_first;
            selected_second = best_second;
        }
        actions[board_id] = static_cast<int64_t>(selected_first);
        pending_second[board_id] = static_cast<int16_t>(selected_second);
    }
}

}  // namespace

extern "C" cudaError_t launch_tactical_bot_small_top12_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_small_top12_kernel<<<batch, v4_detail::THREADS, 0, stream>>>(
        boards,
        current_player,
        stones_left,
        pending_second,
        actions,
        batch);
    return cudaGetLastError();
}
