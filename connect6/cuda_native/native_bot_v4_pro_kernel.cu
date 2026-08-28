#include "native_bot_v4_reply6_kernel.cu"
#include "native_bot_v2_pro_score.cuh"

namespace {

using namespace v4_detail;
using v2pro_detail::score_all_pro;

constexpr int PRO_V4_CANDIDATE_K = 12;
constexpr int PRO_V4_KEEP_K = 4;
constexpr int PRO_V4_REPLY_K = 6;

// Identical V4 pair search/maximin. V2Pro replaces V2 for own TOP12 and for
// the opponent TOP6 reply-cell ordering.
__global__ void tactical_search_v4_pro_top12_replypair6_kernel(
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
    __shared__ int candidate_actions[PRO_V4_CANDIDATE_K];
    __shared__ int candidate_scores[PRO_V4_CANDIDATE_K];
    __shared__ int keep_first[PRO_V4_KEEP_K];
    __shared__ int keep_second[PRO_V4_KEEP_K];
    __shared__ int keep_value[PRO_V4_KEEP_K];
    __shared__ int64_t keep_order[PRO_V4_KEEP_K];
    __shared__ int reply_actions[PRO_V4_REPLY_K];
    __shared__ int reply_scores[PRO_V4_REPLY_K];

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
    __shared__ int reply_worst;
    __shared__ int reply_seen;
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
        select_top_k_serial<PRO_V4_CANDIDATE_K>(scores, candidate_actions, candidate_scores);
        if (candidate_actions[0] >= 0 && candidate_scores[0] >= WIN_SCORE) {
            selected_first = candidate_actions[0];
            selected_second = -1;
        } else {
            selected_first = -1;
            selected_second = -1;
        }
        #pragma unroll
        for (int k = 0; k < PRO_V4_KEEP_K; ++k) {
            keep_first[k] = -1;
            keep_second[k] = -1;
            keep_value[k] = INVALID_SCORE;
            keep_order[k] = static_cast<int64_t>(LLONG_MIN);
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

    #pragma unroll 1
    for (int i = 0; i < PRO_V4_CANDIDATE_K; ++i) {
        const int first = candidate_actions[i];
        if (first < 0) continue;
        #pragma unroll 1
        for (int j = i + 1; j < PRO_V4_CANDIDATE_K; ++j) {
            const int second = candidate_actions[j];
            if (second < 0) continue;
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
                const int64_t order = static_cast<int64_t>(candidate_scores[i]) +
                                      static_cast<int64_t>(candidate_scores[j]);
                for (int pos = 0; pos < PRO_V4_KEEP_K; ++pos) {
                    const bool take = keep_first[pos] < 0 ||
                        pair_better(value, order, first, second,
                                    keep_value[pos], keep_order[pos],
                                    keep_first[pos], keep_second[pos]);
                    if (take) {
                        for (int s = PRO_V4_KEEP_K - 1; s > pos; --s) {
                            keep_first[s] = keep_first[s - 1];
                            keep_second[s] = keep_second[s - 1];
                            keep_value[s] = keep_value[s - 1];
                            keep_order[s] = keep_order[s - 1];
                        }
                        keep_first[pos] = first;
                        keep_second[pos] = second;
                        keep_value[pos] = value;
                        keep_order[pos] = order;
                        break;
                    }
                }
                board[first] = 0;
                board[second] = 0;
            }
            __syncthreads();
        }
    }

    if (keep_first[0] >= 0 && keep_value[0] >= STATE_WIN) {
        if (tid == 0) {
            actions[board_id] = static_cast<int64_t>(keep_first[0]);
            pending_second[board_id] = static_cast<int16_t>(keep_second[0]);
        }
        return;
    }

    int best_pair_value = INVALID_SCORE;
    int64_t best_pair_order = static_cast<int64_t>(LLONG_MIN);
    int best_pair_first = -1;
    int best_pair_second = -1;

    #pragma unroll
    for (int q = 0; q < PRO_V4_KEEP_K; ++q) {
        const int first = keep_first[q];
        const int second = keep_second[q];
        if (first < 0 || second < 0) continue;

        if (tid == 0) {
            board[first] = player;
            board[second] = player;
            reply_worst = STATE_WIN;
            reply_seen = 0;
        }
        __syncthreads();

        score_all_pro(board, static_cast<int8_t>(-player), 2, scores);
        __syncthreads();
        if (tid == 0) {
            select_top_k_serial<PRO_V4_REPLY_K>(scores, reply_actions, reply_scores);
        }
        __syncthreads();

        #pragma unroll
        for (int r1 = 0; r1 < PRO_V4_REPLY_K; ++r1) {
            const int reply_first = reply_actions[r1];
            if (reply_first < 0 || board[reply_first] != 0) continue;
            #pragma unroll
            for (int r2 = r1 + 1; r2 < PRO_V4_REPLY_K; ++r2) {
                const int reply_second = reply_actions[r2];
                if (reply_second < 0 || board[reply_second] != 0) continue;
                if (tid == 0) {
                    board[reply_first] = static_cast<int8_t>(-player);
                    board[reply_second] = static_cast<int8_t>(-player);
                }
                __syncthreads();

                const int leaf = evaluate_state_parallel(
                    board, player, player, reduce,
                    my_ta, my_tb, opp_ta, opp_tb,
                    &my_threat_count, &opp_threat_count,
                    &my_overflow, &opp_overflow, &my_win, &opp_win, &eval_result);

                if (tid == 0) {
                    if (!reply_seen || leaf < reply_worst) reply_worst = leaf;
                    reply_seen = 1;
                    board[reply_first] = 0;
                    board[reply_second] = 0;
                }
                __syncthreads();
            }
        }

        if (tid == 0) {
            if (!reply_seen) reply_worst = keep_value[q];
            board[first] = 0;
            board[second] = 0;
            if (best_pair_first < 0 ||
                pair_better(reply_worst, keep_order[q], first, second,
                            best_pair_value, best_pair_order,
                            best_pair_first, best_pair_second)) {
                best_pair_value = reply_worst;
                best_pair_order = keep_order[q];
                best_pair_first = first;
                best_pair_second = second;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (best_pair_first < 0) {
            selected_first = keep_first[0] >= 0 ? keep_first[0] : candidate_actions[0];
            selected_second = keep_second[0];
        } else {
            selected_first = best_pair_first;
            selected_second = best_pair_second;
        }
        actions[board_id] = static_cast<int64_t>(selected_first);
        pending_second[board_id] = static_cast<int16_t>(selected_second);
    }
}

}  // namespace

extern "C" cudaError_t launch_tactical_bot_v4_pro_top12_replypair6_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_v4_pro_top12_replypair6_kernel<<<batch, v4_detail::THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}
