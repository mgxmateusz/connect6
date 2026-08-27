#include "native_bot_pair_eval_v4.cuh"

namespace {

using namespace v4_detail;

constexpr int CANDIDATE_K = 8;
constexpr int KEEP_K = 3;

__global__ void tactical_search_v4_top8_reply1_kernel(
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

    __shared__ int keep_first[KEEP_K];
    __shared__ int keep_second[KEEP_K];
    __shared__ int keep_value[KEEP_K];
    __shared__ int64_t keep_order[KEEP_K];

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

    // Opening/single-stone fallback: greedy V2.
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

    // 1) Rank the current board once and keep the eight strongest cells.
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

        #pragma unroll
        for (int k = 0; k < KEEP_K; ++k) {
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

    // 2) Evaluate every unordered pair from TOP8: C(8,2)=28 states.
    // Keep only the three best complete own-pair states for the reply stage.
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

                for (int pos = 0; pos < KEEP_K; ++pos) {
                    const bool take =
                        keep_first[pos] < 0 ||
                        pair_better(
                            value,
                            order,
                            first,
                            second,
                            keep_value[pos],
                            keep_order[pos],
                            keep_first[pos],
                            keep_second[pos]);
                    if (take) {
                        for (int s = KEEP_K - 1; s > pos; --s) {
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

    // A terminal six ends the game before the opponent can answer.
    if (keep_first[0] >= 0 && keep_value[0] >= STATE_WIN) {
        if (tid == 0) {
            actions[board_id] = static_cast<int64_t>(keep_first[0]);
            pending_second[board_id] = static_cast<int16_t>(keep_second[0]);
        }
        return;
    }

    // 3) For each of the TOP3 own pairs, try every legal ONE-STONE opponent
    // reply. The opponent chooses the leaf that is worst for us. This is an
    // intentionally shallow one-stone reply experiment, not a full Connect6
    // opponent turn.
    int best_pair_value = INVALID_SCORE;
    int64_t best_pair_order = static_cast<int64_t>(LLONG_MIN);
    int best_pair_first = -1;
    int best_pair_second = -1;

    #pragma unroll
    for (int q = 0; q < KEEP_K; ++q) {
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

        for (int reply = 0; reply < CELLS; ++reply) {
            if (board[reply] != 0) continue;

            if (tid == 0) board[reply] = static_cast<int8_t>(-player);
            __syncthreads();

            // side_to_move=0 deliberately asks for a neutral board-state score.
            // We stop after only one opponent stone, so pretending that a fresh
            // full two-stone turn starts here would distort STATE_IMMEDIATE.
            const int leaf = evaluate_state_parallel(
                board,
                player,
                static_cast<int8_t>(0),
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
                if (!reply_seen || leaf < reply_worst) reply_worst = leaf;
                reply_seen = 1;
                board[reply] = 0;
            }
            __syncthreads();

            if (reply_worst <= -STATE_WIN) break;
        }

        if (tid == 0) {
            if (!reply_seen) reply_worst = keep_value[q];
            board[first] = 0;
            board[second] = 0;

            if (best_pair_first < 0 ||
                pair_better(
                    reply_worst,
                    keep_order[q],
                    first,
                    second,
                    best_pair_value,
                    best_pair_order,
                    best_pair_first,
                    best_pair_second)) {
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

extern "C" cudaError_t launch_tactical_bot_v4_top8_reply1_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_v4_top8_reply1_kernel<<<batch, v4_detail::THREADS, 0, stream>>>(
        boards,
        current_player,
        stones_left,
        pending_second,
        actions,
        batch);
    return cudaGetLastError();
}
