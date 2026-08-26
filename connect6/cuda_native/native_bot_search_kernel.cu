#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {

constexpr int BOARD = 19;
constexpr int CELLS = BOARD * BOARD;
constexpr int THREADS = 256;
constexpr int WIN = 6;
constexpr int MAX_ROOT = 8;
constexpr int MAX_PAIRS = 32;
constexpr int INVALID_SCORE = INT_MIN / 4;
constexpr int WIN_SCORE = 1000000000;

__device__ __forceinline__ bool inside(int r, int c) {
    return static_cast<unsigned>(r) < BOARD && static_cast<unsigned>(c) < BOARD;
}

__device__ __forceinline__ int iabs(int x) {
    return x < 0 ? -x : x;
}

__device__ __forceinline__ int popcount6(unsigned mask) {
    return __popc(mask & 0x3fu);
}

__device__ __forceinline__ int popcount64(uint64_t mask) {
    return __popcll(static_cast<unsigned long long>(mask));
}

__device__ __forceinline__ int max_run6(unsigned mask) {
    int best = 0;
    int run = 0;
    #pragma unroll
    for (int k = 0; k < WIN; ++k) {
        if (mask & (1u << k)) {
            ++run;
            if (run > best) best = run;
        } else {
            run = 0;
        }
    }
    return best;
}

__device__ __forceinline__ int span6(unsigned mask) {
    int first = WIN;
    int last = -1;
    #pragma unroll
    for (int k = 0; k < WIN; ++k) {
        if (mask & (1u << k)) {
            if (first == WIN) first = k;
            last = k;
        }
    }
    return last < 0 ? 0 : last - first + 1;
}

__device__ __forceinline__ int pattern_value_v2(unsigned mask, int open_ends) {
    const int stones = popcount6(mask);
    const int run = max_run6(mask);
    const int span = span6(mask);
    const int compact = stones >= 2 ? (6 - span) : 0;

    switch (stones) {
        case 5: return 460000 + open_ends * 35000 + run * 2500;
        case 4: return 30000 + run * run * 1800 + compact * 900 + open_ends * 3500;
        case 3: return 2200 + run * run * 320 + compact * 160 + open_ends * 320;
        case 2: return 180 + run * run * 45 + compact * 25 + open_ends * 25;
        case 1: return 12 + open_ends * 3;
        default: return 0;
    }
}

__device__ __forceinline__ uint64_t relative_threat_bit(int direction, int offset) {
    if (offset == 0 || offset < -5 || offset > 5) return 0;
    const int local = offset < 0 ? offset + 5 : offset + 4;
    return uint64_t{1} << (direction * 10 + local);
}

__device__ __forceinline__ int contiguous_after(
    const int8_t* board,
    int row,
    int col,
    int dr,
    int dc,
    int8_t stone) {
    int total = 1;
    #pragma unroll
    for (int sign = -1; sign <= 1; sign += 2) {
        #pragma unroll
        for (int k = 1; k < WIN; ++k) {
            const int rr = row + sign * k * dr;
            const int cc = col + sign * k * dc;
            if (!inside(rr, cc) || board[rr * BOARD + cc] != stone) break;
            ++total;
        }
    }
    return total;
}

__device__ __forceinline__ int score_cell_v2(
    const int8_t* board,
    int action,
    int8_t player,
    int8_t stones_left) {
    if (action < 0 || action >= CELLS || board[action] != 0) return INVALID_SCORE;

    const int row = action / BOARD;
    const int col = action - row * BOARD;
    constexpr int DR[4] = {0, 1, 1, 1};
    constexpr int DC[4] = {1, 0, 1, -1};

    int own_win = 0;
    int opp_win = 0;
    int own_four = 0;
    int opp_four = 0;
    int own_three = 0;
    int opp_three = 0;
    unsigned own_four_dirs = 0;
    unsigned opp_four_dirs = 0;
    unsigned own_three_dirs = 0;
    unsigned opp_three_dirs = 0;
    uint64_t own_finish_mask = 0;
    uint64_t opp_finish_mask = 0;
    int attack_score = 0;
    int defense_score = 0;

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        const int dr = DR[d];
        const int dc = DC[d];

        const int own_run = contiguous_after(board, row, col, dr, dc, player);
        const int opp_run = contiguous_after(board, row, col, dr, dc, -player);
        attack_score += own_run * own_run * 105;
        defense_score += opp_run * opp_run * 112;

        #pragma unroll
        for (int rel = 0; rel < WIN; ++rel) {
            const int sr = row - rel * dr;
            const int sc = col - rel * dc;
            const int er = sr + (WIN - 1) * dr;
            const int ec = sc + (WIN - 1) * dc;
            if (!inside(sr, sc) || !inside(er, ec)) continue;

            unsigned mine_mask = 0;
            unsigned theirs_mask = 0;
            #pragma unroll
            for (int k = 0; k < WIN; ++k) {
                const int rr = sr + k * dr;
                const int cc = sc + k * dc;
                const int8_t v = board[rr * BOARD + cc];
                mine_mask |= static_cast<unsigned>(v == player) << k;
                theirs_mask |= static_cast<unsigned>(v == -player) << k;
            }

            int open_ends = 0;
            const int br = sr - dr;
            const int bc = sc - dc;
            const int ar = er + dr;
            const int ac = ec + dc;
            if (inside(br, bc) && board[br * BOARD + bc] == 0) ++open_ends;
            if (inside(ar, ac) && board[ar * BOARD + ac] == 0) ++open_ends;

            const unsigned candidate_bit = 1u << rel;
            if (theirs_mask == 0) {
                const unsigned after_mask = mine_mask | candidate_bit;
                const int after = popcount6(after_mask);
                attack_score += pattern_value_v2(after_mask, open_ends);
                own_win += (after >= 6);
                if (after == 5) {
                    #pragma unroll
                    for (int k = 0; k < WIN; ++k) {
                        if ((after_mask & (1u << k)) == 0)
                            own_finish_mask |= relative_threat_bit(d, k - rel);
                    }
                } else if (after == 4) {
                    ++own_four;
                    own_four_dirs |= 1u << d;
                } else if (after == 3) {
                    ++own_three;
                    own_three_dirs |= 1u << d;
                }
            }

            if (mine_mask == 0) {
                const unsigned after_mask = theirs_mask | candidate_bit;
                const int after = popcount6(after_mask);
                defense_score += pattern_value_v2(after_mask, open_ends);
                opp_win += (after >= 6);
                if (after == 5) {
                    #pragma unroll
                    for (int k = 0; k < WIN; ++k) {
                        if ((after_mask & (1u << k)) == 0)
                            opp_finish_mask |= relative_threat_bit(d, k - rel);
                    }
                } else if (after == 4) {
                    ++opp_four;
                    opp_four_dirs |= 1u << d;
                } else if (after == 3) {
                    ++opp_three;
                    opp_three_dirs |= 1u << d;
                }
            }
        }
    }

    const int own_finish = popcount64(own_finish_mask);
    const int opp_finish = popcount64(opp_finish_mask);
    const int own_four_dir_count = __popc(own_four_dirs);
    const int opp_four_dir_count = __popc(opp_four_dirs);
    const int own_three_dir_count = __popc(own_three_dirs);
    const int opp_three_dir_count = __popc(opp_three_dirs);

    int nearby_own = 0;
    int nearby_opp = 0;
    for (int rr = row - 2; rr <= row + 2; ++rr) {
        for (int cc = col - 2; cc <= col + 2; ++cc) {
            if (!inside(rr, cc) || (rr == row && cc == col)) continue;
            const int8_t v = board[rr * BOARD + cc];
            nearby_own += (v == player);
            nearby_opp += (v == -player);
        }
    }

    const int centre_bonus = 18 - (iabs(row - 9) + iabs(col - 9));
    const int quiet_score =
        attack_score * 4 + defense_score * 5 +
        own_four * 145000 + opp_four * 160000 +
        own_three * 8500 + opp_three * 9500 +
        own_four_dir_count * 18000 + opp_four_dir_count * 21000 +
        own_three_dir_count * 1800 + opp_three_dir_count * 2200 +
        nearby_own * 125 + nearby_opp * 110 + centre_bonus * 8;

    if (own_win > 0)
        return 1000000000 + own_win * 1500000 + quiet_score / 64;
    if (stones_left >= 2 && own_finish > 0)
        return 960000000 + own_finish * 1500000 + quiet_score / 64;
    if (opp_win > 0)
        return 920000000 + opp_win * 1500000 + quiet_score / 64;
    if (stones_left <= 1 && opp_finish > 0)
        return 820000000 + opp_finish * 1500000 + quiet_score / 64;
    if (stones_left <= 1 && own_finish >= 3)
        return 650000000 + own_finish * 1800000 + quiet_score / 32;
    if (opp_finish >= 3)
        return 620000000 + opp_finish * 1800000 + quiet_score / 32;
    if (own_finish >= 2)
        return 360000000 + own_finish * 1200000 + quiet_score / 24;
    if (opp_finish >= 2)
        return 340000000 + opp_finish * 1200000 + quiet_score / 24;
    if (own_finish == 1)
        return 280000000 + quiet_score / 16;
    if (opp_finish == 1)
        return 260000000 + quiet_score / 16;
    if (own_four >= 2 || own_four_dir_count >= 2)
        return 105000000 + own_four * 350000 + own_four_dir_count * 800000 + quiet_score / 8;
    if (opp_four >= 2 || opp_four_dir_count >= 2)
        return 98000000 + opp_four * 350000 + opp_four_dir_count * 800000 + quiet_score / 8;
    return quiet_score;
}

__device__ __forceinline__ bool better(
    int candidate_score,
    int candidate_action,
    int best_score,
    int best_action) {
    return candidate_score > best_score ||
        (candidate_score == best_score && candidate_action >= 0 &&
         (best_action < 0 || candidate_action < best_action));
}

template<int K>
__device__ __forceinline__ void select_top_k_serial(
    const int* scores,
    int* out_actions,
    int* out_scores) {
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        out_actions[i] = -1;
        out_scores[i] = INVALID_SCORE;
    }

    for (int action = 0; action < CELLS; ++action) {
        const int score = scores[action];
        if (score <= INVALID_SCORE) continue;
        #pragma unroll
        for (int pos = 0; pos < K; ++pos) {
            if (better(score, action, out_scores[pos], out_actions[pos])) {
                #pragma unroll
                for (int shift = K - 1; shift > pos; --shift) {
                    out_actions[shift] = out_actions[shift - 1];
                    out_scores[shift] = out_scores[shift - 1];
                }
                out_actions[pos] = action;
                out_scores[pos] = score;
                break;
            }
        }
    }
}

__device__ __forceinline__ void score_all(
    const int8_t* board,
    int8_t player,
    int8_t stones_left,
    int* scores) {
    const int tid = threadIdx.x;
    for (int action = tid; action < CELLS; action += blockDim.x) {
        scores[action] = score_cell_v2(board, action, player, stones_left);
    }
}

template<int ROOT_K, int SECOND_K, bool DEPTH3>
__global__ void tactical_search_kernel(
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    int16_t* __restrict__ pending_second,
    int64_t* __restrict__ actions,
    int batch) {
    static_assert(ROOT_K <= MAX_ROOT, "ROOT_K exceeds shared root capacity");
    static_assert(!DEPTH3 || ROOT_K * SECOND_K <= MAX_PAIRS,
                  "search beam exceeds shared pair capacity");

    const int board_id = blockIdx.x;
    if (board_id >= batch) return;

    const int tid = threadIdx.x;
    const int8_t* src = boards + static_cast<int64_t>(board_id) * CELLS;
    const int8_t player = current_player[board_id];
    const int8_t left = stones_left[board_id];

    __shared__ int8_t board[CELLS];
    __shared__ int scores[CELLS];
    __shared__ int root_actions[MAX_ROOT];
    __shared__ int root_scores[MAX_ROOT];
    __shared__ int pair_first[MAX_PAIRS];
    __shared__ int pair_second[MAX_PAIRS];
    __shared__ int pair_scores[MAX_PAIRS];
    __shared__ int reply_scores[MAX_PAIRS];
    __shared__ int selected_first;
    __shared__ int selected_second;

    // Second stone of a previously planned pair: return it without re-searching.
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
        // Any stale cache belongs to an old/aborted turn.
        pending_second[board_id] = static_cast<int16_t>(-1);
    }

    for (int i = tid; i < CELLS; i += blockDim.x) board[i] = src[i];
    __syncthreads();

    // Opening/single-stone fallback or cache miss: normal greedy V2.
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

    // Ply 1: score all legal first stones and keep ROOT_K.
    score_all(board, player, 2, scores);
    __syncthreads();
    if (tid == 0) {
        select_top_k_serial<ROOT_K>(scores, root_actions, root_scores);
    }
    __syncthreads();

    // Ply 2: after every retained first stone, score all legal second stones.
    // V3 keeps only the best second stone per root. V4/V5 keep SECOND_K.
    #pragma unroll
    for (int r = 0; r < ROOT_K; ++r) {
        const int first = root_actions[r];
        if (first >= 0) {
            if (tid == 0) board[first] = player;
            __syncthreads();

            score_all(board, player, 1, scores);
            __syncthreads();

            if (tid == 0) {
                if constexpr (DEPTH3) {
                    int child_actions[SECOND_K];
                    int child_scores[SECOND_K];
                    select_top_k_serial<SECOND_K>(scores, child_actions, child_scores);
                    #pragma unroll
                    for (int k = 0; k < SECOND_K; ++k) {
                        const int p = r * SECOND_K + k;
                        pair_first[p] = first;
                        pair_second[p] = child_actions[k];
                        pair_scores[p] = child_scores[k];
                    }
                } else {
                    int child_actions[1];
                    int child_scores[1];
                    select_top_k_serial<1>(scores, child_actions, child_scores);
                    pair_first[r] = first;
                    pair_second[r] = child_actions[0];
                    pair_scores[r] = child_scores[0];
                }
            }
            __syncthreads();

            if (tid == 0) board[first] = 0;
            __syncthreads();
        } else if (tid == 0) {
            if constexpr (DEPTH3) {
                #pragma unroll
                for (int k = 0; k < SECOND_K; ++k) {
                    const int p = r * SECOND_K + k;
                    pair_first[p] = -1;
                    pair_second[p] = -1;
                    pair_scores[p] = INVALID_SCORE;
                }
            } else {
                pair_first[r] = -1;
                pair_second[r] = -1;
                pair_scores[r] = INVALID_SCORE;
            }
        }
        __syncthreads();
    }

    if constexpr (!DEPTH3) {
        if (tid == 0) {
            int best = -1;
            int best_second_score = INVALID_SCORE;
            int best_root_score = INVALID_SCORE;

            // An immediate win on the first stone must never be discarded by
            // the pair heuristic. The game will end before cached stone #2.
            if (root_actions[0] >= 0 && root_scores[0] >= WIN_SCORE) {
                best = 0;
            } else {
                #pragma unroll
                for (int r = 0; r < ROOT_K; ++r) {
                    if (pair_first[r] < 0) continue;
                    const int ps = pair_scores[r];
                    const int rs = root_scores[r];
                    if (best < 0 || ps > best_second_score ||
                        (ps == best_second_score && rs > best_root_score) ||
                        (ps == best_second_score && rs == best_root_score &&
                         pair_first[r] < pair_first[best])) {
                        best = r;
                        best_second_score = ps;
                        best_root_score = rs;
                    }
                }
            }

            if (best < 0) {
                selected_first = root_actions[0];
                selected_second = -1;
            } else {
                selected_first = pair_first[best];
                selected_second = pair_second[best];
            }
            actions[board_id] = static_cast<int64_t>(selected_first);
            pending_second[board_id] = static_cast<int16_t>(selected_second);
        }
        return;
    }

    constexpr int PAIRS = ROOT_K * SECOND_K;

    // Ply 3: the opponent gets its first stone. For every candidate pair,
    // measure the opponent's strongest V2 reply. Minimax then prefers the pair
    // whose best opponent reply is weakest.
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        const int first = pair_first[p];
        const int second = pair_second[p];
        if (first >= 0 && second >= 0) {
            if (tid == 0) {
                board[first] = player;
                board[second] = player;
            }
            __syncthreads();

            score_all(board, static_cast<int8_t>(-player), 2, scores);
            __syncthreads();

            if (tid == 0) {
                int reply_action[1];
                int reply_score[1];
                select_top_k_serial<1>(scores, reply_action, reply_score);
                reply_scores[p] = reply_score[0];
            }
            __syncthreads();

            if (tid == 0) {
                board[first] = 0;
                board[second] = 0;
            }
            __syncthreads();
        } else if (tid == 0) {
            reply_scores[p] = INVALID_SCORE;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int best = -1;

        // Same first-ply terminal guard as V3.
        if (root_actions[0] >= 0 && root_scores[0] >= WIN_SCORE) {
            for (int p = 0; p < PAIRS; ++p) {
                if (pair_first[p] == root_actions[0]) {
                    best = p;
                    break;
                }
            }
        }

        // A pair that wins on our second stone dominates any minimax reply.
        if (best < 0) {
            for (int p = 0; p < PAIRS; ++p) {
                if (pair_first[p] < 0 || pair_second[p] < 0) continue;
                if (pair_scores[p] < WIN_SCORE) continue;
                if (best < 0 ||
                    pair_scores[p] > pair_scores[best] ||
                    (pair_scores[p] == pair_scores[best] &&
                     pair_first[p] < pair_first[best]) ||
                    (pair_scores[p] == pair_scores[best] &&
                     pair_first[p] == pair_first[best] &&
                     pair_second[p] < pair_second[best])) {
                    best = p;
                }
            }
        }

        // Otherwise minimize the opponent's best first reply. Pair score is a
        // secondary tie-break so equally safe branches keep our stronger move.
        if (best < 0) {
            for (int p = 0; p < PAIRS; ++p) {
                if (pair_first[p] < 0 || pair_second[p] < 0) continue;
                if (best < 0 ||
                    reply_scores[p] < reply_scores[best] ||
                    (reply_scores[p] == reply_scores[best] &&
                     pair_scores[p] > pair_scores[best]) ||
                    (reply_scores[p] == reply_scores[best] &&
                     pair_scores[p] == pair_scores[best] &&
                     pair_first[p] < pair_first[best]) ||
                    (reply_scores[p] == reply_scores[best] &&
                     pair_scores[p] == pair_scores[best] &&
                     pair_first[p] == pair_first[best] &&
                     pair_second[p] < pair_second[best])) {
                    best = p;
                }
            }
        }

        if (best < 0) {
            selected_first = root_actions[0];
            selected_second = -1;
        } else {
            selected_first = pair_first[best];
            selected_second = pair_second[best];
        }
        actions[board_id] = static_cast<int64_t>(selected_first);
        pending_second[board_id] = static_cast<int16_t>(selected_second);
    }
}

}  // namespace

extern "C" cudaError_t launch_tactical_bot_v3_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_kernel<8, 1, false><<<batch, THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_tactical_bot_v4_8x2_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_kernel<8, 2, true><<<batch, THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_tactical_bot_v5_8x4_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_kernel<8, 4, true><<<batch, THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}