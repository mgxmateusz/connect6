#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {

constexpr int BOARD = 19;
constexpr int CELLS = BOARD * BOARD;
constexpr int THREADS = 256;
constexpr int WIN = 6;
constexpr int ROADS = 924;

constexpr int ROOT_K = 8;
constexpr int SECOND_K = 4;
constexpr int MAX_PAIRS = ROOT_K * SECOND_K;
constexpr int OWN_KEEP = 4;
constexpr int OPP_ROOT_K = 4;
constexpr int OPP_SECOND_K = 2;
constexpr int MAX_THREATS = 128;

constexpr int INVALID_SCORE = INT_MIN / 4;
constexpr int WIN_SCORE = 1000000000;

constexpr int STATE_WIN = 1500000000;
constexpr int STATE_IMMEDIATE = 1350000000;
constexpr int STATE_FORCED = 1200000000;

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

// V2 is deliberately used only for move ordering in the search.
// Its high tactical classes mean "this action is urgent", not "this board is good".
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

// -----------------------------------------------------------------------------
// Full-board evaluator. It scans every contiguous length-6 road exactly once.
// A threat road contains 4 or 5 stones of one player and no opponent stone.
// blocker_count is the exact minimum number of stones (capped at 3) required to
// hit every such road. In Connect6, >=3 means the defender cannot cover all
// winning-next-turn requirements with its two stones.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void decode_road(
    int road,
    int& start,
    int& dr,
    int& dc) {
    if (road < 266) {
        const int r = road / 14;
        const int c = road - r * 14;
        start = r * BOARD + c;
        dr = 0;
        dc = 1;
        return;
    }
    road -= 266;
    if (road < 266) {
        const int c = road / 14;
        const int r = road - c * 14;
        start = r * BOARD + c;
        dr = 1;
        dc = 0;
        return;
    }
    road -= 266;
    if (road < 196) {
        const int r = road / 14;
        const int c = road - r * 14;
        start = r * BOARD + c;
        dr = 1;
        dc = 1;
        return;
    }
    road -= 196;
    const int r = road / 14;
    const int c = 5 + (road - r * 14);
    start = r * BOARD + c;
    dr = 1;
    dc = -1;
}

__device__ __forceinline__ int road_soft_value(int stones, unsigned mask) {
    if (stones <= 0 || stones >= 6) return 0;
    const int run = max_run6(mask);
    const int span = span6(mask);
    const int compact = stones >= 2 ? (6 - span) : 0;
    switch (stones) {
        case 5: return 80000 + run * run * 1400 + compact * 500;
        case 4: return 12000 + run * run * 650 + compact * 300;
        case 3: return 1200 + run * run * 180 + compact * 90;
        case 2: return 80 + run * run * 25 + compact * 12;
        case 1: return 8;
        default: return 0;
    }
}

__device__ __forceinline__ bool threat_hit(
    int16_t a,
    int16_t b,
    int cell) {
    return static_cast<int>(a) == cell || static_cast<int>(b) == cell;
}

__device__ __forceinline__ int blockers_needed_exact(
    const int16_t* ta,
    const int16_t* tb,
    int count,
    int overflow) {
    if (count <= 0) return 0;
    // Overflow is intentionally conservative: never claim a forced >=3 result
    // from truncated data.
    if (overflow) return 2;
    if (count > MAX_THREATS) count = MAX_THREATS;

    const int first_a = static_cast<int>(ta[0]);
    const int first_b = static_cast<int>(tb[0]);
    int first_candidates[2] = {first_a, first_b};

    #pragma unroll
    for (int xi = 0; xi < 2; ++xi) {
        const int x = first_candidates[xi];
        if (x < 0) continue;
        if (xi == 1 && x == first_candidates[0]) continue;

        bool all_x = true;
        int first_unhit = -1;
        for (int i = 0; i < count; ++i) {
            if (!threat_hit(ta[i], tb[i], x)) {
                all_x = false;
                first_unhit = i;
                break;
            }
        }
        if (all_x) return 1;

        int y_candidates[2] = {
            static_cast<int>(ta[first_unhit]),
            static_cast<int>(tb[first_unhit])
        };
        #pragma unroll
        for (int yi = 0; yi < 2; ++yi) {
            const int y = y_candidates[yi];
            if (y < 0 || y == x) continue;
            if (yi == 1 && y == y_candidates[0]) continue;

            bool all_xy = true;
            for (int i = 0; i < count; ++i) {
                if (!threat_hit(ta[i], tb[i], x) &&
                    !threat_hit(ta[i], tb[i], y)) {
                    all_xy = false;
                    break;
                }
            }
            if (all_xy) return 2;
        }
    }
    return 3;
}

__device__ __forceinline__ int evaluate_state_parallel(
    const int8_t* board,
    int8_t perspective,
    int8_t side_to_move,
    int* reduce,
    int16_t* my_ta,
    int16_t* my_tb,
    int16_t* opp_ta,
    int16_t* opp_tb,
    int* my_threat_count,
    int* opp_threat_count,
    int* my_overflow,
    int* opp_overflow,
    int* my_win,
    int* opp_win,
    int* result) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *my_threat_count = 0;
        *opp_threat_count = 0;
        *my_overflow = 0;
        *opp_overflow = 0;
        *my_win = 0;
        *opp_win = 0;
    }
    __syncthreads();

    int local_soft = 0;
    for (int road = tid; road < ROADS; road += blockDim.x) {
        int start, dr, dc;
        decode_road(road, start, dr, dc);

        int mine = 0;
        int opp = 0;
        int empty0 = -1;
        int empty1 = -1;
        int empties = 0;
        unsigned mine_mask = 0;
        unsigned opp_mask = 0;

        int r = start / BOARD;
        int c = start - r * BOARD;
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            const int idx = (r + k * dr) * BOARD + (c + k * dc);
            const int8_t v = board[idx];
            if (v == perspective) {
                ++mine;
                mine_mask |= 1u << k;
            } else if (v == -perspective) {
                ++opp;
                opp_mask |= 1u << k;
            } else {
                if (empties == 0) empty0 = idx;
                else if (empties == 1) empty1 = idx;
                ++empties;
            }
        }

        if (mine == WIN) atomicExch(my_win, 1);
        if (opp == WIN) atomicExch(opp_win, 1);

        if (opp == 0 && mine > 0) {
            local_soft += road_soft_value(mine, mine_mask);
            if ((mine == 4 || mine == 5) && empties >= 1 && empties <= 2) {
                const int slot = atomicAdd(my_threat_count, 1);
                if (slot < MAX_THREATS) {
                    my_ta[slot] = static_cast<int16_t>(empty0);
                    my_tb[slot] = static_cast<int16_t>(empties == 2 ? empty1 : -1);
                } else {
                    atomicExch(my_overflow, 1);
                }
            }
        } else if (mine == 0 && opp > 0) {
            // Slightly defensive soft bias. Hard tactical classes below remain
            // symmetric and dominate this arithmetic.
            local_soft -= (road_soft_value(opp, opp_mask) * 11) / 10;
            if ((opp == 4 || opp == 5) && empties >= 1 && empties <= 2) {
                const int slot = atomicAdd(opp_threat_count, 1);
                if (slot < MAX_THREATS) {
                    opp_ta[slot] = static_cast<int16_t>(empty0);
                    opp_tb[slot] = static_cast<int16_t>(empties == 2 ? empty1 : -1);
                } else {
                    atomicExch(opp_overflow, 1);
                }
            }
        }
    }

    reduce[tid] = local_soft;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        if (*my_win) {
            *result = STATE_WIN;
        } else if (*opp_win) {
            *result = -STATE_WIN;
        } else {
            const int mt = *my_threat_count;
            const int ot = *opp_threat_count;
            const int mb = blockers_needed_exact(
                my_ta, my_tb, mt, *my_overflow);
            const int ob = blockers_needed_exact(
                opp_ta, opp_tb, ot, *opp_overflow);

            // A clean 4/5-stone road is completable with the two stones of the
            // side to move. Check this before "forced next turn" pressure.
            if (side_to_move == perspective && mt > 0) {
                *result = STATE_IMMEDIATE;
            } else if (side_to_move == -perspective && ot > 0) {
                *result = -STATE_IMMEDIATE;
            } else if (side_to_move == -perspective && mb >= 3) {
                *result = STATE_FORCED;
            } else if (side_to_move == perspective && ob >= 3) {
                *result = -STATE_FORCED;
            } else {
                int value = reduce[0];

                if (mb == 2) value += 50000000;
                else if (mb == 1) value += 12000000;

                if (ob == 2) value -= 55000000;
                else if (ob == 1) value -= 13500000;

                const int mt_cap = mt < 12 ? mt : 12;
                const int ot_cap = ot < 12 ? ot : 12;
                value += mt_cap * 650000;
                value -= ot_cap * 720000;

                *result = value;
            }
        }
    }
    __syncthreads();
    return *result;
}

__device__ __forceinline__ bool pair_better(
    int value,
    int64_t order,
    int first,
    int second,
    int best_value,
    int64_t best_order,
    int best_first,
    int best_second) {
    if (value != best_value) return value > best_value;
    if (order != best_order) return order > best_order;
    if (first != best_first) return first < best_first;
    return second < best_second;
}

__global__ void tactical_search_v3_kernel(
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

    __shared__ int root_actions[ROOT_K];
    __shared__ int root_scores[ROOT_K];

    __shared__ int pair_first[MAX_PAIRS];
    __shared__ int pair_second[MAX_PAIRS];
    __shared__ int pair_state[MAX_PAIRS];
    __shared__ int64_t pair_order[MAX_PAIRS];
    __shared__ int pair_count;

    __shared__ int keep_idx[OWN_KEEP];
    __shared__ int opp_root_actions[OPP_ROOT_K];
    __shared__ int opp_root_scores[OPP_ROOT_K];
    __shared__ int opp_second_actions[OPP_SECOND_K];
    __shared__ int opp_second_scores[OPP_SECOND_K];
    __shared__ int reply_worst;
    __shared__ int reply_seen;

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

    // Cached stone #2 from the previous call.
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

    // Opening/single-stone fallback or cache miss: greedy V2.
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

    // 1) V2 move ordering for the first stone.
    score_all(board, player, 2, scores);
    __syncthreads();
    if (tid == 0) {
        select_top_k_serial<ROOT_K>(scores, root_actions, root_scores);
        pair_count = 0;

        // A win on stone #1 ends the game before any cached second stone exists.
        if (root_actions[0] >= 0 && root_scores[0] >= WIN_SCORE) {
            selected_first = root_actions[0];
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

    // 2) Generate up to 32 unique final own-pair states: TOP8 first x TOP4 second.
    #pragma unroll
    for (int r = 0; r < ROOT_K; ++r) {
        const int first = root_actions[r];
        if (first >= 0) {
            if (tid == 0) board[first] = player;
            __syncthreads();

            score_all(board, player, 1, scores);
            __syncthreads();

            if (tid == 0) {
                int child_actions[SECOND_K];
                int child_scores[SECOND_K];
                select_top_k_serial<SECOND_K>(scores, child_actions, child_scores);

                #pragma unroll
                for (int k = 0; k < SECOND_K; ++k) {
                    const int second = child_actions[k];
                    if (second < 0) continue;

                    const int lo = first < second ? first : second;
                    const int hi = first < second ? second : first;
                    const int64_t order =
                        static_cast<int64_t>(root_scores[r]) +
                        static_cast<int64_t>(child_scores[k]);

                    int duplicate = -1;
                    for (int p = 0; p < pair_count; ++p) {
                        const int plo =
                            pair_first[p] < pair_second[p] ? pair_first[p] : pair_second[p];
                        const int phi =
                            pair_first[p] < pair_second[p] ? pair_second[p] : pair_first[p];
                        if (plo == lo && phi == hi) {
                            duplicate = p;
                            break;
                        }
                    }

                    if (duplicate >= 0) {
                        if (order > pair_order[duplicate]) {
                            pair_first[duplicate] = first;
                            pair_second[duplicate] = second;
                            pair_order[duplicate] = order;
                        }
                    } else if (pair_count < MAX_PAIRS) {
                        const int p = pair_count++;
                        pair_first[p] = first;
                        pair_second[p] = second;
                        pair_order[p] = order;
                        pair_state[p] = INVALID_SCORE;
                    }
                }
            }
            __syncthreads();

            if (tid == 0) board[first] = 0;
            __syncthreads();
        }
    }

    // 3) Evaluate the actual board after both own stones. A+B and B+A now have
    // the same state value; V2 action urgency is no longer abused as leaf value.
    for (int p = 0; p < pair_count; ++p) {
        const int first = pair_first[p];
        const int second = pair_second[p];
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

        if (tid == 0) pair_state[p] = value;
        __syncthreads();

        if (tid == 0) {
            board[first] = 0;
            board[second] = 0;
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < OWN_KEEP; ++i) keep_idx[i] = -1;

        for (int p = 0; p < pair_count; ++p) {
            for (int pos = 0; pos < OWN_KEEP; ++pos) {
                const int cur = keep_idx[pos];
                const bool take =
                    cur < 0 ||
                    pair_better(
                        pair_state[p], pair_order[p], pair_first[p], pair_second[p],
                        pair_state[cur], pair_order[cur], pair_first[cur], pair_second[cur]);
                if (take) {
                    for (int s = OWN_KEEP - 1; s > pos; --s)
                        keep_idx[s] = keep_idx[s - 1];
                    keep_idx[pos] = p;
                    break;
                }
            }
        }

        if (keep_idx[0] >= 0 && pair_state[keep_idx[0]] >= STATE_FORCED) {
            const int p = keep_idx[0];
            selected_first = pair_first[p];
            selected_second = pair_second[p];
        } else {
            selected_first = -1;
            selected_second = -1;
        }
    }
    __syncthreads();

    // Exact >=3 threat pressure is already a forced win unless the opponent can
    // win immediately; evaluate_state checks that first, so no reply tree needed.
    if (selected_first >= 0) {
        if (tid == 0) {
            actions[board_id] = static_cast<int64_t>(selected_first);
            pending_second[board_id] = static_cast<int16_t>(selected_second);
        }
        return;
    }

    // 4) Full opponent-turn minimax only for the four best own pairs.
    int best_pair_value = INVALID_SCORE;
    int64_t best_pair_order = static_cast<int64_t>(LLONG_MIN);
    int best_pair_first = -1;
    int best_pair_second = -1;

    #pragma unroll
    for (int q = 0; q < OWN_KEEP; ++q) {
        const int p = keep_idx[q];
        if (p < 0) continue;

        const int first = pair_first[p];
        const int second = pair_second[p];

        if (tid == 0) {
            board[first] = player;
            board[second] = player;
        }
        __syncthreads();

        score_all(board, static_cast<int8_t>(-player), 2, scores);
        __syncthreads();

        if (tid == 0) {
            select_top_k_serial<OPP_ROOT_K>(
                scores, opp_root_actions, opp_root_scores);
            reply_worst = STATE_WIN;
            reply_seen = 0;
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < OPP_ROOT_K; ++r) {
            const int ofirst = opp_root_actions[r];
            if (ofirst < 0) continue;

            if (opp_root_scores[r] >= WIN_SCORE) {
                if (tid == 0) {
                    reply_worst = -STATE_WIN;
                    reply_seen = 1;
                }
                __syncthreads();
                break;
            }

            if (tid == 0) board[ofirst] = static_cast<int8_t>(-player);
            __syncthreads();

            score_all(board, static_cast<int8_t>(-player), 1, scores);
            __syncthreads();

            if (tid == 0) {
                select_top_k_serial<OPP_SECOND_K>(
                    scores, opp_second_actions, opp_second_scores);
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < OPP_SECOND_K; ++k) {
                const int osecond = opp_second_actions[k];
                if (osecond < 0) continue;

                if (tid == 0) board[osecond] = static_cast<int8_t>(-player);
                __syncthreads();

                const int leaf = evaluate_state_parallel(
                    board,
                    player,
                    player,
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
                    board[osecond] = 0;
                }
                __syncthreads();
            }

            if (tid == 0) board[ofirst] = 0;
            __syncthreads();
        }

        if (tid == 0 && !reply_seen) reply_worst = pair_state[p];
        __syncthreads();

        if (tid == 0) {
            board[first] = 0;
            board[second] = 0;

            if (best_pair_first < 0 ||
                pair_better(
                    reply_worst, pair_order[p], first, second,
                    best_pair_value, best_pair_order,
                    best_pair_first, best_pair_second)) {
                best_pair_value = reply_worst;
                best_pair_order = pair_order[p];
                best_pair_first = first;
                best_pair_second = second;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (best_pair_first < 0) {
            selected_first = root_actions[0];
            selected_second = -1;
        } else {
            selected_first = best_pair_first;
            selected_second = best_pair_second;
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
    tactical_search_v3_kernel<<<batch, THREADS, 0, stream>>>(
        boards,
        current_player,
        stones_left,
        pending_second,
        actions,
        batch);
    return cudaGetLastError();
}
