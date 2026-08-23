#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace connect6_rollout_bot {

constexpr int BOARD = 19;
constexpr int CELLS = BOARD * BOARD;
constexpr int THREADS = 256;
constexpr int WIN = 6;

__device__ __forceinline__ bool inside(int r, int c) {
    return static_cast<unsigned>(r) < BOARD && static_cast<unsigned>(c) < BOARD;
}

__device__ __forceinline__ int iabs(int x) {
    return x < 0 ? -x : x;
}

__device__ __forceinline__ int window_value(int stones_after, int open_ends) {
    switch (stones_after) {
        case 5: return 400000 + open_ends * 30000;
        case 4: return 22000 + open_ends * 2500;
        case 3: return 1600 + open_ends * 250;
        case 2: return 120 + open_ends * 20;
        case 1: return 8 + open_ends * 2;
        default: return 0;
    }
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

__device__ __forceinline__ int score_cell_v1(
    const int8_t* board,
    int action,
    int8_t player,
    int8_t stones_left) {
    if (board[action] != 0) return INT_MIN / 4;

    const int row = action / BOARD;
    const int col = action - row * BOARD;
    constexpr int DR[4] = {0, 1, 1, 1};
    constexpr int DC[4] = {1, 0, 1, -1};

    int own_win = 0;
    int opp_win = 0;
    int own_five = 0;
    int opp_five = 0;
    int own_four = 0;
    int opp_four = 0;
    int own_three = 0;
    int opp_three = 0;
    int attack_score = 0;
    int defense_score = 0;

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        const int dr = DR[d];
        const int dc = DC[d];

        const int own_run = contiguous_after(board, row, col, dr, dc, player);
        const int opp_run = contiguous_after(board, row, col, dr, dc, -player);
        attack_score += own_run * own_run * 90;
        defense_score += opp_run * opp_run * 95;

        #pragma unroll
        for (int rel = 0; rel < WIN; ++rel) {
            const int sr = row - rel * dr;
            const int sc = col - rel * dc;
            const int er = sr + (WIN - 1) * dr;
            const int ec = sc + (WIN - 1) * dc;
            if (!inside(sr, sc) || !inside(er, ec)) continue;

            int mine = 0;
            int theirs = 0;
            #pragma unroll
            for (int k = 0; k < WIN; ++k) {
                const int rr = sr + k * dr;
                const int cc = sc + k * dc;
                const int8_t v = board[rr * BOARD + cc];
                mine += (v == player);
                theirs += (v == -player);
            }

            int open_ends = 0;
            const int br = sr - dr;
            const int bc = sc - dc;
            const int ar = er + dr;
            const int ac = ec + dc;
            if (inside(br, bc) && board[br * BOARD + bc] == 0) ++open_ends;
            if (inside(ar, ac) && board[ar * BOARD + ac] == 0) ++open_ends;

            if (theirs == 0) {
                const int after = mine + 1;
                attack_score += window_value(after, open_ends);
                own_win += (after >= 6);
                own_five += (after == 5);
                own_four += (after == 4);
                own_three += (after == 3);
            }

            if (mine == 0) {
                const int after = theirs + 1;
                defense_score += window_value(after, open_ends);
                opp_win += (after >= 6);
                opp_five += (after == 5);
                opp_four += (after == 4);
                opp_three += (after == 3);
            }
        }
    }

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
        own_four * 120000 + opp_four * 130000 +
        own_three * 7000 + opp_three * 7500 +
        nearby_own * 120 + nearby_opp * 100 + centre_bonus * 8;

    if (own_win > 0) return 1000000000 + own_win * 1000000 + quiet_score / 64;
    if (stones_left >= 2 && own_five > 0)
        return 950000000 + own_five * 1000000 + quiet_score / 64;
    if (opp_win > 0) return 900000000 + opp_win * 1000000 + quiet_score / 64;
    if (stones_left <= 1 && opp_five > 0)
        return 700000000 + opp_five * 1000000 + quiet_score / 64;
    if (own_five >= 2) return 520000000 + own_five * 1000000 + quiet_score / 32;
    if (opp_five >= 2) return 480000000 + opp_five * 1000000 + quiet_score / 32;
    if (own_five > 0) return 260000000 + own_five * 1000000 + quiet_score / 16;
    if (opp_five > 0) return 230000000 + opp_five * 1000000 + quiet_score / 16;
    if (own_four >= 2) return 85000000 + own_four * 300000 + quiet_score / 8;
    if (opp_four >= 2) return 78000000 + opp_four * 300000 + quiet_score / 8;
    return quiet_score;
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

__device__ __forceinline__ int score_cell_v2(
    const int8_t* board,
    int action,
    int8_t player,
    int8_t stones_left) {
    if (board[action] != 0) return INT_MIN / 4;

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

    if (own_win > 0) return 1000000000 + own_win * 1500000 + quiet_score / 64;
    if (stones_left >= 2 && own_finish > 0)
        return 960000000 + own_finish * 1500000 + quiet_score / 64;
    if (opp_win > 0) return 920000000 + opp_win * 1500000 + quiet_score / 64;
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
    if (own_finish == 1) return 280000000 + quiet_score / 16;
    if (opp_finish == 1) return 260000000 + quiet_score / 16;
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

template <bool V2, uint8_t MODE>
__global__ void tactical_bot_kernel(
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    const uint8_t* __restrict__ actor_mode,
    int16_t* __restrict__ actions,
    int slots) {
    const int board_id = blockIdx.x;
    if (board_id >= slots || actor_mode[board_id] != MODE) return;

    __shared__ int8_t shared_board[CELLS];
    __shared__ int shared_score[THREADS];
    __shared__ int shared_action[THREADS];

    const int tid = threadIdx.x;
    const int8_t* src = boards + static_cast<int64_t>(board_id) * CELLS;
    for (int i = tid; i < CELLS; i += blockDim.x) shared_board[i] = src[i];
    __syncthreads();

    int local_score = INT_MIN / 4;
    int local_action = -1;
    const int8_t player = current_player[board_id];
    const int8_t left = stones_left[board_id];

    for (int action = tid; action < CELLS; action += blockDim.x) {
        const int score = V2
            ? score_cell_v2(shared_board, action, player, left)
            : score_cell_v1(shared_board, action, player, left);
        if (better(score, action, local_score, local_action)) {
            local_score = score;
            local_action = action;
        }
    }

    shared_score[tid] = local_score;
    shared_action[tid] = local_action;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const int other_score = shared_score[tid + stride];
            const int other_action = shared_action[tid + stride];
            if (better(other_score, other_action, shared_score[tid], shared_action[tid])) {
                shared_score[tid] = other_score;
                shared_action[tid] = other_action;
            }
        }
        __syncthreads();
    }

    if (tid == 0) actions[board_id] = static_cast<int16_t>(shared_action[0]);
}

}  // namespace connect6_rollout_bot
