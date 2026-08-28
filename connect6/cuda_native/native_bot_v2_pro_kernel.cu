#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {

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

__device__ __forceinline__ int pattern_value(unsigned mask, int open_ends) {
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

__device__ __forceinline__ uint64_t relative_bit(int direction, int offset) {
    // Ten cells on each line through the candidate: -5..-1,+1..+5.
    // Four directions fit in 40 bits. Different directions intentionally remain
    // distinct because they represent independent tactical roads.
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

struct SideFeatures {
    int win = 0;
    int four = 0;
    int three = 0;
    int two = 0;
    unsigned four_dirs = 0;
    unsigned three_dirs = 0;
    unsigned two_dirs = 0;
    uint64_t finish_mask = 0;
    uint64_t activation_mask = 0;
    uint64_t four_empty_mask = 0;
    int soft = 0;
};

__device__ __forceinline__ void accumulate_after_pattern(
    SideFeatures& f,
    unsigned after_mask,
    int open_ends,
    int direction,
    int rel) {
    const int after = popcount6(after_mask);
    f.soft += pattern_value(after_mask, open_ends);

    if (after >= WIN) {
        ++f.win;
        return;
    }

    if (after == 5) {
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            if ((after_mask & (1u << k)) == 0)
                f.finish_mask |= relative_bit(direction, k - rel);
        }
        return;
    }

    if (after == 4) {
        ++f.four;
        f.four_dirs |= 1u << direction;
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            if ((after_mask & (1u << k)) == 0)
                f.four_empty_mask |= relative_bit(direction, k - rel);
        }
        return;
    }

    if (after == 3) {
        ++f.three;
        f.three_dirs |= 1u << direction;
        // These are the possible partner cells for the *second* stone of a
        // future Connect6 pair. A candidate which exposes many such cells, or
        // exposes them in several independent roads, is a latent fork anchor.
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            if ((after_mask & (1u << k)) == 0)
                f.activation_mask |= relative_bit(direction, k - rel);
        }
        return;
    }

    if (after == 2) {
        ++f.two;
        f.two_dirs |= 1u << direction;
    }
}

__device__ __forceinline__ bool latent_fork_anchor(
    int three_windows,
    int three_dirs,
    int activation_cells) {
    // Proxy for the exploit seen in the latest CNN: before the killer pair
    // there is no immediate 4/5. Instead one anchor participates in several
    // clean 2->3 roads; the partner stone then turns several of them into
    // simultaneous 4-in-6 threats whose blocker hitting-set is >2.
    return
        (three_windows >= 3 && activation_cells >= 4) ||
        (three_dirs >= 2 && three_windows >= 2 && activation_cells >= 4) ||
        (three_windows >= 2 && activation_cells >= 7);
}

__device__ __forceinline__ int score_cell_v2_pro(
    const int8_t* board,
    int action,
    int8_t player,
    int8_t stones_left) {
    if (action < 0 || action >= CELLS || board[action] != 0)
        return INT_MIN / 4;

    const int row = action / BOARD;
    const int col = action - row * BOARD;
    constexpr int DR[4] = {0, 1, 1, 1};
    constexpr int DC[4] = {1, 0, 1, -1};

    SideFeatures own;
    SideFeatures opp;

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        const int dr = DR[d];
        const int dc = DC[d];

        const int own_run = contiguous_after(board, row, col, dr, dc, player);
        const int opp_run = contiguous_after(board, row, col, dr, dc, -player);
        own.soft += own_run * own_run * 105;
        opp.soft += opp_run * opp_run * 112;

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
                accumulate_after_pattern(
                    own, mine_mask | candidate_bit, open_ends, d, rel);
            }
            if (mine_mask == 0) {
                // Defensive leverage: this asks how dangerous the candidate
                // would be if the opponent owned it. Occupying it ourselves
                // kills every such clean opponent road at once.
                accumulate_after_pattern(
                    opp, theirs_mask | candidate_bit, open_ends, d, rel);
            }
        }
    }

    const int own_finish = popcount64(own.finish_mask);
    const int opp_finish = popcount64(opp.finish_mask);
    const int own_four_dirs = __popc(own.four_dirs);
    const int opp_four_dirs = __popc(opp.four_dirs);
    const int own_three_dirs = __popc(own.three_dirs);
    const int opp_three_dirs = __popc(opp.three_dirs);
    const int own_two_dirs = __popc(own.two_dirs);
    const int opp_two_dirs = __popc(opp.two_dirs);
    const int own_activation = popcount64(own.activation_mask);
    const int opp_activation = popcount64(opp.activation_mask);
    const int own_four_empty = popcount64(own.four_empty_mask);
    const int opp_four_empty = popcount64(opp.four_empty_mask);

    const bool own_latent_fork = latent_fork_anchor(
        own.three, own_three_dirs, own_activation);
    const bool opp_latent_fork = latent_fork_anchor(
        opp.three, opp_three_dirs, opp_activation);

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

    // V2 baseline terms are intentionally retained. V2_Pro adds explicit
    // road-leverage terms for latent 2->3 and 3->4 structures, with defense
    // slightly heavier because the observed CNN exploit is a delayed fork.
    const int centre_bonus = 18 - (iabs(row - 9) + iabs(col - 9));
    const int quiet_score =
        own.soft * 4 + opp.soft * 5 +
        own.four * 145000 + opp.four * 160000 +
        own.three * 8500 + opp.three * 9500 +
        own.four_dirs * 0 + opp.four_dirs * 0 +
        own_four_dirs * 18000 + opp_four_dirs * 21000 +
        own_three_dirs * 4500 + opp_three_dirs * 6000 +
        own.two * 700 + opp.two * 900 +
        own_two_dirs * 900 + opp_two_dirs * 1200 +
        own_activation * 4200 + opp_activation * 5800 +
        own_four_empty * 2600 + opp_four_empty * 3400 +
        nearby_own * 125 + nearby_opp * 110 + centre_bonus * 8;

    // Preserve V2's hard tactical ordering first.
    if (own.win > 0)
        return 1000000000 + own.win * 1500000 + quiet_score / 64;
    if (stones_left >= 2 && own_finish > 0)
        return 960000000 + own_finish * 1500000 + quiet_score / 64;
    if (opp.win > 0)
        return 920000000 + opp.win * 1500000 + quiet_score / 64;
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
    if (own.four >= 2 || own_four_dirs >= 2)
        return 112000000 + own.four * 400000 + own_four_dirs * 900000 + quiet_score / 8;
    if (opp.four >= 2 || opp_four_dirs >= 2)
        return 106000000 + opp.four * 400000 + opp_four_dirs * 950000 + quiet_score / 8;

    // New V2_Pro tier: detect/block a latent pair-fork anchor *before* a
    // visible immediate threat exists. Defensive anchors rank slightly higher.
    if (opp_latent_fork)
        return 90000000 + opp.three * 550000 + opp_three_dirs * 1400000 +
            opp_activation * 180000 + quiet_score / 8;
    if (own_latent_fork)
        return 84000000 + own.three * 500000 + own_three_dirs * 1250000 +
            own_activation * 160000 + quiet_score / 8;

    // Softer multi-road precursor tiers keep useful candidate ordering even
    // when the hard latent-fork predicate is not crossed.
    if (opp.three >= 2 || opp_three_dirs >= 2)
        return 43000000 + opp.three * 350000 + opp_three_dirs * 650000 +
            opp_activation * 90000 + quiet_score / 8;
    if (own.three >= 2 || own_three_dirs >= 2)
        return 39000000 + own.three * 320000 + own_three_dirs * 600000 +
            own_activation * 80000 + quiet_score / 8;

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

__global__ void tactical_bot_v2_pro_kernel(
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ current_player,
    const int8_t* __restrict__ stones_left,
    int64_t* __restrict__ actions,
    int batch) {
    const int board_id = blockIdx.x;
    if (board_id >= batch) return;

    __shared__ int8_t shared_board[CELLS];
    __shared__ int shared_score[THREADS];
    __shared__ int shared_action[THREADS];

    const int tid = threadIdx.x;
    const int8_t* src = boards + static_cast<int64_t>(board_id) * CELLS;
    for (int i = tid; i < CELLS; i += blockDim.x)
        shared_board[i] = src[i];
    __syncthreads();

    int local_score = INT_MIN / 4;
    int local_action = -1;
    const int8_t player = current_player[board_id];
    const int8_t left = stones_left[board_id];

    for (int action = tid; action < CELLS; action += blockDim.x) {
        const int s = score_cell_v2_pro(shared_board, action, player, left);
        if (better(s, action, local_score, local_action)) {
            local_score = s;
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

    if (tid == 0)
        actions[board_id] = static_cast<int64_t>(shared_action[0]);
}

}  // namespace

extern "C" cudaError_t launch_tactical_bot_v2_pro_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_bot_v2_pro_kernel<<<batch, THREADS, 0, stream>>>(
        boards, current_player, stones_left, actions, batch);
    return cudaGetLastError();
}
