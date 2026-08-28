#pragma once

#include "native_bot_pair_eval_v4.cuh"

namespace v2pro_detail {

using namespace v4_detail;

struct SideFeaturesPro {
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

__device__ __forceinline__ void accumulate_after_pattern_pro(
    SideFeaturesPro& f,
    unsigned after_mask,
    int open_ends,
    int direction,
    int rel) {
    const int after = popcount6(after_mask);
    f.soft += pattern_value_v2(after_mask, open_ends);

    if (after >= WIN) {
        ++f.win;
        return;
    }

    if (after == 5) {
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            if ((after_mask & (1u << k)) == 0)
                f.finish_mask |= relative_threat_bit(direction, k - rel);
        }
        return;
    }

    if (after == 4) {
        ++f.four;
        f.four_dirs |= 1u << direction;
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            if ((after_mask & (1u << k)) == 0)
                f.four_empty_mask |= relative_threat_bit(direction, k - rel);
        }
        return;
    }

    if (after == 3) {
        ++f.three;
        f.three_dirs |= 1u << direction;
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            if ((after_mask & (1u << k)) == 0)
                f.activation_mask |= relative_threat_bit(direction, k - rel);
        }
        return;
    }

    if (after == 2) {
        ++f.two;
        f.two_dirs |= 1u << direction;
    }
}

__device__ __forceinline__ bool latent_fork_anchor_pro(
    int three_windows,
    int three_dirs,
    int activation_cells) {
    return
        (three_windows >= 3 && activation_cells >= 4) ||
        (three_dirs >= 2 && three_windows >= 2 && activation_cells >= 4) ||
        (three_windows >= 2 && activation_cells >= 7);
}

// Reusable copy of the validated V2Pro one-cell heuristic. Search variants use
// this only as candidate ordering/prior; their pair/state evaluators stay
// unchanged so baseline-vs-Pro comparisons isolate the scorer.
__device__ __forceinline__ int score_cell_v2_pro_shared(
    const int8_t* board,
    int action,
    int8_t player,
    int8_t stones_left) {
    if (action < 0 || action >= CELLS || board[action] != 0)
        return INVALID_SCORE;

    const int row = action / BOARD;
    const int col = action - row * BOARD;
    constexpr int DR[4] = {0, 1, 1, 1};
    constexpr int DC[4] = {1, 0, 1, -1};

    SideFeaturesPro own;
    SideFeaturesPro opp;

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
                accumulate_after_pattern_pro(
                    own, mine_mask | candidate_bit, open_ends, d, rel);
            }
            if (mine_mask == 0) {
                accumulate_after_pattern_pro(
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

    const bool own_latent_fork = latent_fork_anchor_pro(
        own.three, own_three_dirs, own_activation);
    const bool opp_latent_fork = latent_fork_anchor_pro(
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

    const int centre_bonus = 18 - (iabs(row - 9) + iabs(col - 9));
    const int quiet_score =
        own.soft * 4 + opp.soft * 5 +
        own.four * 145000 + opp.four * 160000 +
        own.three * 8500 + opp.three * 9500 +
        own_four_dirs * 18000 + opp_four_dirs * 21000 +
        own_three_dirs * 4500 + opp_three_dirs * 6000 +
        own.two * 700 + opp.two * 900 +
        own_two_dirs * 900 + opp_two_dirs * 1200 +
        own_activation * 4200 + opp_activation * 5800 +
        own_four_empty * 2600 + opp_four_empty * 3400 +
        nearby_own * 125 + nearby_opp * 110 + centre_bonus * 8;

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

    if (opp_latent_fork)
        return 90000000 + opp.three * 550000 + opp_three_dirs * 1400000 +
            opp_activation * 180000 + quiet_score / 8;
    if (own_latent_fork)
        return 84000000 + own.three * 500000 + own_three_dirs * 1250000 +
            own_activation * 160000 + quiet_score / 8;

    if (opp.three >= 2 || opp_three_dirs >= 2)
        return 43000000 + opp.three * 350000 + opp_three_dirs * 650000 +
            opp_activation * 90000 + quiet_score / 8;
    if (own.three >= 2 || own_three_dirs >= 2)
        return 39000000 + own.three * 320000 + own_three_dirs * 600000 +
            own_activation * 80000 + quiet_score / 8;

    return quiet_score;
}

__device__ __forceinline__ void score_all_pro(
    const int8_t* board,
    int8_t player,
    int8_t stones_left,
    int* scores) {
    const int tid = threadIdx.x;
    for (int action = tid; action < CELLS; action += blockDim.x) {
        scores[action] = score_cell_v2_pro_shared(
            board, action, player, stones_left);
    }
}

}  // namespace v2pro_detail
