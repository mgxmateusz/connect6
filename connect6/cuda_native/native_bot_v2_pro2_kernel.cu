#include "native_bot_v2_pro_kernel.cu"

namespace {

struct PairForceFeatures {
    int best_four_windows = 0;
    int best_five_windows = 0;
    int best_win_windows = 0;
    int best_run = 0;
    int forcing_partners = 0;
    unsigned forcing_dirs = 0;
    int soft = 0;
};

__device__ __forceinline__ PairForceFeatures pair_force_features_v2_pro2(
    const int8_t* board,
    int action,
    int8_t stone) {
    PairForceFeatures out;
    if (action < 0 || action >= CELLS || board[action] != 0)
        return out;

    const int row = action / BOARD;
    const int col = action - row * BOARD;
    constexpr int DR2[4] = {0, 1, 1, 1};
    constexpr int DC2[4] = {1, 0, 1, -1};

    // V2_Pro2 is still a one-cell scorer. For each candidate X, inspect only
    // possible partner Y cells sharing one of the four Connect6 lines through X.
    // This is deliberately much cheaper than enumerating every legal board pair.
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        const int dr = DR2[d];
        const int dc = DC2[d];
        #pragma unroll
        for (int off = -5; off <= 5; ++off) {
            if (off == 0) continue;
            const int pr = row + off * dr;
            const int pc = col + off * dc;
            if (!inside(pr, pc)) continue;
            const int partner = pr * BOARD + pc;
            if (board[partner] != 0) continue;

            int four_windows = 0;
            int five_windows = 0;
            int win_windows = 0;
            int best_run = 0;
            int partner_soft = 0;

            // Only six-windows on the shared X-Y line can contain both stones.
            // Scan windows containing X and keep those that also contain Y.
            #pragma unroll
            for (int rel = 0; rel < WIN; ++rel) {
                const int partner_rel = rel + off;
                if (partner_rel < 0 || partner_rel >= WIN) continue;

                const int sr = row - rel * dr;
                const int sc = col - rel * dc;
                const int er = sr + (WIN - 1) * dr;
                const int ec = sc + (WIN - 1) * dc;
                if (!inside(sr, sc) || !inside(er, ec)) continue;

                unsigned own_mask = 0;
                unsigned block_mask = 0;
                #pragma unroll
                for (int k = 0; k < WIN; ++k) {
                    const int rr = sr + k * dr;
                    const int cc = sc + k * dc;
                    const int8_t v = board[rr * BOARD + cc];
                    own_mask |= static_cast<unsigned>(v == stone) << k;
                    block_mask |= static_cast<unsigned>(v == -stone) << k;
                }
                if (block_mask != 0) continue;

                const unsigned pair_mask =
                    own_mask | (1u << rel) | (1u << partner_rel);
                const int after = popcount6(pair_mask);
                const int run = max_run6(pair_mask);
                if (run > best_run) best_run = run;

                int open_ends = 0;
                const int br = sr - dr;
                const int bc = sc - dc;
                const int ar = er + dr;
                const int ac = ec + dc;
                if (inside(br, bc) && board[br * BOARD + bc] == 0) ++open_ends;
                if (inside(ar, ac) && board[ar * BOARD + ac] == 0) ++open_ends;

                if (after >= 6) {
                    ++win_windows;
                    partner_soft += 250000;
                } else if (after == 5) {
                    ++five_windows;
                    partner_soft += 30000 + run * 4500 + open_ends * 2500;
                } else if (after == 4) {
                    ++four_windows;
                    // Contiguous/open fours are the forcing ladder observed in
                    // CNN 2354: three overlapping 4-in-6 windows can consume both
                    // defensive stones on the following turn.
                    partner_soft += 3500 + run * run * 700 + open_ends * 500;
                    if (run >= 4) partner_soft += 12000;
                } else if (after == 3) {
                    partner_soft += 500 + run * run * 80;
                }
            }

            if (four_windows > out.best_four_windows)
                out.best_four_windows = four_windows;
            if (five_windows > out.best_five_windows)
                out.best_five_windows = five_windows;
            if (win_windows > out.best_win_windows)
                out.best_win_windows = win_windows;
            if (best_run > out.best_run)
                out.best_run = best_run;
            if (partner_soft > out.soft)
                out.soft = partner_soft;

            const bool forcing =
                win_windows > 0 || five_windows > 0 ||
                (four_windows >= 3 && best_run >= 4) ||
                (four_windows >= 2 && best_run >= 4);
            if (forcing) {
                ++out.forcing_partners;
                out.forcing_dirs |= 1u << d;
            }
        }
    }
    return out;
}

__device__ __forceinline__ int pair_force_bonus_v2_pro2(
    const PairForceFeatures& f,
    bool defensive) {
    const int dirs = __popc(f.forcing_dirs);

    // A pair that wins immediately next turn is the strongest new signal.
    if (f.best_win_windows > 0)
        return (defensive ? 150000000 : 120000000) +
            f.best_win_windows * 3000000 + f.forcing_partners * 250000;

    // Pair-created fives are highly forcing, but still below V2/V2Pro's visible
    // current-turn 4/5 hard tiers.
    if (f.best_five_windows > 0)
        return (defensive ? 52000000 : 42000000) +
            f.best_five_windows * 2200000 + dirs * 1200000 +
            f.forcing_partners * 180000;

    // This is the key Pro2 addition: X participates in a partner pair X+Y that
    // converts an otherwise quiet two-stone road directly into a forcing four.
    if (f.best_four_windows >= 3 && f.best_run >= 4)
        return (defensive ? 24000000 : 19000000) +
            f.best_four_windows * 1000000 + dirs * 1400000 +
            f.forcing_partners * 220000 + f.soft / 16;

    if (f.best_four_windows >= 2 && f.best_run >= 4)
        return (defensive ? 12500000 : 9500000) +
            f.best_four_windows * 700000 + dirs * 850000 +
            f.forcing_partners * 140000 + f.soft / 20;

    if (f.best_four_windows > 0)
        return (defensive ? 3800000 : 2800000) +
            f.best_four_windows * 300000 + f.soft / 32;

    return f.soft / (defensive ? 24 : 32);
}

__device__ __forceinline__ int score_cell_v2_pro2(
    const int8_t* board,
    int action,
    int8_t player,
    int8_t stones_left) {
    const int base = score_cell_v2_pro(board, action, player, stones_left);
    if (base <= INT_MIN / 8) return base;

    // Do not disturb V2Pro's visible immediate tactical ordering. Pro2 is meant
    // to improve the quiet/latent ordering that precedes those hard tiers.
    if (base >= 260000000)
        return base;

    const PairForceFeatures opp_pair =
        pair_force_features_v2_pro2(board, action, -player);
    const PairForceFeatures own_pair =
        pair_force_features_v2_pro2(board, action, player);

    const int opp_bonus = pair_force_bonus_v2_pro2(opp_pair, true);
    int own_bonus = pair_force_bonus_v2_pro2(own_pair, false);
    if (stones_left <= 1)
        own_bonus /= 3;  // no second own stone remains in the current turn

    // Defense is intentionally dominant: occupying X now removes X from every
    // dangerous opponent X+Y pair. Own forcing-pair value remains a tie-breaker
    // and attack signal when two stones are still available.
    int score = base + opp_bonus + own_bonus;
    if (score > 255000000) score = 255000000;
    return score;
}

__global__ void tactical_bot_v2_pro2_kernel(
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
        const int s = score_cell_v2_pro2(shared_board, action, player, left);
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

extern "C" cudaError_t launch_tactical_bot_v2_pro2_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_bot_v2_pro2_kernel<<<batch, THREADS, 0, stream>>>(
        boards, current_player, stones_left, actions, batch);
    return cudaGetLastError();
}
