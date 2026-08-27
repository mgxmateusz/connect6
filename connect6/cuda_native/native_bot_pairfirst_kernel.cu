#include "native_bot_pair_eval_v4.cuh"

namespace {

using namespace v4_detail;

constexpr int TOP_PAIR_K = 128;
constexpr int LOCAL_KEEP = 4;
constexpr int LOCAL_SLOTS = THREADS * LOCAL_KEEP;

struct PairFeatures {
    int64_t soft_delta;
    int own_win;
    int own_threats;
    int own_fives;
    int own_fours;
    int own_threes;
    unsigned threat_dirs;
    uint64_t finish_lo;
    uint64_t finish_hi;
};

__device__ __forceinline__ bool cheap_pair_better(
    int64_t score,
    int first,
    int second,
    int64_t best_score,
    int best_first,
    int best_second) {
    if (score != best_score) return score > best_score;
    if (first != best_first) return best_first < 0 || first < best_first;
    return best_second < 0 || second < best_second;
}

__device__ __forceinline__ int signed_road_soft(
    int mine,
    int opp,
    unsigned mine_mask,
    unsigned opp_mask) {
    if (opp == 0 && mine > 0 && mine < WIN) {
        return road_soft_value(mine, mine_mask);
    }
    if (mine == 0 && opp > 0 && opp < WIN) {
        return -(road_soft_value(opp, opp_mask) * 11) / 10;
    }
    return 0;
}

__device__ __forceinline__ bool window_contains_cell(
    int sr,
    int sc,
    int dr,
    int dc,
    int cell) {
    const int rr = cell / BOARD;
    const int cc = cell - rr * BOARD;
    #pragma unroll
    for (int k = 0; k < WIN; ++k) {
        if (sr + k * dr == rr && sc + k * dc == cc) return true;
    }
    return false;
}

__device__ __forceinline__ void mark_finish(PairFeatures& f, int cell) {
    const int bit = cell & 127;
    if (bit < 64) f.finish_lo |= uint64_t{1} << bit;
    else f.finish_hi |= uint64_t{1} << (bit - 64);
}

__device__ __forceinline__ void accumulate_pair_window(
    const int8_t* board,
    int sr,
    int sc,
    int dr,
    int dc,
    int first,
    int second,
    int8_t player,
    int direction,
    PairFeatures& f) {
    int base_mine = 0;
    int base_opp = 0;
    int after_mine = 0;
    int after_opp = 0;
    unsigned base_mine_mask = 0;
    unsigned base_opp_mask = 0;
    unsigned after_mine_mask = 0;
    unsigned after_opp_mask = 0;

    #pragma unroll
    for (int k = 0; k < WIN; ++k) {
        const int idx = (sr + k * dr) * BOARD + (sc + k * dc);
        const int8_t v = board[idx];
        if (v == player) {
            ++base_mine;
            base_mine_mask |= 1u << k;
        } else if (v == -player) {
            ++base_opp;
            base_opp_mask |= 1u << k;
        }

        const int8_t av = (idx == first || idx == second) ? player : v;
        if (av == player) {
            ++after_mine;
            after_mine_mask |= 1u << k;
        } else if (av == -player) {
            ++after_opp;
            after_opp_mask |= 1u << k;
        }
    }

    f.soft_delta += static_cast<int64_t>(signed_road_soft(
        after_mine, after_opp, after_mine_mask, after_opp_mask));
    f.soft_delta -= static_cast<int64_t>(signed_road_soft(
        base_mine, base_opp, base_mine_mask, base_opp_mask));

    if (after_opp != 0) return;
    if (after_mine == WIN) {
        ++f.own_win;
        return;
    }
    if (after_mine == 5 || after_mine == 4) {
        ++f.own_threats;
        f.threat_dirs |= 1u << direction;
        if (after_mine == 5) ++f.own_fives;
        else ++f.own_fours;

        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            if ((after_mine_mask & (1u << k)) == 0) {
                mark_finish(f, (sr + k * dr) * BOARD + (sc + k * dc));
            }
        }
    } else if (after_mine == 3) {
        ++f.own_threes;
    }
}

__device__ __forceinline__ int64_t cheap_pair_score(
    const int8_t* board,
    int first,
    int second,
    int8_t player,
    const int16_t* base_opp_ta,
    const int16_t* base_opp_tb,
    int base_opp_threat_count,
    int base_opp_overflow) {
    PairFeatures f{};
    constexpr int DR[4] = {0, 1, 1, 1};
    constexpr int DC[4] = {1, 0, 1, -1};

    const int ar = first / BOARD;
    const int ac = first - ar * BOARD;
    const int br = second / BOARD;
    const int bc = second - br * BOARD;

    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        const int dr = DR[d];
        const int dc = DC[d];
        #pragma unroll
        for (int rel = 0; rel < WIN; ++rel) {
            const int sr = ar - rel * dr;
            const int sc = ac - rel * dc;
            const int er = sr + (WIN - 1) * dr;
            const int ec = sc + (WIN - 1) * dc;
            if (!inside(sr, sc) || !inside(er, ec)) continue;
            accumulate_pair_window(
                board, sr, sc, dr, dc, first, second, player, d, f);
        }
    }

    // Scan roads touched only by the second stone. Any road containing both
    // stones was already processed above, so every changed six-cell road is
    // evaluated exactly once.
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        const int dr = DR[d];
        const int dc = DC[d];
        #pragma unroll
        for (int rel = 0; rel < WIN; ++rel) {
            const int sr = br - rel * dr;
            const int sc = bc - rel * dc;
            const int er = sr + (WIN - 1) * dr;
            const int ec = sc + (WIN - 1) * dc;
            if (!inside(sr, sc) || !inside(er, ec)) continue;
            if (window_contains_cell(sr, sc, dr, dc, first)) continue;
            accumulate_pair_window(
                board, sr, sc, dr, dc, first, second, player, d, f);
        }
    }

    // Existing clean opponent 4/5 roads are immediate threats on its next
    // two-stone turn. A non-winning pair must hit every such road with A or B.
    bool blocks_all_immediate = base_opp_overflow == 0;
    int covered = 0;
    int capped = base_opp_threat_count;
    if (capped > MAX_THREATS) capped = MAX_THREATS;
    if (base_opp_threat_count == 0) blocks_all_immediate = true;
    for (int i = 0; i < capped; ++i) {
        const int ta = static_cast<int>(base_opp_ta[i]);
        const int tb = static_cast<int>(base_opp_tb[i]);
        const bool hit = ta == first || tb == first || ta == second || tb == second;
        covered += hit;
        if (!hit) blocks_all_immediate = false;
    }

    const int finish_diversity = popcount64(f.finish_lo) + popcount64(f.finish_hi);
    const int direction_count = __popc(f.threat_dirs);
    const int centre_first = 18 - (iabs(ar - 9) + iabs(ac - 9));
    const int centre_second = 18 - (iabs(br - 9) + iabs(bc - 9));

    // Lexicographic-style scale: terminal win first; if the opponent already
    // has an immediate clean 4/5, covering all of it is mandatory; afterwards
    // prefer pairs that create several diverse 4/5 threats. Soft road delta is
    // deliberately lower-order and mirrors the exact evaluator's road value.
    int64_t score = 0;
    if (f.own_win > 0) {
        score += INT64_C(8000000000000000);
        score += static_cast<int64_t>(f.own_win) * INT64_C(10000000000000);
    } else if (base_opp_threat_count > 0) {
        if (blocks_all_immediate) {
            score += INT64_C(3000000000000000);
            score += static_cast<int64_t>(covered) * INT64_C(1000000000000);
        } else {
            score -= INT64_C(4000000000000000);
            score += static_cast<int64_t>(covered) * INT64_C(1000000000000);
        }
    }

    score += static_cast<int64_t>(f.own_fives) * INT64_C(220000000000);
    score += static_cast<int64_t>(f.own_fours) * INT64_C(30000000000);
    score += static_cast<int64_t>(f.own_threats) * INT64_C(10000000000);
    score += static_cast<int64_t>(direction_count) * INT64_C(6000000000);
    score += static_cast<int64_t>(finish_diversity) * INT64_C(2500000000);
    score += static_cast<int64_t>(f.own_threes) * INT64_C(120000000);
    score += f.soft_delta * INT64_C(1000);
    score += static_cast<int64_t>(centre_first + centre_second) * INT64_C(1000);
    return score;
}

__global__ void tactical_search_pairfirst_p128_kernel(
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
    __shared__ int empty_actions[CELLS];
    __shared__ int empty_count;

    __shared__ int local_first[LOCAL_SLOTS];
    __shared__ int local_second[LOCAL_SLOTS];
    __shared__ int64_t local_scores[LOCAL_SLOTS];
    __shared__ int selected_first[TOP_PAIR_K];
    __shared__ int selected_second[TOP_PAIR_K];
    __shared__ int64_t selected_scores[TOP_PAIR_K];

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
    __shared__ int base_opp_threat_count;
    __shared__ int base_opp_overflow;

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

    // One-stone turns keep the proven V2 greedy behaviour used by V3/V4/Full.
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

    // Keep the same immediate single-stone win shortcut as the existing bots.
    score_all(board, player, 2, scores);
    __syncthreads();
    if (tid == 0) {
        int a[1];
        int s[1];
        select_top_k_serial<1>(scores, a, s);
        if (a[0] >= 0 && s[0] >= WIN_SCORE) {
            actions[board_id] = static_cast<int64_t>(a[0]);
            pending_second[board_id] = static_cast<int16_t>(-1);
            empty_count = -1;
        } else {
            empty_count = 0;
            for (int cell = 0; cell < CELLS; ++cell) {
                if (board[cell] == 0) empty_actions[empty_count++] = cell;
            }
        }
    }
    __syncthreads();
    if (empty_count < 0) return;

    // Collect the current opponent immediate 4/5 roads once. The cheap pair
    // scorer can then reject pairs that fail to cover them without rescanning
    // all 924 roads per pair.
    if (tid == 0) {
        base_opp_threat_count = 0;
        base_opp_overflow = 0;
    }
    __syncthreads();
    for (int road = tid; road < ROADS; road += blockDim.x) {
        int start, dr, dc;
        decode_road(road, start, dr, dc);
        int mine = 0;
        int opp = 0;
        int empties = 0;
        int empty0 = -1;
        int empty1 = -1;
        const int r = start / BOARD;
        const int c = start - r * BOARD;
        #pragma unroll
        for (int k = 0; k < WIN; ++k) {
            const int idx = (r + k * dr) * BOARD + (c + k * dc);
            const int8_t v = board[idx];
            if (v == player) ++mine;
            else if (v == -player) ++opp;
            else {
                if (empties == 0) empty0 = idx;
                else if (empties == 1) empty1 = idx;
                ++empties;
            }
        }
        if (mine == 0 && (opp == 4 || opp == 5) && empties >= 1 && empties <= 2) {
            const int slot = atomicAdd(&base_opp_threat_count, 1);
            if (slot < MAX_THREATS) {
                opp_ta[slot] = static_cast<int16_t>(empty0);
                opp_tb[slot] = static_cast<int16_t>(empties == 2 ? empty1 : -1);
            } else {
                atomicExch(&base_opp_overflow, 1);
            }
        }
    }
    __syncthreads();

    int64_t best_local_score[LOCAL_KEEP];
    int best_local_first[LOCAL_KEEP];
    int best_local_second[LOCAL_KEEP];
    #pragma unroll
    for (int k = 0; k < LOCAL_KEEP; ++k) {
        best_local_score[k] = INT64_MIN;
        best_local_first[k] = -1;
        best_local_second[k] = -1;
    }

    // Every legal unordered pair is scored directly as a two-stone action.
    // For each first index, threads stride the second index, which distributes
    // C(E,2) work evenly without triangular-index decoding.
    for (int i = 0; i + 1 < empty_count; ++i) {
        const int first = empty_actions[i];
        for (int j = i + 1 + tid; j < empty_count; j += blockDim.x) {
            const int second = empty_actions[j];
            const int64_t pair_score = cheap_pair_score(
                board,
                first,
                second,
                player,
                opp_ta,
                opp_tb,
                base_opp_threat_count,
                base_opp_overflow);

            #pragma unroll
            for (int pos = 0; pos < LOCAL_KEEP; ++pos) {
                if (cheap_pair_better(
                        pair_score,
                        first,
                        second,
                        best_local_score[pos],
                        best_local_first[pos],
                        best_local_second[pos])) {
                    #pragma unroll
                    for (int s = LOCAL_KEEP - 1; s > pos; --s) {
                        best_local_score[s] = best_local_score[s - 1];
                        best_local_first[s] = best_local_first[s - 1];
                        best_local_second[s] = best_local_second[s - 1];
                    }
                    best_local_score[pos] = pair_score;
                    best_local_first[pos] = first;
                    best_local_second[pos] = second;
                    break;
                }
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < LOCAL_KEEP; ++k) {
        const int slot = tid * LOCAL_KEEP + k;
        local_scores[slot] = best_local_score[k];
        local_first[slot] = best_local_first[k];
        local_second[slot] = best_local_second[k];
    }
    __syncthreads();

    // Merge the per-thread shortlists to global TOP128 pair candidates.
    if (tid == 0) {
        for (int q = 0; q < TOP_PAIR_K; ++q) {
            selected_scores[q] = INT64_MIN;
            selected_first[q] = -1;
            selected_second[q] = -1;
        }
        for (int slot = 0; slot < LOCAL_SLOTS; ++slot) {
            const int first = local_first[slot];
            const int second = local_second[slot];
            if (first < 0 || second < 0) continue;
            const int64_t pair_score = local_scores[slot];
            for (int pos = 0; pos < TOP_PAIR_K; ++pos) {
                if (cheap_pair_better(
                        pair_score,
                        first,
                        second,
                        selected_scores[pos],
                        selected_first[pos],
                        selected_second[pos])) {
                    for (int s = TOP_PAIR_K - 1; s > pos; --s) {
                        selected_scores[s] = selected_scores[s - 1];
                        selected_first[s] = selected_first[s - 1];
                        selected_second[s] = selected_second[s - 1];
                    }
                    selected_scores[pos] = pair_score;
                    selected_first[pos] = first;
                    selected_second[pos] = second;
                    break;
                }
            }
        }
    }
    __syncthreads();

    int best_value = INVALID_SCORE;
    int64_t best_order = INT64_MIN;
    int best_first = -1;
    int best_second = -1;

    // Only the best cheap pair candidates pay for the full 924-road evaluator.
    #pragma unroll 1
    for (int q = 0; q < TOP_PAIR_K; ++q) {
        const int first = selected_first[q];
        const int second = selected_second[q];
        if (first < 0 || second < 0) continue;

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
            if (best_first < 0 ||
                pair_better(
                    value,
                    selected_scores[q],
                    first,
                    second,
                    best_value,
                    best_order,
                    best_first,
                    best_second)) {
                best_value = value;
                best_order = selected_scores[q];
                best_first = first;
                best_second = second;
            }
            board[first] = 0;
            board[second] = 0;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (best_first < 0) {
            best_first = empty_count > 0 ? empty_actions[0] : -1;
            best_second = empty_count > 1 ? empty_actions[1] : -1;
        }
        actions[board_id] = static_cast<int64_t>(best_first);
        pending_second[board_id] = static_cast<int16_t>(best_second);
    }
}

}  // namespace

extern "C" cudaError_t launch_tactical_bot_pairfirst_p128_cuda(
    const int8_t* boards,
    const int8_t* current_player,
    const int8_t* stones_left,
    int16_t* pending_second,
    int64_t* actions,
    int batch,
    cudaStream_t stream) {
    if (batch <= 0) return cudaSuccess;
    tactical_search_pairfirst_p128_kernel<<<batch, v4_detail::THREADS, 0, stream>>>(
        boards, current_player, stones_left, pending_second, actions, batch);
    return cudaGetLastError();
}
