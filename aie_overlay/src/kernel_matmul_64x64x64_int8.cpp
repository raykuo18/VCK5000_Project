#include <aie_api/aie.hpp>

// Matrix dimensions
constexpr int M = 64;
constexpr int K = 64;
constexpr int N = 64;

// Tiling dimensions
constexpr int TM = 4;
constexpr int TK = 16;
constexpr int TN = 8;

constexpr int TILE_A = TM * TK;
constexpr int TILE_B = TK * TN;
constexpr int TILE_C = TM * TN;

// Compute C = A x B, where A and B are 64x64 int8 matrices, packed in one input window
__attribute__((noinline))
void kernel_matmul_64x64x64_int8(
    input_window<int8>* __restrict in_win,
    output_window<int8>* __restrict out_win) {

  for (int i = 0; i < M; i += TM) {
    for (int j = 0; j < N; j += TN) {
      aie::mmul<TM, TK, TN, int8, int8> acc;

      for (int k = 0; k < K; k += TK) {
        // -----------------------------
        // Read A tile: A[i:i+4, k:k+16]
        // -----------------------------
        // Move to start of tile A[i, k]
        window_incr(in_win, i * K + k);

        aie::vector<int8, TILE_A> a_tile;
        for (int ii = 0; ii < TM; ++ii) {
          for (int kk = 0; kk < TK; ++kk) {
            a_tile[ii * TK + kk] = window_readincr(in_win);
          }
          if (ii != TM - 1 && K > TK) window_incr(in_win, K - TK);  // stop at end of tile row
        }

        // Rewind back to start of A using bottom-right index
        window_decr(in_win, (i + TM - 1) * K + k + TK - 1);

        // -----------------------------
        // Read B tile: B[k:k+16, j:j+8]
        // -----------------------------
        window_incr(in_win, M * K + k * N + j);

        aie::vector<int8, TILE_B> b_tile;
        for (int kk = 0; kk < TK; ++kk) {
          for (int jj = 0; jj < TN; ++jj) {
            b_tile[kk * TN + jj] = window_readincr(in_win);
          }
          if (kk != TK - 1 && N > TN) window_incr(in_win, N - TN);  // stop at end of tile row
        }

        // Rewind back to start of A using bottom-right index
        window_decr(in_win, M * K + (k + TK - 1) * N + j + TN - 1);

        // -----------------------------
        // Multiply-accumulate
        // -----------------------------
        if (k == 0)
          acc.mul(a_tile, b_tile);
        else
          acc.mac(a_tile, b_tile);
      }

      // -----------------------------
      // Write C tile
      // -----------------------------
      aie::vector<int8, TILE_C> c_tile = acc.to_vector<int8>(0);

      // Move to the start of tile C[i, j] in output
      window_incr(out_win, i * N + j);

      for (int ii = 0; ii < TM; ++ii) {
        for (int jj = 0; jj < TN; ++jj) {
          window_writeincr(out_win, c_tile[ii * TN + jj]);
        }
        if (ii != TM - 1 && N > TN) window_incr(out_win, N - TN);  // skip rest of the row
      }

      // Rewind to start of tile
      window_decr(out_win, TM * TN + (TM - 1) * (N - TN) + i * N + j);
    }
  }
} // Explanation:
  // - The input window stores A followed by B
  // - The pointer is initially at the start of A
  // - We move to A[i,k] and rewind after tile use
  // - Then we move to B[k,j] and rewind after tile use
  // - Output now correctly written to C[i,j] by navigating and rewinding the output window
