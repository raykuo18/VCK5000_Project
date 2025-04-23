#include <aie_api/aie.hpp>

__attribute__((noinline)) void kernel_matmul_4x4x4_int16_stream(
    input_stream<int16>* __restrict in_stream,
    output_stream<int16>* __restrict out_stream) {

  constexpr int TILE_SIZE = 16; // 4x4 matrix = 16 elements

  aie::vector<int16, TILE_SIZE> A;
  aie::vector<int16, TILE_SIZE> B;

  // Read A matrix (16 elements)
  for (int i = 0; i < TILE_SIZE; ++i)
    A[i] = readincr(in_stream);

  // Read B matrix (16 elements)
  for (int i = 0; i < TILE_SIZE; ++i)
    B[i] = readincr(in_stream);

  // Compute C = A x B
  aie::mmul<4, 4, 4, int16, int16> M;
  M.mul(A, B);
  aie::vector<int16, TILE_SIZE> C = M.to_vector<int16>(0);

  // Write C matrix (16 elements)
  for (int i = 0; i < TILE_SIZE; ++i)
    writeincr(out_stream, C[i]);
}
