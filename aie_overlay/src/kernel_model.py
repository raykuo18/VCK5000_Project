import numpy as np

# Matrix dimensions
M, K, N = 64, 64, 64
TM, TK, TN = 4, 16, 8

def simulate_kernel_matmul_stream(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Simulate the tiled matmul kernel behavior in Python (int8 version)"""
    C = np.zeros((M, N), dtype=np.int8)

    for i in range(0, M, TM):
        for j in range(0, N, TN):
            acc = np.zeros((TM, TN), dtype=np.int32)  # accumulate in higher precision

            for k in range(0, K, TK):
                a_tile = A[i:i+TM, k:k+TK].astype(np.int32)
                b_tile = B[k:k+TK, j:j+TN].astype(np.int32)

                acc += a_tile @ b_tile  # TMxTK @ TKxTN = TMxTN

            # Clip to int8 after accumulation
            C[i:i+TM, j:j+TN] = np.clip(acc, -128, 127).astype(np.int8)

    return C

# Example usage: generate random A and B
np.random.seed(0)
A = np.random.randint(-128, 128, size=(M, K), dtype=np.int8)
B = np.random.randint(-128, 128, size=(K, N), dtype=np.int8)

C_simulated = simulate_kernel_matmul_stream(A, B)

import ace_tools as tools; tools.display_dataframe_to_user(name="Simulated Output Matrix C", dataframe=pd.DataFrame(C_simulated))
