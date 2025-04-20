import numpy as np
import argparse

def generate_matrix(rows, cols, kind='random', dtype=np.int16):
    if kind == 'identity':
        assert rows == cols, "Identity matrix must be square"
        return np.eye(rows, dtype=dtype)
    elif kind == 'ones':
        return np.ones((rows, cols), dtype=dtype)
    elif kind == 'zeros':
        return np.zeros((rows, cols), dtype=dtype)
    else:
        info = np.iinfo(dtype)
        return np.random.randint(info.min, info.max + 1, size=(rows, cols), dtype=dtype)

def write_plain_text(values, per_line, filename):
    with open(filename, "w") as f:
        for i in range(0, len(values), per_line):
            line = values[i:i+per_line]
            f.write(" ".join(str(x) for x in line) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["int8", "int16"], required=True)
    parser.add_argument("--per-line", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--mode", choices=["random", "identity", "zeros", "ones"], default="random")
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--output", default="output.npy")

    args = parser.parse_args()

    dtype = np.int8 if args.type == "int8" else np.int16
    M, K, N = args.M, args.K, args.N

    if args.mode == "identity":
        assert M == K, "Identity matrix A must be square (M=K)"
        A = np.eye(M, dtype=dtype)
        info = np.iinfo(dtype)
        B = np.random.randint(info.min, info.max + 1, size=(K, N), dtype=dtype)
    else:
        A = generate_matrix(M, K, kind=args.mode, dtype=dtype)
        B = generate_matrix(K, N, kind=args.mode, dtype=dtype)

    # Compute and save C = A @ B
    C = (A.astype(np.int32) @ B.astype(np.int32)).astype(dtype)
    np.save(args.output, C)
    print(f"✅ Saved ground-truth matrix to {args.output} with shape {C.shape}")

    AB = np.concatenate([A.flatten(), B.flatten()])
    write_plain_text(AB, args.per_line, args.input)
    print(f"✅ Saved input stream to {args.input} ({len(AB)} values, {args.per_line} per line)")

if __name__ == "__main__":
    main()
