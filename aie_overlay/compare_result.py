import numpy as np
import argparse
import pandas as pd

def parse_aie_output(path, M, N):
    values = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("T ") or line.strip() == "TLAST":
                continue  # skip timestamp and TLAST
            try:
                values += list(map(int, line.strip().split()))
            except ValueError:
                print(f"⚠️ Skipping non-numeric line: {line.strip()}")
    return np.array(values, dtype=np.int8).reshape(M, N)


def load_reference(path):
    return np.load(path)

def compare_matrices(hw, ref):
    equal = np.array_equal(hw, ref)
    diff = np.abs(hw.astype(np.int16) - ref.astype(np.int16))
    return {
        "match": equal,
        "max_error": int(np.max(diff)),
        "mismatch_count": int(np.sum(diff != 0)),
    }

def extract_sim_time(path):
    with open(path, "r") as f:
        lines = [l for l in f if l.startswith("T ")]
        t_start = lines[0]
        t_end = lines[-1]

        start_val, start_unit = int(t_start.split()[1]), t_start.split()[2]
        end_val, end_unit = int(t_end.split()[1]), t_end.split()[2]

        # assert start_unit == end_unit
        if start_unit == 'ns':
            start_val *= 1000
            start_unit = 'ps'
        if end_unit == 'ns':
            end_val *= 1000
            end_unit = 'ps'
        
        return end_val - start_val, start_unit

def main():
    parser = argparse.ArgumentParser(description="Compare AIE output with reference matrix")
    parser.add_argument("--per-line", type=int, required=True, help="Values per line in output file")
    parser.add_argument("--M", type=int, required=True, help="Rows of A / C")
    parser.add_argument("--K", type=int, required=True, help="Cols of A / Rows of B")
    parser.add_argument("--N", type=int, required=True, help="Cols of B / C")
    parser.add_argument("--hw-output", type=str, required=True, help="Path to hardware output .txt file")
    parser.add_argument("--ref-output", type=str, required=True, help="Path to reference output .npy file")

    args = parser.parse_args()

    hw = parse_aie_output(args.hw_output, args.M, args.N)
    ref = load_reference(args.ref_output)
    result = compare_matrices(hw, ref)
    runtime_val, runtime_unit = extract_sim_time(args.hw_output)

    print("✅ Match" if result["match"] else "❌ Mismatch")
    print(f"Max Absolute Error: {result['max_error']}")
    print(f"Mismatch Count: {result['mismatch_count']}")
    print(f"Total Runtime: {runtime_val} {runtime_unit}")
    print(f"Matrix Shape: {args.M}x{args.K} x {args.K}x{args.N}")
    print(f"Values per Line: {args.per_line}")

if __name__ == "__main__":
    main()
