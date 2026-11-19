#!/usr/bin/env python3
"""Print diagnostic scores for Gray code and Log PE positional encodings."""

import argparse
import json
import math

from SeqSNN.module.comrt import (
    build_gray_code_table,
    diagnostic_score_from_table,
    diagnostic_score_log_pe,
)

HEADER = f"  {'encoding':>12}  {'bits':>5}  {'f1':>6}  {'f2':>6}  {'f3':>6}  {'f4':>6}  {'quality':>8}  {'bit_fac':>7}  {'final':>8}"
ROW    = "  {name:>12}  {bits:>5}  {f1:>6.4f}  {f2:>6.4f}  {f3:>6.4f}  {f4:>6.4f}  {quality:>8.4f}  {bit_fac:>7.4f}  {final:>8.4f}"


def print_result(name: str, r: dict) -> None:
    print(ROW.format(
        name=name,
        bits=r["total_bits"],
        f1=r["f1_monotonicity"],
        f2=r["f2_tail_resolution"],
        f3=r["f3_local_resolution"],
        f4=r["f4_consistency"],
        quality=r["quality_score"],
        bit_fac=r["bit_factor"],
        final=r["final_score"],
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic scores for Gray code and Log PE positional encodings."
    )
    parser.add_argument("--L", type=int, default=168, help="Sequence length (default: 168)")
    parser.add_argument("--num_bits", type=int, default=None,
                        help="Bits for Gray code (default: ceil(log2(L)))")
    parser.add_argument("--json", action="store_true",
                        help="Also dump full JSON for each encoding")
    args = parser.parse_args()

    num_bits = args.num_bits or math.ceil(math.log2(max(args.L, 2)))

    gray_table  = build_gray_code_table(args.L, num_bits=num_bits)
    gray_result = diagnostic_score_from_table(gray_table, args.L, encoding_name="gray_code")
    log_result  = diagnostic_score_log_pe(args.L)

    print(f"\n  L={args.L}  gray_bits={num_bits}  log_bits={log_result['total_bits']}")
    print(HEADER)
    print("  " + "-" * (len(HEADER) - 2))
    print_result("gray_code", gray_result)
    print_result("log_pe",    log_result)
    print()

    if args.json:
        print("── gray_code ──")
        print(json.dumps(gray_result, indent=2))
        print("── log_pe ──")
        print(json.dumps(log_result, indent=2))


if __name__ == "__main__":
    main()
