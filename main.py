import argparse
from partitioning.swamy.base import BaseSwamyPartitioner as BSP
from partitioning.swamy.optimal import OptimalPartitioner
from partitioning.swamy.heuristic import HeuristicPartitioner
from partitioning.metis import MetisPartitioner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Districting")
    parser.add_argument("-s", "--state", default="RI", help="State to district")
    parser.add_argument("-m", "--method", default="swamy_h", help="Partitioner to use")
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=BSP.ALPHA_DEFAULT,
        help="Alpha value for optimal partitioner",
    )
    parser.add_argument("--map", type=bool, default=True, help="Show map of districts")
    parser.add_argument("--verbose", type=bool, default=True, help="Print solution")
    parser.add_argument(
        "-t",
        "--slack_type",
        type=str,
        default=BSP.SLACK_DEFAULT,
        help="Starting slack value for fixed and dynamic solution types",
    )
    parser.add_argument(
        "-v",
        "--slack_value",
        type=float,
        default=BSP.SLACK_VALUE_DEFAULT,
        help="Starting slack value for fixed and dynamic solution types",
    )
    parser.add_argument("-g", "--gap", type=float, default=0.0, help="Gap for optimal partitioner")
    parser.add_argument(
        "-l", "--size_limit", type=int, default=15, help="Size limit for heuristic partitioner"
    )

    args = parser.parse_args()
    p_method = args.method

    if p_method == "metis":
        dp = MetisPartitioner(args.state)
    elif p_method in ["swamy_o", "optimal"]:
        dp = OptimalPartitioner.from_files(
            args.state, args.alpha, slack_type=args.slack_type, slack_value=args.slack_value
        )
        dp.optimize(gap=args.gap)
    elif p_method in ["swamy_h", "heuristic"]:
        dp = HeuristicPartitioner.from_files(
            args.state, args.alpha, args.slack_type, args.slack_value, 1000
        )
        dp.optimize(gap=args.gap, size_limit=args.size_limit)

    if args.verbose:
        dp.print_solution()
    if args.map:
        dp.show_map()
