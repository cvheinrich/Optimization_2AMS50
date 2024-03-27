import os
import time
import argparse

from settings.local_settings import DATA_PATH
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
    parser.add_argument("--no_map", action="store_false", help="Show map of districts")
    parser.add_argument("--silent", action="store_false", help="Print solution")
    parser.add_argument("--save", action="store_true", help="Save log and map to file")
    parser.add_argument(
        "-t",
        "--slack_type",
        type=str,
        default=BSP.SLACK_DEFAULT,
        help=f"Slack type. One of {', '.join(BSP.SLACK_TYPES)}",
    )
    parser.add_argument(
        "-v",
        "--slack_value",
        type=float,
        default=BSP.SLACK_VALUE_DEFAULT,
        help=f'Slack value. Used for fixed slack type, and when policy is not "{BSP.POLICY_KEEP}"',
    )
    parser.add_argument("-g", "--gap", type=float, default=0.0, help="Gap for optimal partitioner")
    parser.add_argument(
        "-l", "--size_limit", type=int, default=15, help="Size limit for heuristic partitioner"
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default=BSP.POLICY_DEFAULT,
        help=f"Policy for large nodes. One of {', '.join(BSP.POLICIES)}",
    )
    parser.add_argument(
        "-u",
        "--update_avg",
        action="store_true",
        help="Whether to update average population after applying policy",
    )

    args = parser.parse_args()
    p_method = args.method

    run_properties = {}
    model = None

    start_time = time.time()

    if p_method == "metis":
        raise NotImplementedError("Metis partitioner not implemented")
        model = MetisPartitioner(args.state)
    elif p_method in ["swamy_o", "optimal", "swamy_h", "heuristic"]:
        run_properties = {
            "gap": args.gap,
            "size_limit": args.size_limit,
            "policy": args.policy,
            "update_avg": args.update_avg,
        }

        if p_method in ["swamy_o", "optimal"]:
            Warning(
                "Some options do not work for optimal partitioner. Using heuristic partitioner with large size limit instead."
            )
            run_properties["size_limit"] = 100000

        model = HeuristicPartitioner.from_files(
            args.state, args.alpha, args.slack_type, args.slack_value, 1000
        )
        model.optimize(**run_properties)

    duration = time.time() - start_time

    if not args.silent:
        print(f"Run properties: {run_properties}\n")
        print("Solution:")
        model.print_solution()
        print(f"\nDuration: {duration:.2f} seconds")

    if args.save:
        file_location = os.path.join(DATA_PATH, "..", "figures", args.state)
        os.makedirs(file_location, exist_ok=True)
        file_title = "__".join(
            [str(v) for v in run_properties.values()]
            + [str(v) for v in model.get_model_properties().values()]
        ).replace(".", "_")
        with open(os.path.join(file_location, f"{file_title}.log"), "w") as f:
            f.write(f"Arguments: {args}\n")
            f.write(f"\nDuration: {duration:.2f} seconds\n")
            f.write(f"\n{model.get_model_title()}")

    model.create_map(run_properties, args.no_map, args.save)
