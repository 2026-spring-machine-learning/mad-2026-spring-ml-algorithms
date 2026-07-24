import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "diabetes": PROJECT_ROOT / "classification" / "analyze_diabetes.py",
    "cherry-tree": PROJECT_ROOT / "linear_regression" / "analyze_cherry_tree.py",
    "bonuses": PROJECT_ROOT / "linear_regression" / "pay_bonuses_non_modular.py",
}


def run_script(script_key: str) -> int:
    script_path = SCRIPTS[script_key]
    result = subprocess.run(
        [sys.executable, script_path.name],
        cwd=script_path.parent,
        check=False,
    )
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run project analysis scripts.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="list",
        choices=["list", *SCRIPTS.keys()],
        help="Script to run, or 'list' to show available options.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.target == "list":
        print("Available scripts:")
        for key, script_path in SCRIPTS.items():
            print(f"- {key}: {script_path.relative_to(PROJECT_ROOT)}")
        return 0

    return run_script(args.target)


if __name__ == "__main__":
    raise SystemExit(main())
