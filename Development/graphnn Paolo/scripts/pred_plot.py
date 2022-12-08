import argparse
import pandas as pd

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Plot uncertainty for the predictions",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "--pred_file", type=str, default=None, help="Predictions",
    )

    return parser.parse_args(arg_list)


def main():
    args = get_arguments()
    
    predictions = pd.read_csv(args['pred_file'])



if __name__ == "__main__":
    main()