import argparse
from train import train_base, train_ensemble, train_quantile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['base', 'ensemble', 'quantile'])
    parser.add_argument('--csv_path', default='data/example_dataset.csv')
    parser.add_argument('--x_columns', required=True)
    parser.add_argument('--target_column', required=True)
    args = parser.parse_args()

    if args.mode == 'base':
        from train.train_base import train as run
    elif args.mode == 'ensemble':
        from train.train_ensemble import train as run
    else:
        from train.train_quantile import train as run
    run(args)


if __name__ == '__main__':
    main()
