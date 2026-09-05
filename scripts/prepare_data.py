import argparse

from custom_llm.data.spam import download_spam_data, prepare_spam_splits


def main():
    parser = argparse.ArgumentParser(description="Download and split the balanced SMS spam dataset.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    tsv = download_spam_data(args.data_dir)
    paths = prepare_spam_splits(tsv, args.data_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
