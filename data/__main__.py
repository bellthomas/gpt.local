import argparse
from . import DatasetManager

if __name__ == "__main__":
    # Handle arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--collection', default="herodotus", help='')
    parser.add_argument('--validation-percentage', default=0.05, help='')
    args = parser.parse_args()

    try:
        DatasetManager(
            dataset=args.collection,
            test_set_ratio=args.validation_percentage
        ).prepare()
    except KeyboardInterrupt:
        exit(0)