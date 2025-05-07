import logging
import os
import sys

from datasets import load_from_disk


def convert_ds(
        input_dir: str,
        output_dir: str,
):

    ds = load_from_disk(input_dir)

    n_lines = 0
    with open(os.path.join(output_dir, "train_data.txt"), "w", encoding="utf-8") as f:
        for x in ds:
            for line in x["text"].splitlines():
                if line != "":
                    n_lines += 1
                    f.write(line + "\n")

    logging.info(f"Saved {n_lines} lines to {output_dir}/train_data.txt")



if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(convert_ds)
