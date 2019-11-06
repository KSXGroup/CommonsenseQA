import argparse
import json
from model import QAModel


def main(args):
    with open(args.config_file, "r") as f:
        config = json.loads(f.read())
    encoder_config = config["encoder"]
    decoder_config = config["decoder"]
    model = QAModel(encoder_config, decoder_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/QAconfig.json", type=str)
    parser.add_argument("--is_training", default=True, type=bool)
    args = parser.parse_args()
    main(args)
