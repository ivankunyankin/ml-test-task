import os
import yaml
import torch
import random
import librosa
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from models import QuartzNet
from utils import audio_to_mel


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class Tester:

    def __init__(self, config):

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = QuartzNet().to(self.device)

        weights = config["weights"]
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        print("=> Loaded checkpoint")

    def test(self, path_to_csv):

        self.model.eval()

        data = pd.read_csv(path_to_csv)

        with torch.no_grad():
            for index, row in data.iterrows():
                path = os.path.join(self.config["data_dir"], row['path'])
                audio, sr = librosa.load(path, sr=self.config["spec_params"]["sr"])
                melspec = audio_to_mel(audio, self.config["spec_params"])

                melspec = melspec.unsqueeze(0).to(self.device)
                output = self.model(melspec)

                args = torch.argmax(output, dim=1)

                decodes = []
                for item in args:
                    pred = "".join(str(int(i)) for i in item)
                    decodes.append(pred)

                data.loc[int(index), "number"] = decodes[0].lstrip('0')  # save predict
        new_path = f"{'.'.join(path_to_csv.split('.')[:-1])}_pred.csv"
        data.to_csv(new_path)


def main():

    parser = ArgumentParser()
    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')
    parser.add_argument('--path', help='Path to the csv with testing data paths')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.conf))
    path = args.path
    tester = Tester(config)

    print("=> Initialised tester")
    print("=> Testing...")
    tester.test(path)
    print("...Done!")


if __name__ == "__main__":
    main()
