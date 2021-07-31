import os
import yaml
import librosa
import argparse
import numpy as np
import pandas as pd

from utils import audio_to_mel


def main(config):

    val_size = config["val_size"]

    data = pd.read_csv(os.path.join(config["data_dir"], config["csv"]))

    labeled = data[~data["number"].isna()]
    labeled = labeled.sample(frac=1).reset_index(drop=True)  # shuffle the data. just in case

    # generate spectrograms
    print("=> Converting audio to spectrograms...")
    for index, row in labeled.iterrows():
        path = os.path.join(config["data_dir"], row['path'])
        audio, sr = librosa.load(path, sr=config["spec_params"]["sr"])
        melspec = audio_to_mel(audio, config["spec_params"])
        spec_name = f"{'.'.join(path.split('.')[:-1])}.npy"
        np.save(spec_name, melspec)  # save spectrogram
        labeled.loc[index, "spect_path"] = spec_name  # save path to spectrogram
    print("...Done!")

    # split the data
    val = labeled[:int(len(labeled)*val_size)]
    train = labeled[int(len(labeled)*val_size):]

    # save data files
    val.to_csv(os.path.join(config["data_dir"], config["val_data"]))
    train.to_csv(os.path.join(config["data_dir"], config["train_data"]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.conf))
    main(config)
