import os

from train import config
from prediction import run_validation_cases


def main():

    prediction_dir = os.path.abspath("data/prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file="ich_segmentation_model_incomplete.h5",
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)

if __name__ == "__main__":
    main()

