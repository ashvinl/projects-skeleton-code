import os
import pandas as pd

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    leafdata = pd.read_csv("/content/train.csv")
    leafdata1 = leafdata[:len(leafdata)//2]
    leafdata2 = leafdata[len(leafdata)//2:]


    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset(leafdata1['image_id'], leafdata1['label'])
    val_dataset = StartingDataset(leafdata2['image_id'], leafdata2['label'])
    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
