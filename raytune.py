import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

import config
from model import SequencePredModel
from dataset import SequencePredDataModule

import pandas as pd

import numpy as np
import time
import os
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import matplotlib.pyplot as plt
import json


start_time = time.time()


SEED = 2048 
torch.manual_seed(SEED)
np.random.seed(SEED)

# Number of combos of hyperparameters to try in grid search
NUM_TRIALS = 20

data_root = "/home/fangm/data/aiiih/projects/fangm/data_extracted_sentence_attr/"
SAVE_DIR = "/lhome/fangm/data/aiiih/projects/fangm/code_output/"


########################################################################################
# Helper functions 
########################################################################################

def build_dataset(train_or_test, feature_list):
   """
      Creates a dataset after specifying "Train" or "Test" and what features to include 
      Returns a Dataset object 
   """
   first = True 
   for feature in feature_list:
        temp = pd.read_csv(data_root + "%s/data_for_%s.csv" %(train_or_test, feature))
      
        if first:
            first = False 
            data = temp 
        else: 
            data = pd.merge(data, temp, on='sentence', how='left')

   # Standardize dataframe so columns match 
   data.rename(columns={'sentence': 'text'}, inplace=True)

   return data


def get_labels(directory):
    """
    Get list of all labels being predicted in multilabel classification 
    """
    file_tag_names = []
    labels = [] # file tag names with each of their attributes 

    for filename in os.listdir(directory):
        # if filename == "all_text.csv":
        #     continue 

        tag_name = filename.replace("data_for_", "").replace(".csv", "")
        file_tag_names.append(tag_name)
        labels.extend(pd.read_csv(directory + filename).columns[1:])

    print("%d tags being analyzed:" % len(labels), labels)

    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    return file_tag_names, labels, id2label, label2id


seed_everything(SEED, workers=True)

directory = data_root + "Train/"

file_tag_names, labels, id2label, label2id = get_labels(directory)


train_df = build_dataset("Train", file_tag_names)
train2_df, val_df = train_test_split(train_df, test_size=0.2, random_state = SEED)

test_df = build_dataset("Test", file_tag_names)


# Scale the columns that require ordinal regression 
# scaled_features = {}

# ordinal_cols = [col for col in train2_df.columns if "mention" in col or "severity" in col]
# for ordinal_col in ordinal_cols:

#     mean, std = train2_df[ordinal_col].mean(), train2_df[ordinal_col].std()

#     train2_df.loc[:, ordinal_col] = (train2_df[ordinal_col] - mean)/std
#     val_df.loc[:, ordinal_col] = (val_df[ordinal_col] - mean)/std
#     test_df.loc[:, ordinal_col] = (test_df[ordinal_col] - mean)/std

#     scaled_features[ordinal_col] = [mean, std]


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

data_module = SequencePredDataModule(train2_df, val_df, test_df, tokenizer, 10)
data_module.setup()

certainty_label_names = data_module.certainty_label_names
severity_label_names = data_module.severity_label_names
location_label_names = data_module.location_label_names


print("Number of training data:", len(data_module.train_dataset))
print("Number of validation data:", len(data_module.val_dataset))
print("Number of test data:", len(data_module.test_dataset))

print("Certainty labels", data_module.certainty_label_names)
print("Severity labels", data_module.severity_label_names)
print("Location labels", data_module.location_label_names)



accelerator = "gpu"
devices = 1  # number of gpus
precision = "16-mixed"

def train_model(config, data_dir = None, data_module = None): 

    data_module.batch_size = config["batch_size"]

    # Initialize Model
    model = SequencePredModel(config=config,
        n_batches=len(data_module.train_dataset) / config["batch_size"],
        # n_epochs=config["num_epochs"],
        # lr=config["lr"],
        certainty_label_names = certainty_label_names,
        severity_label_names = severity_label_names,
        location_label_names = location_label_names,
        # unfreeze = 0.3
    )

    metrics = {"loss": "total val loss"}

    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    #             # EarlyStopping(monitor="certainty val_loss", min_delta=1e-3, patience=1, mode="min")]


    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy="ddp",
        logger=TensorBoardLogger(SAVE_DIR + "logs", name="Contrastive learning"),
        max_epochs=config["n_epochs"],
        precision="16-mixed",
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)

config = {
    "num_hidden_nodes": tune.choice([64, 128, 150, 180]),
    "lr": tune.loguniform(1e-6, 1e-4),
    "batch_size": tune.choice([8, 10, 16]),
    "n_epochs": tune.choice([12, 15, 18, 20]),
    "unfreeze": tune.choice([0, 0.3, 0.5, 0.8, 1]) 
}

trainable = tune.with_parameters(train_model, data_module=data_module)

                    
analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": 1
    },
    metric="loss",
    mode="min",
    config=config,
    num_samples=NUM_TRIALS,
    name=SAVE_DIR + "hyperparameter_search")  # Change the logging directory here


best_config = analysis.best_config

print(best_config)


# best_config = {'num_hidden_nodes': 150, 'lr': 2.434198820973321e-05, 'batch_size': 64, 'n_epochs': 20, 'unfreeze': 1}

with open(SAVE_DIR + "trained_model_results/best_config.json", 'w') as file:
    json.dump(best_config, file)


end_time = time.time()
print("\n\nTotal runtime:", (end_time - start_time) / 60, "min")