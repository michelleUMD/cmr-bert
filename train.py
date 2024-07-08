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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import json



start_time = time.time()


SEED = 2048 
torch.manual_seed(SEED)
np.random.seed(SEED)

# Number of combos of hyperparameters to try in grid search
# NUM_TRIALS = 10

data_root = "/home/fangm/data/aiiih/projects/fangm/data_extracted_sentence_attr/"
SAVE_DIR = "/lhome/fangm/data/aiiih/projects/fangm/code_output/trained_model_results/"


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

print(file_tag_names)

train_df = build_dataset("Train", file_tag_names)
test_df = build_dataset("Test", file_tag_names)
# make sure no extra columns in the test_df 
assert (train_df.columns == test_df.columns).all(), "ERROR: TRAIN AND TEST COLUMNS DO NOT MATCH"

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


# Read the dictionary back from the file
with open(SAVE_DIR + "best_config.json", 'r') as file:
    best_config = json.load(file)


data_module = SequencePredDataModule(train_df, test_df, test_df, tokenizer, best_config["batch_size"])
data_module.setup()

certainty_label_names = data_module.certainty_label_names
severity_label_names = data_module.severity_label_names
location_label_names = data_module.location_label_names


# best_config = {'num_hidden_nodes': 150, 'lr': 2.434198820973321e-05, 'batch_size': 64, 'n_epochs': 20, 'unfreeze': 1}

print("Training using hyperparameters", best_config)


model = SequencePredModel(config=best_config,
    n_batches=len(data_module.train_dataset) / best_config["batch_size"],
    certainty_label_names = certainty_label_names,
    severity_label_names = severity_label_names,
    location_label_names = location_label_names,
)



accelerator = "gpu"
devices = 1  # number of gpus
precision = "16-mixed"

# Initialize Trainer
trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    strategy="ddp",
    logger=TensorBoardLogger(SAVE_DIR + "logs", name="Contrastive learning"),
    max_epochs=best_config["n_epochs"],
    # min_epochs=best_config["n_epochs"] - 1,
    precision="16-mixed",
    callbacks=[EarlyStopping(monitor="epoch_end_val_loss", min_delta=1e-5, patience=1, mode="min")],
)

trainer.fit(model, data_module)

# trainer.save_checkpoint(SAVE_DIR + 'trained_model.ckpt')

trainer.validate(model, data_module)
predictions = trainer.predict(model, data_module)

def save_full_predictions_df(predictions, label_type, col_names):
    predictions_array = []

    if label_type == "certainty_out" or label_type == "severity_out":

        individual_matrices = []

        # Flatten each batch and append it to the list
        for batch in predictions:
            batch_size, num_classes, num_features = batch[label_type].size()
            flattened_batch = batch[label_type].reshape(batch_size * num_classes, num_features)
            individual_matrices.append(flattened_batch)

        # Concatenate all flattened batches along axis 0
        predictions_array = torch.cat(individual_matrices, dim=0)


    else: 
        individual_matrices = [batch_dict[label_type] for batch_dict in predictions]
        big_matrix = torch.cat(individual_matrices, dim=0)
        predictions_array = big_matrix.numpy()

    predictions_df = pd.DataFrame(predictions_array, columns=col_names)
    predictions_df.to_csv(SAVE_DIR + "predictions_" + label_type + ".csv", index=False)
    return predictions_df


certainty_predictions = save_full_predictions_df(predictions, "certainty_out", certainty_label_names)
severity_predictions = save_full_predictions_df(predictions, "severity_out", severity_label_names)
location_predictions = save_full_predictions_df(predictions, "location_out", location_label_names)

def save_full_true_df(data_module, label_type):
    y_true_df = data_module.get_test_true(label_type)
    y_true_df.to_csv(SAVE_DIR + "true_" + label_type + ".csv", index=False)


save_full_true_df(data_module, "location")
save_full_true_df(data_module, "severity")
save_full_true_df(data_module, "certainty")


end_time = time.time()
print("\n\nTotal runtime:", (end_time - start_time) / 60, "min")