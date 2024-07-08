import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
import pandas as pd
import copy
from ast import literal_eval
from sklearn.model_selection import train_test_split
import random
import numpy as np


def col_requires_multilabel(col):
    if 'location' in col:
        return True 
    if '_type_' in col or '_oth_type_' in col or '_pattern_pattern' in col:
        return True
    if col in ["no_valve_abnorm", "no_aortic_abnorm", "no_ventricular_abnorm"]:
        return True

    return False 


class SequencePredDataset(Dataset):
    def __init__(self, data):
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]
        text = row.text

        certainty_label = row[[col for col in row.index if 'mention' in col]].tolist()
        severity_label = row[[col for col in row.index if 'severity' in col]].tolist()
        location_label = row[[col for col in row.index if col_requires_multilabel(col)]].tolist()

        # raise Exception (location_label)

        return {
            "text": text, 
            "certainty_label": certainty_label,
            "severity_label": severity_label,
            "location_label": location_label}


def collate_function(batch, tokenizer):
    texts = [item["text"] for item in batch]

    certainty_labels = [item["certainty_label"] for item in batch]
    severity_labels = [item["severity_label"] for item in batch]
    location_labels = [item["location_label"] for item in batch]


    df = pd.DataFrame(columns=["input_ids", "attention_mask"])

    for text in texts:
        encoded_text = tokenizer.encode_plus(
            text,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        row = pd.DataFrame(
            {
                "input_ids": encoded_text["input_ids"].tolist(),
                "attention_mask": encoded_text["attention_mask"].tolist(),
                # "certainty_label": certainty_label.tolist(),
                # "severity_label": severity_label.tolist()
            }
        )
        df = pd.concat([df, row])

    input_ids_tsr = list(map(lambda x: torch.tensor(x), df["input_ids"]))
    padded_input_ids = pad_sequence(input_ids_tsr, padding_value=tokenizer.pad_token_id)
    padded_input_ids = torch.transpose(padded_input_ids, 0, 1)

    attention_mask_tsr = list(map(lambda x: torch.tensor(x), df["attention_mask"]))
    padded_attention_mask = pad_sequence(attention_mask_tsr, padding_value=0)
    padded_attention_mask = torch.transpose(padded_attention_mask, 0, 1)

    certainty_label = torch.tensor(certainty_labels, dtype=torch.float32)
    severity_label = torch.tensor(severity_labels, dtype=torch.float32)
    location_label = torch.tensor(location_labels, dtype=torch.float32)

    # raise Exception (location_label)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "certainty_label": certainty_label,
        "severity_label": severity_label,
        "location_label": location_label
    }


def create_dataloader(dataset, tokenizer, shuffle, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers = 1,
        collate_fn=lambda batch: collate_function(batch, tokenizer),
    )


class SequencePredDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, batch_size):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer

        self.batch_size = batch_size

        self.certainty_label_names = [col for col in train_df.columns if "mention" in col]
        self.severity_label_names = [col for col in train_df.columns if "severity" in col]
        self.location_label_names = [col for col in train_df.columns if col_requires_multilabel(col)]
        # self.location_label_names_val = [col for col in val_df.columns if "location" in col]

    def setup(self, stage=None):
        self.train_dataset = SequencePredDataset(self.train_df)
        self.val_dataset = SequencePredDataset(self.val_df)
        self.test_dataset = SequencePredDataset(self.test_df)
    
    def get_test_true(self, y_data_type = None):
        if y_data_type == "certainty":
            return self.test_df[self.certainty_label_names]
        elif y_data_type == "severity":
            return self.test_df[self.severity_label_names] 
        elif y_data_type == "location":
            return self.test_df[self.location_label_names] 
        else:
            raise Exception("get_test_true's y_data_type must be certainty, severity, or location")

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.tokenizer, shuffle=True, batch_size = self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.tokenizer, shuffle=False, batch_size = self.batch_size)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.tokenizer, shuffle=False, batch_size = self.batch_size)

    def predict_dataloader(self):
        return create_dataloader(self.test_dataset, self.tokenizer, shuffle=False, batch_size = self.batch_size)

