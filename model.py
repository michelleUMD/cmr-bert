import lightning.pytorch as pl
from transformers import (
    AdamW,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import BertLMPredictionHead
import torch
from torch import nn
import config
from ray.tune.integration.pytorch_lightning import TuneReportCallback

NUM_CERTAINTY_CLASSES = 4       # not tagged, negated, possible, positive 
NUM_SEVERITY_CLASSES = 6        # not tagged, + 5 categories in severity_dict

class SequencePredModel(pl.LightningModule):
    def __init__(
        self, config, 
        n_batches=None, 
        # n_epochs=None, lr=None, 
        certainty_label_names = None, severity_label_names = None, location_label_names = None, 
        **kwargs
    ):
        super().__init__()

        ## Params
        self.n_batches = n_batches
        self.n_epochs = config["n_epochs"]
        self.lr = config["lr"]
        self.num_hidden_nodes = config["num_hidden_nodes"]
        self.validation_step_outputs = []

        # self.save_hyperparameters()

        self.certainty_label_names = certainty_label_names
        self.severity_label_names = severity_label_names
        self.location_label_names = location_label_names

        self.config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        ## Encoder
        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", return_dict=True
        )
        # Unfreeze layers
        self.bert_layer_num = sum(1 for _ in self.bert.named_parameters())
        self.num_unfreeze_layer = self.bert_layer_num
        self.ratio_unfreeze_layer = 0.0
        # if kwargs:
        for key, value in config.items():
            if key == "unfreeze" and isinstance(value, float):
                assert (
                    value >= 0.0 and value <= 1.0
                ), "ValueError: value must be a ratio between 0.0 and 1.0"
                self.ratio_unfreeze_layer = value
        if self.ratio_unfreeze_layer > 0.0:
            self.num_unfreeze_layer = int(
                self.bert_layer_num * self.ratio_unfreeze_layer
            )
        for param in list(self.bert.parameters())[: -self.num_unfreeze_layer]:
            param.requires_grad = False

        # self.lm_head = BertLMPredictionHead(self.config)


        self.certainty_hidden = nn.Linear(self.bert.config.hidden_size, self.num_hidden_nodes)
        self.certainty_final = nn.Linear(self.num_hidden_nodes, NUM_CERTAINTY_CLASSES * len(self.certainty_label_names))

        self.severity_hidden = nn.Linear(self.bert.config.hidden_size, self.num_hidden_nodes)
        self.severity_final = nn.Linear(self.num_hidden_nodes, NUM_SEVERITY_CLASSES * len(self.severity_label_names)) 

        self.location_hidden = nn.Linear(self.bert.config.hidden_size, self.num_hidden_nodes)
        self.location_final = nn.Linear(self.num_hidden_nodes, len(self.location_label_names))

        self.certainty_softmax= nn.Softmax(dim = 1)
        self.severity_softmax = nn.Softmax(dim = 1)
        self.location_sigmoid = nn.Sigmoid()

        # MAY NEED TO DO A SIGMOID BEFORE 

        print("Model Initialized!")

        ## Losses
        self.certainty_loss = nn.CrossEntropyLoss()
        self.severity_loss = nn.CrossEntropyLoss()
        self.location_loss = nn.BCEWithLogitsLoss()

        # Save a list of losses to plot later on
        self.certainty_losses = []
        self.severity_losses = []
        self.location_losses = []
        self.certainty_losses_epoch_end = []
        self.severity_losses_epoch_end = []
        self.location_losses_epoch_end = []

        # self.multitask_loss = multitask_loss()

        ## Logs
        self.num_batches = 0
        self.train_loss, self.val_loss = 0, 0
        self.train_loss_certainty, self.val_loss_certainty = 0, 0
        self.train_loss_severity, self.val_loss_severity = 0, 0
        self.training_step_outputs, self.validation_step_outputs = [], []


    def forward(self, input_ids, certainty_label, severity_label, location_label, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # embs_certainty = self.certainty_sigmoid(self.certainty_final(self.certainty_hidden(output.pooler_output)))
        # embs_severity = self.severity_sigmoid(self.severity_final(self.severity_hidden(output.pooler_output)))
        # embs_location = self.location_sigmoid(self.location_final(self.location_hidden(output.pooler_output)))

        embs_certainty = self.certainty_final(self.certainty_hidden(output.pooler_output))
        embs_severity = self.severity_final(self.severity_hidden(output.pooler_output))
        embs_location = self.location_final(self.location_hidden(output.pooler_output))

        # Reshape so columns match number of classes in multiclass classification 
        embs_certainty = embs_certainty.view(-1, NUM_CERTAINTY_CLASSES, len(self.certainty_label_names))
        embs_severity = embs_severity.view(-1, NUM_SEVERITY_CLASSES, len(self.severity_label_names))

        # Convert labels to type long
        certainty_label = certainty_label.long()
        severity_label = severity_label.long()

        certainty_loss = self.certainty_loss(embs_certainty, certainty_label)
        severity_loss = self.severity_loss(embs_severity, severity_label)
        location_loss = self.location_loss(embs_location, location_label)

        return output.pooler_output, embs_certainty, embs_severity, embs_location, certainty_loss, severity_loss, location_loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        certainty_label = batch["certainty_label"]
        severity_label = batch["severity_label"]
        location_label = batch["location_label"]

        out, embs_certainty, embs_severity, embs_location, certainty_loss, severity_loss, location_loss = self(input_ids, certainty_label, severity_label, location_label, attention_mask)

        # if batch_idx % 10 == 0: 
        self.log("certainty train_loss", certainty_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("severity train_loss", severity_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("location train_loss", location_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("total train loss", certainty_loss + severity_loss + location_loss, prog_bar=True, logger=True, sync_dist=True)

        return certainty_loss + severity_loss + location_loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        certainty_label = batch["certainty_label"]
        severity_label = batch["severity_label"]
        location_label = batch["location_label"]

        # raise Exception("Shape" + str(location_label.shape))

        out, embs_certainty, embs_severity, embs_location, certainty_loss, severity_loss, location_loss = self(input_ids, certainty_label, severity_label, location_label, attention_mask)

        # if batch_idx % 10 == 0: 
        self.log("certainty val_loss", certainty_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("severity val_loss", severity_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("location val_loss", location_loss, prog_bar=True, logger=True, sync_dist=True)
        total_loss = certainty_loss + severity_loss + location_loss
        self.log("total val loss", total_loss, prog_bar=True, logger=True, sync_dist=True)

        self.certainty_losses.append(certainty_loss)
        self.severity_losses.append(severity_loss)
        self.location_losses.append(location_loss)
        self.validation_step_outputs.append(total_loss)

        return {
            "certainty_loss": certainty_loss, 
            "severity_loss": severity_loss,
            "location_loss": location_loss,
            "total_loss": total_loss
        }

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()

        self.certainty_losses_epoch_end.append(torch.stack(self.certainty_losses).mean().item())
        self.severity_losses_epoch_end.append(torch.stack(self.severity_losses).mean().item())
        self.location_losses_epoch_end.append(torch.stack(self.location_losses).mean().item())

        self.log("epoch_end_val_loss", avg_loss)

        self.certainty_losses.clear()
        self.severity_losses.clear()
        self.location_losses.clear()
        self.validation_step_outputs.clear()

        # self.log("ptl/val_accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        certainty_label = batch["certainty_label"]
        severity_label = batch["severity_label"]
        location_label = batch["location_label"]

        out, embs_certainty, embs_severity, embs_location, certainty_loss, severity_loss, location_loss = self(input_ids, certainty_label, severity_label, location_label, attention_mask)

        # if batch_idx % 10 == 0: 
        self.log("certainty test_loss", certainty_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("severity test_loss", severity_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("location test_loss", location_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("total test loss", certainty_loss + severity_loss + location_loss, prog_bar=True, logger=True, sync_dist=True)

        return {
            "certainty_loss": certainty_loss, 
            "severity_loss": severity_loss,
            "location_loss": location_loss
        }

    def predict_step(self, batch, attention_mask):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        certainty_label = batch["certainty_label"]
        severity_label = batch["severity_label"]
        location_label = batch["location_label"]

        out, embs_certainty, embs_severity, embs_location, certainty_loss, severity_loss, location_loss = self(input_ids, certainty_label, severity_label, location_label, attention_mask)

        # get prediction probabilities for each class 
        embs_certainty_softmax = self.certainty_softmax(embs_certainty)
        embs_severity_softmax = self.severity_softmax(embs_severity)
        embs_location_sigmoid = self.location_sigmoid(embs_location)

        return {
            "certainty_out": embs_certainty_softmax, 
            "severity_out": embs_severity_softmax,
            "location_out": embs_location_sigmoid
        }



    def configure_optimizers(self):

        self.trainable_params = [
            param for param in self.parameters() if param.requires_grad
        ]
        optimizer = AdamW(self.trainable_params, lr=self.lr)

        # Scheduler
        warmup_steps = self.n_batches // 3
        total_steps = self.n_batches * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        return [optimizer], [scheduler]