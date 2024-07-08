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

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from collections import Counter

# import grouping_tags
from grouping_tags import *  

start_time = time.time()


SEED = 2048 
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CERTAINTY_CLASSES = 4       # not tagged, negated, possible, positive 
NUM_SEVERITY_CLASSES = 6        # not tagged, + 5 categories in severity_dict

data_root = "/home/fangm/data/aiiih/projects/fangm/data_extracted_sentence_attr/"
SAVE_DIR = "/lhome/fangm/data/aiiih/projects/fangm/code_output/trained_model_results/"

certainty_num_to_text = {0: 'untagged', 1: 'negated', 2: 'possible', 3: 'positive'}
severity_num_to_text = {0: 'untagged', 1: 'mild', 2: 'moderate', 3: 'mild-moderate', 4: 'moderate-severe', 5: 'severe'}

########################################################################################
# Reading in training results
########################################################################################


def read_full_true_or_predictions_df(true_or_predications, label_type):

    file_path = SAVE_DIR + true_or_predications + "_" + label_type + ".csv"
    df = pd.read_csv(file_path)

    return np.array(df), df.columns


certainty_predictions, certainty_label_names = read_full_true_or_predictions_df("predictions", "certainty_out")
severity_predictions, severity_label_names = read_full_true_or_predictions_df("predictions", "severity_out")
location_predictions, location_label_names = read_full_true_or_predictions_df("predictions", "location_out")


certainty_true, _ = read_full_true_or_predictions_df("true", "certainty")
severity_true, _ = read_full_true_or_predictions_df("true", "severity")
location_true, _ = read_full_true_or_predictions_df("true", "location")


###############################################################################################
###############################################################################################
# Multiclass classification evaluation 
###############################################################################################
###############################################################################################


def get_AUROC_figure_certainty_severity(certainty_or_severity, predictions, true, label_names):
    if certainty_or_severity == "certainty":
        num_classes = NUM_CERTAINTY_CLASSES
        num_to_text_dict = certainty_num_to_text
    else:
        num_classes = NUM_SEVERITY_CLASSES
        num_to_text_dict = severity_num_to_text

    best_cutoff = []
    auroc_values = {label: [] for label in label_names}

    for class_idx in range(num_classes):

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        best_cutoff.append([])

        plt.figure(figsize=(21, 10), dpi=300)
        plt.subplots_adjust(right=0.5)

        for label_idx, label_name in enumerate(label_names):
            y_pred_prob_matrix = predictions[:, label_idx].reshape(-1, num_classes)
            y_true = true[:, label_idx] == class_idx

            fpr[label_idx], tpr[label_idx], thresholds = roc_curve(y_true, y_pred_prob_matrix[:, class_idx])
            roc_auc[label_idx] = auc(fpr[label_idx], tpr[label_idx])

            # Store AUROC value
            auroc_values[label_name].append(roc_auc[label_idx])

            distances = np.sqrt((1 - tpr[label_idx])**2 + (fpr[label_idx])**2)
            optimal_idx = np.argmin(distances)  # Index of the point closest to (1,0)
            best_cutoff[class_idx].append(thresholds[optimal_idx])

            if not np.isnan(roc_auc[label_idx]):
                plt.plot(fpr[label_idx], tpr[label_idx], label=f'{var_rename_dict[label_name]} (AUROC = {roc_auc[label_idx]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) curves for %s: %s' % (certainty_or_severity, num_to_text_dict[class_idx]))
        plt.legend(loc="lower right", bbox_to_anchor=(1.55, 0), borderaxespad=0)

        plt.savefig(SAVE_DIR + certainty_or_severity + "_" + str(class_idx) + num_to_text_dict[class_idx]) 

    best_cutoff = np.array(best_cutoff)

    best_cutoff_dict = {}

    # Iterate over each row in the array
    for row_idx, row in enumerate(best_cutoff.T):
        best_cutoff_dict[label_names[row_idx]] = list(row)


    roc_micro_list = []

    for label_idx, label_name in enumerate(label_names):

        y_pred_prob_stacked_array = None
        y_true_stacked_array = None

        for class_idx in range(num_classes):

            y_pred_prob = predictions[:, label_idx].reshape(-1, num_classes)[:, class_idx]
            y_true = true[:, label_idx] == class_idx

            if sum(y_true) > 0:

                if y_pred_prob_stacked_array is not None:
                    y_pred_prob_stacked_array = np.concatenate((y_pred_prob_stacked_array, y_pred_prob))
                    y_true_stacked_array = np.concatenate((y_true_stacked_array, y_true))
                else:
                    y_pred_prob_stacked_array = y_pred_prob
                    y_true_stacked_array = y_true
            
        # if sum(y_true_stacked_array) == 0:
        #     roc_micro_list.append(np.nan)
        # else:
        # print(y_true_stacked_array.shape, y_pred_prob_stacked_array.shape)
        # print(Counter(y_true_stacked_array))
        roc_micro_list.append(roc_auc_score(y_true_stacked_array, y_pred_prob_stacked_array))
        

    # Convert AUROC values dictionary to pandas DataFrame
    auroc_df = pd.DataFrame(auroc_values, index=[num_to_text_dict[class_idx] for class_idx in range(num_classes)]).transpose()
    auroc_df["Label"] = label_names
    auroc_df["Micro_average"] = roc_micro_list

    if certainty_or_severity == "certainty":
        df = pd.DataFrame(all_tag_names_ordered_with_mention, columns=["Label"])
        roc_scores_df = pd.merge(df, auroc_df, on="Label", how="left")
    else:
        df = pd.DataFrame(all_tag_names_ordered_with_severity, columns=["Label"])
        roc_scores_df = pd.merge(df, auroc_df, on="Label", how="left") 

    roc_scores_df['Label'] = roc_scores_df['Label'].replace(var_rename_dict)
    roc_scores_df.to_csv(SAVE_DIR + "roc_%s.csv" % certainty_or_severity, index=False, na_rep="-1")

    return best_cutoff_dict


best_cutoff_certainty = get_AUROC_figure_certainty_severity(certainty_or_severity = "certainty", predictions = certainty_predictions, true = certainty_true, label_names = certainty_label_names)
best_cutoff_severity = get_AUROC_figure_certainty_severity(certainty_or_severity = "severity", predictions = severity_predictions, true = severity_true, label_names = severity_label_names)

print("Certainty Cutoffs", best_cutoff_certainty)
print("Severity Cutoffs", best_cutoff_certainty)


def get_multiclass_predictions(y_pred_prob_matrix, cutoff_list):
    # How far above threshold was the predicted probability? Choose the greatest one as the predicted class 
    is_above_threshold = (y_pred_prob_matrix >= cutoff_list).astype(int)
    distance_from_threshold = (y_pred_prob_matrix - cutoff_list)

    predicted_labels = []

    for row_is_above_threshold, row_distance_from_threshold in zip(is_above_threshold, distance_from_threshold):

        # none of the probabilities are above threshold, then get the class closest to its threshold 
        if sum(row_is_above_threshold) == 0:
            predicted_labels.append(np.argmin(row_distance_from_threshold))
        # One clear winner above threshold
        elif sum(row_is_above_threshold) == 1:
            predicted_labels.append(np.argmax(row_is_above_threshold))

        # More than 1 class is above threshold, then whichever class is higher above the threshold
        else:
            predicted_labels.append(np.argmax(row_distance_from_threshold))
            
            # Get second highest class if the top predicted is Not Labeled
            # sorted_indices = np.argsort(-row_distance_from_threshold)

            # if sorted_indices[0] == 0:
            #     predicted_labels.append(sorted_indices[1])
            # else:
            #     predicted_labels.append(sorted_indices[0])

    # # Reverse the array to prioritize last occurrence (more extreme attribute) in case of ties
    # flipped_labels = predicted_labels[:, ::-1]
    # last_argmax_index = np.argmax(flipped_labels, axis = 1)
    # original_index = num_classes - 1 - last_argmax_index

    # predicted_labels = original_index

    return np.array(predicted_labels)

def get_f1_table_certainty_severity(certainty_or_severity, predictions, true, label_names, best_cutoff_dict):

    if certainty_or_severity == "certainty":
        num_classes = NUM_CERTAINTY_CLASSES
        num_to_text_dict = certainty_num_to_text
    else:
        num_classes = NUM_SEVERITY_CLASSES
        num_to_text_dict = severity_num_to_text

    f1_scores_list = []

    for label_idx, label_name in enumerate(label_names):

        predictions_for_label = predictions[:, label_idx]
        y_pred_prob_matrix = predictions_for_label.reshape(-1, num_classes)


        # How far above threshold was the predicted probability? Choose the greatest one as the predicted class 
        predicted_labels = get_multiclass_predictions(y_pred_prob_matrix, best_cutoff_dict[label_name])

        true_labels = true[:, label_idx]

        def find_unused_classes(true_labels, num_classes):
            used_classes = set(true_labels)
            all_classes = set(range(num_classes))
            unused_classes = all_classes - used_classes
            return unused_classes
        
        unused_classes = find_unused_classes(true_labels, num_classes)
        non_zero_classes = np.unique(true_labels)

        # Calculate F1 scores only for non-zero classes
        f1_per_class = f1_score(true_labels, predicted_labels, labels=non_zero_classes, average=None)
        # print(len(f1_per_class))

        # Initialize dictionary with NaN values for all classes
        f1_per_class_dict = {f'F1_Class_{int(cls)}': np.nan for cls in range(num_classes)}
        
        # Update F1 scores for non-zero classes using their original indices
        for i, cls in enumerate(non_zero_classes):
            f1_per_class_dict[f'F1_Class_{int(cls)}'] = f1_per_class[i]

        # print(f1_per_class_dict)


        f1_micro = f1_score(true_labels, predicted_labels, labels=non_zero_classes, average='micro')
        f1_macro = f1_score(true_labels, predicted_labels, labels=non_zero_classes, average='macro')
        f1_weighted = f1_score(true_labels, predicted_labels, labels=non_zero_classes, average='weighted')


        # Create DataFrame row for the label
        f1_scores_dict = {
            'Label': label_name,
            'F1_Micro': f1_micro,
            'F1_Macro': f1_macro,
            'F1_Weighted': f1_weighted,
            **f1_per_class_dict  # Unpack F1 per class dictionary
        }

        f1_scores_list.append(f1_scores_dict)


    f1_scores_df = pd.DataFrame.from_dict(f1_scores_list)

    if certainty_or_severity == "certainty":
        df = pd.DataFrame(all_tag_names_ordered_with_mention, columns=["Label"])
        f1_scores_df = pd.merge(df, f1_scores_df, on="Label", how="left")
    else:
        df = pd.DataFrame(all_tag_names_ordered_with_severity, columns=["Label"])
        f1_scores_df = pd.merge(df, f1_scores_df, on="Label", how="left") 


    f1_scores_df['Label'] = f1_scores_df['Label'].replace(var_rename_dict)
    f1_scores_df.to_csv(SAVE_DIR + "F1_%s.csv" % certainty_or_severity, index=False, na_rep="-1")

get_f1_table_certainty_severity(certainty_or_severity = "certainty", predictions = certainty_predictions, true = certainty_true, label_names = certainty_label_names, best_cutoff_dict = best_cutoff_certainty)
get_f1_table_certainty_severity(certainty_or_severity = "severity", predictions = severity_predictions, true = severity_true, label_names = severity_label_names, best_cutoff_dict = best_cutoff_severity)


# ###############################################################################################
# ###############################################################################################
# # Binary classification evaluation 
# ###############################################################################################
# ###############################################################################################

def get_AUROC_figure(y_pred_probabilities, y_true, figure_save_dir, label_names):
    """
    Given predictions, a file save location for the png, and labels
    Will create the AUROC figure and save it 
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    best_cutoff = dict()
    auroc_scores = []

    plt.figure(figsize=(25, 10), dpi=300)
    plt.subplots_adjust(right=0.5)

    for i in range(len(label_names)):
        fpr[i], tpr[i], thresholds = roc_curve(y_true[:, i], y_pred_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Finding the best cutoff
        distances = np.sqrt((1 - tpr[i])**2 + (fpr[i])**2)
        optimal_idx = np.argmin(distances)  # Index of the point closest to (1,0)
        best_cutoff[label_names[i]] = thresholds[optimal_idx]

        plt.plot(fpr[i], tpr[i], label=f'{var_rename_dict[label_names[i]]} (AUROC = {roc_auc[i]:.2f})')

        # Store AUROC score
        auroc_scores.append([label_names[i], roc_auc[i]])

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curves for location')
    plt.legend(loc="lower right", bbox_to_anchor=(1.55, 0), borderaxespad=0)

    plt.savefig(figure_save_dir) 

    # Convert AUROC scores to DataFrame and print it
    roc_scores_df = pd.DataFrame(auroc_scores, columns=['Label', 'AUROC'])

    df = pd.DataFrame(all_tag_names_ordered_with_location, columns=["Label"])
    roc_scores_df = pd.merge(df, roc_scores_df, on="Label", how="left")
    roc_scores_df['Label'] = roc_scores_df['Label'].replace(var_rename_dict)

    roc_scores_df.to_csv(SAVE_DIR + "roc_location.csv", index=False, na_rep="-1")

    return best_cutoff


best_cutoff_dict = get_AUROC_figure(location_predictions, location_true, "location_AUROC.png", location_label_names)
print("Location cutoffs", best_cutoff_dict)



def get_f1_table(y_pred_probabilities, y_true, label_names, table_save_name, best_cutoff_dict):

    f1_scores_list = []

    all_predicted_labels = []

    for label_idx, label_name in enumerate(label_names):

        predictions_for_label = y_pred_probabilities[:, label_idx]

        if best_cutoff_dict:
            predicted_labels = (predictions_for_label >= best_cutoff_dict[label_name]).astype(int)
        else: 
            predicted_labels = (predictions_for_label >= 0.5).astype(int)

        all_predicted_labels.append(predicted_labels)
        true_labels = y_true[:, label_idx]

        if sum(true_labels) == 0:
            f1 = np.nan
        else: 
            f1 = f1_score(true_labels, predicted_labels)

        f1_scores_list.append({
            'Label': label_name,
            'F1': f1,
        })

    f1_scores_df = pd.DataFrame.from_dict(f1_scores_list)

    df = pd.DataFrame(all_tag_names_ordered_with_location, columns=["Label"])
    f1_scores_df = pd.merge(df, f1_scores_df, on="Label", how="left")

    f1_scores_df['Label'] = f1_scores_df['Label'].replace(var_rename_dict)
    f1_scores_df.to_csv(SAVE_DIR + "F1_location_%s.csv" % table_save_name, index=False, na_rep="NA")

    return np.array(all_predicted_labels).T

location_predictions = get_f1_table(location_predictions, location_true, location_label_names, "auroc_cutoffs", best_cutoff_dict)
# get_f1_table(location_predictions, location_true, location_label_names, "0.5_cutoffs", None)


aortic_dilation_location_tags = ['aortic_dilation_location_asc', 'aortic_dilation_location_dsc', 'aortic_dilation_location_root'] 
lv_lge_pattern_tags = [
                        'lv_lge_pattern_pattern_epicardial',
                        'lv_lge_pattern_pattern_mid-myocardial',
                        'lv_lge_pattern_pattern_subendocardial',
                        'lv_lge_pattern_pattern_transmural',
                      ]


is_in_aortic_dilation_location_tags = [True if label in aortic_dilation_location_tags else False for label in location_label_names]
is_in_lv_lge_pattern_tags = [True if label in lv_lge_pattern_tags else False for label in location_label_names]

print("Aortic dilation location micro average", f1_score(location_true[:, is_in_aortic_dilation_location_tags], location_predictions[:, is_in_aortic_dilation_location_tags], average = "micro"))
print("LV LGE pattern micro average",f1_score(location_true[:, is_in_lv_lge_pattern_tags], location_predictions[:, is_in_lv_lge_pattern_tags], average = "micro"))


# ########################################################################################
# ########################################################################################
# ########################################################################################
# # Custom figures 
# ########################################################################################
# ########################################################################################
# ########################################################################################
# ########################################################################################

# def get_AUROC_figure_custom(certainty_or_severity, class_idx, predictions, true, labels_to_show, label_names, fig_title, fig_name):
#     if certainty_or_severity == "certainty":
#         num_classes = NUM_CERTAINTY_CLASSES
#         num_to_text_dict = certainty_num_to_text
#     else:
#         num_classes = NUM_SEVERITY_CLASSES
#         num_to_text_dict = severity_num_to_text

#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()

#     plt.figure(figsize=(10, 10), dpi=300)
#     # plt.subplots_adjust(right=0.5)

#     for label_idx, label_name in enumerate(label_names):
#         if label_name in labels_to_show: 
#             y_pred_prob_matrix = predictions[:, label_idx].reshape(-1, num_classes)
#             y_true = true[:, label_idx] == class_idx

#             fpr[label_idx], tpr[label_idx], thresholds = roc_curve(y_true, y_pred_prob_matrix[:, class_idx])
#             roc_auc[label_idx] = auc(fpr[label_idx], tpr[label_idx])

#             fig_label = var_rename_dict[label_name].replace("Mention", "")
#             fig_label = fig_label.replace("Severity", "")

#             if not np.isnan(roc_auc[label_idx]):
#                 plt.plot(fpr[label_idx], tpr[label_idx], label=f'{fig_label} (AUROC = {roc_auc[label_idx]:.2f})')

#     plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
#     plt.xlim([-0.01, 1.01])
#     plt.ylim([-0.01, 1.01])

#     fontsize = 18

#     # plt.rcParams.update({'font.size': fontsize})  # Adjust 20 to the desired fontsize

#     # params = {'axes.labelsize': fontsize,'axes.titlesize':fontsize, 'text.fontsize': fontsize, 'legend.fontsize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
#     # matplotlib.rcParams.update(params)


#     plt.xlabel('False Positive Rate', fontsize=fontsize)  # Adjust fontsize as needed
#     plt.ylabel('True Positive Rate', fontsize=fontsize)  # Adjust fontsize as needed
#     plt.title('ROC curves for %s' % fig_title, fontsize=fontsize)  # Adjust fontsize as needed
#     plt.legend(loc="lower right", fontsize=14) 



#     plt.savefig(SAVE_DIR + "custom_" + fig_name, bbox_inches='tight')


# labels_to_show = [s + "_mention" for s in valve_tags]
# get_AUROC_figure_custom(certainty_or_severity = "certainty", class_idx = 3, 
#                         predictions = certainty_predictions, true = certainty_true, labels_to_show = labels_to_show, 
#                         label_names = certainty_label_names, fig_title = "Valve Certainty: Positive", fig_name = "positive_valves.png")

# labels_to_show = [s + "_mention" for s in chamber_tags]
# get_AUROC_figure_custom(certainty_or_severity = "certainty", class_idx = 3, 
#                         predictions = certainty_predictions, true = certainty_true, labels_to_show = labels_to_show, 
#                         label_names = certainty_label_names, fig_title = "Chamber Certainty: Positive", fig_name = "positive_chambers.png")

# labels_to_show = [s + "_severity" for s in valve_tags_with_severity]
# get_AUROC_figure_custom(certainty_or_severity = "severity", class_idx = 5, 
#                         predictions = severity_predictions, true = severity_true, labels_to_show = labels_to_show, 
#                         label_names = severity_label_names, fig_title = "Valve Severity: Severe", fig_name = "severe_valve.png")
# get_AUROC_figure_custom(certainty_or_severity = "severity", class_idx = 1, 
#                         predictions = severity_predictions, true = severity_true, labels_to_show = labels_to_show, 
#                         label_names = severity_label_names, fig_title = "Valve Severity: Mild", fig_name = "mild_valve.png")


# labels_to_show = [s + "_severity" for s in chamber_tags_with_severity]
# get_AUROC_figure_custom(certainty_or_severity = "severity", class_idx = 5, 
#                         predictions = severity_predictions, true = severity_true, labels_to_show = labels_to_show, 
#                         label_names = severity_label_names, fig_title = "Chamber Valve Severity: Severe", fig_name = "severe_chambers.png")
# get_AUROC_figure_custom(certainty_or_severity = "severity", class_idx = 1, 
#                         predictions = severity_predictions, true = severity_true, labels_to_show = labels_to_show, 
#                         label_names = severity_label_names, fig_title = "Chamber Severity: Mild", fig_name = "mild_chambers.png")


end_time = time.time()
print("\n\nTotal runtime:", (end_time - start_time) / 60, "min")