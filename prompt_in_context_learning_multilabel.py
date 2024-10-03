from model import *
from argparse import ArgumentParser
import pandas as pd
from simcse import SimCSE
import json
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import random
# from sklearn.metrics import classification_report
from itertools import chain
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelClassificationReport:
    def __init__(self, y_true, y_pred):
        """
        Initialize with the true and predicted labels.

        :param y_true: List of lists, where each sublist contains the true labels for a sample
        :param y_pred: List of lists, where each sublist contains the predicted labels for a sample
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.all_labels = sorted(set(chain.from_iterable(y_true + y_pred)))
        self.y_true_bin = self._binarize_labels(y_true)
        self.y_pred_bin = self._binarize_labels(y_pred)
        self.report = self._generate_report()

    def _binarize_labels(self, labels):
        """
        Binarize the labels for multi-label classification.

        :param labels: List of lists, where each sublist contains the labels for a sample
        :return: Binarized numpy array of shape (n_samples, n_labels)
        """
        return np.array([[1 if label in sample else 0 for label in self.all_labels] for sample in labels])

    def _generate_report(self):
        """
        Generate the classification report for each label and aggregate the results.

        :return: Dictionary containing the aggregated classification report
        """
        reports = []
        for i, label in enumerate(self.all_labels):
            report = classification_report(self.y_true_bin[:, i], self.y_pred_bin[:, i], output_dict=True, zero_division=0)
            reports.append((label, report))
        final_report = {}
        for label, report in reports:
            for metric in report.keys():
                if metric not in final_report:
                    final_report[metric] = {}
                for key in report[metric].keys():
                    if key not in final_report[metric]:
                        final_report[metric][key] = []
                    final_report[metric][key].append(report[metric][key])

        for metric in final_report:
            for key in final_report[metric]:
                final_report[metric][key] = np.mean(final_report[metric][key])

        return final_report

    def print_report(self):
        """
        Print the aggregated classification report.
        """
        for metric in self.report:
            print(f"{metric}:")
            for key in self.report[metric]:
                print(f"  {key}: {self.report[metric][key]:.2f}")


random.seed(42)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_json(text_dir):
    abstract_id = []
    sentences = []
    labels = []
    with open(text_dir, encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            abstract_id.append(data["abstract_id"])
            sentences.append(data["sentences"])
            labels.append(data["labels"])
    pubmed_df = pd.DataFrame({"abstract_id": abstract_id, "text": sentences, "labels": labels})
    return pubmed_df

def construct_dataset_train(paragraph, true_labels, model_retrieve, retrieve_df_train, dataset):
    if dataset == "biorc800":
        dataset_items = []
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as methods or resutls, and the last several sentences might play a rhetorical role as conclusion. "

        whole_paragraph = " ".join(paragraph)

        results = model_retrieve.search(whole_paragraph, top_k=300, threshold=0.1)
        illustrations = "<Start>"
        sample_count = 0
        if config.sample_count == 0:
            illustrations = ""
        else:
            for i in results:
                if len(i[0].split(" ")) > 150:
                    continue
                elif whole_paragraph[:100].lower() in i[0].lower():
                    continue
                else:
                    retrieved_text_single = re.escape(i[0])
                    retrieve_in_context = retrieve_df_train[
                        retrieve_df_train['paragraph_train'].str.contains(retrieved_text_single)]
                    if len(retrieve_in_context) > 0:
                        for key, i in retrieve_in_context.iterrows():
                            illustrations += "The paragraph is \"" + " ".join(
                                i['sentences']) + "\". Select from rhetorical labels including background, objective, method, result and conclusion"
                            for sentence, label in zip(i['sentences'], i['labels']):
                                illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + ", ".join([i.lower().rstrip("s") for i in label]) + ">"
                            illustrations += " <End>\n"

                        sample_count += 1
                    if sample_count >= config.sample_count:
                        break
        dataset_items.append({"samples": illustrations, "sentences": paragraph, "labels": [[label.lower().rstrip("s") for label in sublist] for sublist in true_labels]})
        return dataset_items

def construct_dataset_single(data, model_retrieve, retrieve_df_train, dataset):
    dataset_prepare = []
    paragraphs = data["text"].to_list()
    abstract_ids = data["abstract_id"].to_list()
    labels = data["labels"].to_list()
    for abstract_id, paragraph, label in zip(abstract_ids, paragraphs, labels):
        dataset_prepare.extend(construct_dataset_train(paragraph, label, model_retrieve, retrieve_df_train, dataset))
    return dataset_prepare

def map_to_normal(s1, dataset):
    # for candidates in ["background", "objective", "method", "result", "conclusion", "other"]:
    labels = []
    if dataset == "biorc800":
        for candidates in ["background", "objective", "method", "result", "conclusion", "other"]:
            if candidates in s1.lower() or candidates == s1:
                labels.append(candidates)
        if len(labels) == 0:
            return ["background"]
        else:
            return labels
def load_data(config, model_retrieve, mode, retrieve_df_train):
    data = load_json("data_llm/" + config.dataset + "/" + mode + ".jsonl")
    data = data
    df = pd.DataFrame(data=construct_dataset_single(data, model_retrieve, retrieve_df_train, config.dataset))
    return df

def save_div(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0


def evaluation(true_labels, predicted_labels, classes):

    precision_micro = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
    recall_micro = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
    f1_score_micro = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)

    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_score_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1_score_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    classification_report_ = classification_report(true_labels, predicted_labels, target_names=classes)

    return precision_micro, recall_micro, f1_score_micro, precision_macro, recall_macro, f1_score_macro, precision_weighted, recall_weighted, f1_score_weighted, classification_report_


def preprocess_function(model, test_df, note, dataset, tokenizer, max_length, labels_to_ids, config, device):
    label_space = len(labels_to_ids)
    model_inputs = {}
    model_inputs["input_ids"] = []
    model_inputs["attention_mask"] = []
    model_inputs["labels"] = []
    sequential_prompt = False
    long_seq_count = 0
    pred_labels = []
    gold_labels = []

    for key, i in test_df.iterrows():
        samples = i["samples"]
        inputs = i["sentences"]
        labels = i["labels"]
        whole_paragraph = " ".join(inputs)
        if config.with_illustration == True:
            if dataset in ["biorc800"]:
                query = samples + "<Start> The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result and conclusion"

        else:
            if dataset in ["biorc800"]:
                query = "The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result and conclusion"

        if sequential_prompt == True:
            tokenized_text = tokenizer.tokenize(query)
            # Convert tokens to input IDs
            input_ids_initial = tokenizer.convert_tokens_to_ids(tokenized_text)
            attention_mask_initial = [1] * len(input_ids_initial)
            label_ids_initial = [-100] * len(attention_mask_initial)
            for sentence, label in zip(inputs, labels):
                query_single_sentence = ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as "
                tokenized_text = tokenizer.tokenize(query_single_sentence)
                single_sentence_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                input_ids = input_ids_initial + single_sentence_ids
                attention_mask = attention_mask_initial + [1] * len(single_sentence_ids)
                label_ids = [-100] * len(label_ids_initial) + [-100] * len(single_sentence_ids)

                all_input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids)) + input_ids
                all_label_ids = [-100] * (max_length - len(label_ids)) + label_ids
                all_attention_masks = [0] * (max_length - len(attention_mask)) + attention_mask

                input_ids_initial += input_ids
                tokenized_label = tokenizer.tokenize(", ".join(label) + ">")
                single_label_ids = tokenizer.convert_tokens_to_ids(tokenized_label)
                input_ids_initial += single_label_ids

                if len(all_input_ids) > max_length:
                    long_seq_count += 1
                    continue
                else:
                    model_inputs["input_ids"].append(all_input_ids[:max_length])
                    label_one_hot = [0] * label_space
                    for i in range(len(label)):
                        label_one_hot[labels_to_ids[label[i]]] = 1
                    model_inputs["labels"].append(label_one_hot)
                    model_inputs["attention_mask"].append(all_attention_masks[:max_length])

        elif sequential_prompt == False:
            # Tokenize input text
            tokenized_text = tokenizer.tokenize(query)
            # Convert tokens to input IDs
            input_ids_initial = tokenizer.convert_tokens_to_ids(tokenized_text)
            attention_mask_initial = [1] * len(input_ids_initial)
            label_ids_initial = [-100] * len(attention_mask_initial)
            for sentence, label in zip(inputs, labels):
                query_single_sentence = ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as "
                tokenized_text = tokenizer.tokenize(query_single_sentence)
                single_sentence_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                input_ids = input_ids_initial + single_sentence_ids
                attention_mask = attention_mask_initial + [1] * len(single_sentence_ids)

                all_input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids)) + input_ids
                input_text = tokenizer.decode(input_ids)

                if len(input_ids) > max_length:
                    long_seq_count += 1
                    print("Long sequence")
                    continue
                    # input_text = input_text.split("<End>")[1]
                inputs = tokenizer(input_text, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model.generate(**inputs, max_new_tokens=5)
                outputs = outputs.to("cpu")
                pred_label = str(tokenizer.batch_decode(outputs, skip_special_tokens=True))[-40:]
                pred_label = map_to_normal(pred_label, dataset)

                pred_labels.append(pred_label)
                gold_labels.append(label)

                # else:
                #     inputs = tokenizer(input_text, return_tensors="pt")
                #     inputs = {key: value.to(device) for key, value in inputs.items()}
                #     outputs = model.generate(**inputs, max_new_tokens=3)
                #     outputs = outputs.to("cpu")
                #     pred_label = str(tokenizer.batch_decode(outputs, skip_special_tokens=True))[-20:]
                #     pred_label = map_to_normal(pred_label, dataset)
                #     pred_labels.append([pred_label])
                #     gold_labels.append(label)
    print("long_seq_count: ", long_seq_count)
    return gold_labels, pred_labels


def convert_to_one_hot(y, num_classes):
    one_hot = []
    for labels in y:
        one_hot_vector = [0] * num_classes
        for label in labels:
            one_hot_vector[label] = 1
        one_hot.append(one_hot_vector)
    return one_hot

def train_llm(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_retrieve = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    data_train = load_json("data_llm/" + config.dataset + "/train.jsonl")
    paragraph_train = data_train["text"].to_list()
    paragraph_train = [" ".join(i) for i in paragraph_train]
    retrieve_df_train = pd.DataFrame(
        zip(data_train["abstract_id"].to_list(), data_train["text"].to_list(), data_train["labels"].to_list(),
            paragraph_train), columns=['abstract_id', "sentences", "labels", 'paragraph_train'])
    model_retrieve.build_index(paragraph_train)

    if config.dataset in ["biorc800"]:
        labels_to_ids = {"background": 0, "objective": 1, "method": 2, "result": 3, "conclusion": 4, "other": 5}


    if config.dataset in ["biorc800"]:
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as methods or resutls, and the last several sentences might play a rhetorical role as conclusions. "

    test_df = load_data(config, model_retrieve, "test", retrieve_df_train)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto")
    model = model.to(device)

    gold_labels, pred_labels = preprocess_function(model, test_df, note, config.dataset, tokenizer, config.max_length, labels_to_ids, config, device)
    classes = list(labels_to_ids.keys())
    mlb = MultiLabelBinarizer(classes=classes)
    gold_labels = mlb.fit_transform(gold_labels)
    pred_labels = mlb.transform(pred_labels)

    early_stopper = EarlyStopper(patience=3, min_delta=10)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.75)

    precision_micro, recall_micro, f1_score_micro, precision_macro, recall_macro, f1_score_macro, precision_weighted, recall_weighted, f1_score_weighted, classification_report_ = evaluation(gold_labels, pred_labels, classes)
    # precision_micro, recall_micro, f1_score_micro, precision_macro, recall_macro, f1_score_macro, precision_weighted, recall_weighted, f1_score_weighted, classification_report_ = evaluation_single_label(all_val_labels, all_val_outputs)

    print("\nOverall Precision (micro):", precision_micro)
    print("Overall Recall (micro):", recall_micro)
    print("Overall F1-Score (micro):", f1_score_micro)

    print("\nOverall Precision (macro):", precision_macro)
    print("Overall Recall (macro):", recall_macro)
    print("Overall F1-Score (macro):", f1_score_macro)

    print("\nOverall Precision (weighted):", precision_weighted)
    print("Overall Recall (weighted):", recall_weighted)
    print("Overall F1-Score (weighted):", f1_score_weighted)
    print("classification report: ")
    print(classification_report_)


if __name__ == '__main__':
    # hyperparameters
    parser = ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default="google/gemma-2b",
                        help='llm model')

    parser.add_argument('--tokenizer_name_or_path', type=str,default="google/gemma-2b",
                        help='llm model tokenizer')

    parser.add_argument('--dropout_thinking_linear', type=float, default=0.1,
                        help='dropout rate')

    parser.add_argument('--num_generate_tokens', type=int, default=2,
                        help='number of the generated tokens')

    parser.add_argument('--max_length', type=int, default=8192,
                        help='maximum input length')

    parser.add_argument('--dataset', type=str, default="biorc800",
                        help='name of the dataset')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')

    parser.add_argument('--bank_size', type=int, default=2000,
                        help='memory bank size')

    parser.add_argument('--default_threshold', type=float, default=0.4,
                        help='default threshold')

    parser.add_argument('--with_illustration', type=bool, default=True,
                        help='with illustration')

    parser.add_argument("--update_memory_bank_steps", type=int, default=50000)

    parser.add_argument("--sample_count", type=int, default=5)

    config = parser.parse_args()
    print("config", config)
    train_llm(config)
