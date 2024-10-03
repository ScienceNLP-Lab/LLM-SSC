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
    if dataset in ["csabstract"]:
        dataset_items = []
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as methods or resutls, and the last several sentences might play a rhetorical role as conclusion. "
        sample_count = 0
        if config.sample_count != 0:
            whole_paragraph = " ".join(paragraph)

            results = model_retrieve.search(whole_paragraph, top_k=300)
            illustrations = "<Start>"
            for i in results:
                if len(i[0].split(" ")) > 180:
                    continue
                elif whole_paragraph[:100].lower() in i[0].lower():
                    continue
                else:
                    retrieved_text_single = re.escape(i[0])
                    retrieve_in_context = retrieve_df_train[
                        retrieve_df_train['paragraph_train'].str.contains(retrieved_text_single)]
                    if len(retrieve_in_context) > 0:
                        for key, i in retrieve_in_context.iterrows():
                            sentence_count = 0
                            illustrations += "The paragraph is \"" + " ".join(
                                i['sentences']) + "\". Select from rhetorical labels including background, objective, method, result and conclusion"
                            for sentence, label in zip(i['sentences'], i['labels']):
                                if sentence_count % 4 == 0:
                                    illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + label.lower().rstrip("s") + ">"
                                sentence_count += 1
                        illustrations += " <End>\n"
                        sample_count += 1
                if sample_count >= config.sample_count:
                    break
        else:
            illustrations = ""
        dataset_items.append({"samples": illustrations, "sentences": paragraph, "labels": [[label.lower().rstrip("s")] for label in true_labels]})
        return dataset_items
    if dataset in ["art_coresc"]:
        dataset_items = []
        sample_count = 0
        if config.sample_count != 0:

            whole_paragraph = " ".join(paragraph)

            results = model_retrieve.search(whole_paragraph, top_k=300)
            illustrations = "<Start>"
            for i in results:
                if len(i[0].split(" ")) > 180:
                    continue
                elif whole_paragraph[:100].lower() in i[0].lower():
                    continue
                else:
                    retrieved_text_single = re.escape(i[0])
                    retrieve_in_context = retrieve_df_train[
                        retrieve_df_train['paragraph_train'].str.contains(retrieved_text_single)]
                    if len(retrieve_in_context) > 0:
                        for key, i in retrieve_in_context.iterrows():
                            sentence_count = 0
                            illustrations += "The paragraph is \"" + " ".join(
                                i['sentences']) + "\". Select from rhetorical labels including background, motivation, hypothesis, goal, objective, method, result, experiment, conclusion"
                            for sentence, label in zip(i['sentences'], i['labels']):
                                if sentence_count % 4 == 0:
                                    illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + label.lower().rstrip(
                                        "s") + ">"
                                sentence_count += 1
                        illustrations += " <End>\n"
                        sample_count += 1
                if sample_count >= config.sample_count:
                    break
        else:
            illustrations = ""
        dataset_items.append({"samples": illustrations, "sentences": paragraph, "labels": [[label.lower().rstrip("s")] for label in true_labels]})
        return dataset_items
        # if illustrations == "<Start> ":
        #     illustration = '''<Start>The paragraph is "This paper presents the modeling of the light-weight BioRob robot arm with series elastic actuation for simulation and controller design. We describe the kinematic coupling introduced by the cable \
        #     actuation and the robot arm dynamics including the elastic actuator and motor and gear model. We show how the inverse dynamics model derived from these equations can be used as a basis for a position tracking controller that is able to \
        #     sufficiently damp the oscillations caused by the high, nonlinear joint elasticity. We presents results from simulation and briefly describe the implementation for a real world application.". Select from background, objective, method, result \
        #     and conclusion, the sentence "This paper presents the modeling of the light-weight BioRob robot arm with series elastic actuation for simulation and controller design." plays rhetorical role in the paragraph as <background>, the sentence "We \
        #     describe the kinematic coupling introduced by the cable actuation and the robot arm dynamics including the elastic actuator and motor and gear model." plays rhetorical role in the paragraph as <method>, the sentence "We show how the inverse \
        #     dynamics model derived from these equations can be used as a basis for a position tracking controller that is able to sufficiently damp the oscillations caused by the high, nonlinear joint elasticity.(" plays rhetorical role in the paragraph \
        #     as <method>, the sentence "We presents results from simulation and briefly describe the implementation for a real world application.") plays rhetorical role in the paragraph as <result> <End> \
        #     '''
        #     dataset_items.append({"samples": illustration, "sentences": paragraph,
        #                           "labels": [[label.lower().strip("s")] for label in true_labels]})
        #     return dataset_items
        # else:


    elif dataset == "csabstruct":
        dataset_items = []
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as method, and the last several sentences might play a rhetorical role as result. "

        whole_paragraph = " ".join(paragraph)
        results = model_retrieve.search(whole_paragraph, top_k=500, threshold=0.1)
        # illustrations = "<Start> " + note
        sample_count = 0
        illustrations = "<Start> "

        for i in results:
            if len(i[0].split(" ")) > 120:
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
                            i['sentences']) + "\". Select from rhetorical labels including background, objective, method, and result"
                        for sentence, label in zip(i['sentences'], i['labels']):
                            illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + label.lower().rstrip("s") + ">"
                        illustrations += " <End>\n"

                    sample_count += 1
                if sample_count >= config.sample_count:
                    break
        dataset_items.append({"samples": illustrations, "sentences": paragraph, "labels": [[label.lower().rstrip("s")] for label in true_labels]})
        return dataset_items
    elif dataset == "pubmed_20k":
        dataset_items = []
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as method, and the last several sentences might play a rhetorical role as result. "

        whole_paragraph = " ".join(paragraph)
        results = model_retrieve.search(whole_paragraph, top_k=500, threshold=0.1)
        # illustrations = "<Start> " + note
        sample_count = 0
        illustrations = "<Start> "
        if config.sample_count > 0:

            for i in results:
                if len(i[0].split(" ")) > 180:
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
                                i['sentences']) + "\". Select from rhetorical labels including background, objective, method, result, and conclusion"
                            for sentence, label in zip(i['sentences'], i['labels']):
                                illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + label.lower().rstrip("s") + ">"
                            illustrations += " <End>\n"

                        sample_count += 1
                    if sample_count >= config.sample_count:
                        break
        else:
            illustrations = ""
    #         if illustrations == "<Start> ":
#             illustration = '''<Start> The paragraph is "Computerized algorithms and solutions in processing and diagnosis mammography X-ray, cardiovascular CT/MRI scans, and microscopy image play \
#  an important role in disease detection and computer-aided decision-making. Machine learning techniques have powered many aspects in medical investigations and clini \
# cal practice. Recently, deep learning is emerging a leading machine learning tool in computer vision and begins attracting considerable attentions in medical imaging \
# . In this chapter, we provide a snapshot of this fast growing field specifically for mammography, cardiovascular, and microscopy image analysis. We briefly explain t \
# he popular deep neural networks and summarize current deep learning achievements in various tasks such as detection, segmentation, and classification in these hetero \
# geneous imaging modalities. In addition, we discuss the challenges and the potential future trends for ongoing work. 2.". Select from background, objective, method,  \
# and result, the sentence "Computerized algorithms and solutions in processing and diagnosis mammography X-ray, cardiovascular CT/MRI scans, and microscopy image play \
#  an important role in disease detection and computer-aided decision-making." plays rhetorical role in the paragraph as <background>, the sentence "Machine learning t \
# echniques have powered many aspects in medical investigations and clinical practice." plays rhetorical role in the paragraph as <background>, the sentence "Recently, \
#  deep learning is emerging a leading machine learning tool in computer vision and begins attracting considerable attentions in medical imaging." plays rhetorical rol \
# e in the paragraph as <background>, the sentence "In this chapter, we provide a snapshot of this fast growing field specifically for mammography, cardiovascular, and \
#  microscopy image analysis." plays rhetorical role in the paragraph as <objective>, the sentence "We briefly explain the popular deep neural networks and summarize c \
# urrent deep learning achievements in various tasks such as detection, segmentation, and classification in these heterogeneous imaging modalities." plays rhetorical r \
# ole in the paragraph as <objective>, the sentence "In addition, we discuss the challenges and the potential future trends for ongoing work." plays rhetorical role in the paragraph as <method>, the sentence "2." plays rhetorical role in the paragraph as <other> <End>
#
# '''
#             dataset_items.append({"samples": illustration, "sentences": paragraph,
#                                   "labels": [[label.lower().strip("s")] for label in true_labels]})
#             return dataset_items
#         else:
        dataset_items.append({"samples": illustrations, "sentences": paragraph, "labels": [[label.lower().rstrip("s")] for label in true_labels]})
        return dataset_items


    elif dataset == "nicta_piboso":
        dataset_items = []

        whole_paragraph = " ".join(paragraph)
        results = model_retrieve.search(whole_paragraph, top_k=500, threshold=0.1)
        # illustrations = "<Start> " + note
        illustrations = "<Start> "

        sample_count = 0
        for i in results:
            if len(i[0].split(" ")) > 150:
                continue
            else:
                retrieved_text_single = re.escape(i[0])
                retrieve_in_context = retrieve_df_train[
                    retrieve_df_train['paragraph_train'].str.contains(retrieved_text_single)]
                if len(retrieve_in_context) > 0:
                    for key, i in retrieve_in_context.iterrows():
                        illustrations += "The paragraph is \"" + " ".join(
                            i['sentences']) + "\". Select from rhetorical labels including background, study desgin, population, intervention, outcome, and other"
                        for sentence, label in zip(i['sentences'], i['labels']):
                            illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + label.lower().rstrip("s") + ">"
                        illustrations += " <End>\n"
                    sample_count += 1
                if sample_count >= config.sample_count:
                    break

        dataset_items.append({"samples": illustrations, "sentences": paragraph,
                              "labels": [[label.lower().rstrip("s")] for label in true_labels]})
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
    if dataset == "csabstract":
        for candidates in ["background", "objective", "method", "result", "conclusion", "other"]:
            if candidates in s1.lower() or candidates == s1:
                return candidates
        return "background"
    if dataset == "pubmed_20k":
        for candidates in ["background", "objective", "method", "result", "conclusion"]:
            if candidates in s1.lower() or candidates == s1:
                return candidates
        return "background"
    if dataset == "nicta_piboso":
        for candidates in ["background", "study", "population", "intervention", "outcome", "other"]:
            if candidates in s1.lower() or candidates == s1:
                if candidates == "study":
                    candidates = "study design"
                return candidates
        return "background"
    if dataset == "art_coresc":
        for candidates in ["background", "conclusion", "observation", "method", "objective", "result", "goal", "experiment", "motivation", "hypothesis"]:
            if candidates in s1.lower() or candidates == s1:
                return candidates
        return "background"
def load_data(config, model_retrieve, mode, retrieve_df_train):
    data = load_json("data_llm/" + config.dataset + "/" + mode + ".jsonl")
    data = data[:1000]
    df = pd.DataFrame(data=construct_dataset_single(data, model_retrieve, retrieve_df_train, config.dataset))
    return df

def evaluation(true_labels, predicted_labels):

    precision_micro = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
    recall_micro = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
    f1_score_micro = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)

    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_score_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1_score_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    classification_report_ = classification_report(true_labels, predicted_labels)

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
            if dataset in ["csabstract"]:
                query = samples + "<Start> The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result and conclusion"
            elif dataset == "csabstruct":
                query = samples + "<Start> The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result, and other"
            elif dataset == "nicta_piboso":
                query = samples + "<Start> The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, study desgin, population, intervention, outcome, and other"
            elif dataset == "pubmed_20k":
                query = samples + "<Start> The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result and conclusion"
            elif dataset == "art_coresc":
                query = samples + "<Start> The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, motivation, hypothesis, goal, objective, method, observation, result, experiment, conclusion"

        else:
            if dataset in ["csabstract"]:
                query = "The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result and conclusion"
            elif dataset == "csabstruct":
                query = "The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result, and other"
            elif dataset == "csabstract":
                query = "The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, objective, method, result and conclusion"
            elif dataset == "nicta_piboso":
                query = "The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, study desgin, population, intervention, outcome, and other"
            elif dataset == "pubmed_20k":
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
                # print("input prompt: ", f"{tokenizer.decode(input_ids)}")

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
                input_text = tokenizer.decode(input_ids)
                # print("input_text: ", input_text)

                if len(input_ids) > max_length:
                    long_seq_count += 1
                    print("Long sequence")
                    continue
                    # input_text = input_text.split("<End>")[1]
                inputs = tokenizer(input_text, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model.generate(**inputs, max_new_tokens=3)
                outputs = outputs.to("cpu")
                pred_label = str(tokenizer.batch_decode(outputs, skip_special_tokens=True))[-20:]
                pred_label = map_to_normal(pred_label, dataset)
                pred_labels.append([pred_label])
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
    if config.dataset == "csabstruct":
        labels_to_ids = {"background": 0, "objective": 1, "method": 2, "result": 3, "other": 4}
    elif config.dataset == "csabstract":
        labels_to_ids = {"background": 0, "objective": 1, "method": 2, "result": 3, "conclusion": 4}
    elif config.dataset == "nicta_piboso":
        labels_to_ids = {"background": 0, "study design": 1, "population": 2, "intervention": 3, "outcome": 4, "other": 5}
    elif config.dataset == "pubmed_20k":
        labels_to_ids = {"background": 0, "objective": 1, "method": 2, "result": 3, "conclusion": 4}
    elif config.dataset == "art_coresc":
        labels_to_ids = {"background": 0, "motivation": 1, "hypothesis": 2, "goal": 3, "objective": 4, "method": 5, "result": 6, "experiment": 7, "observation":8, "conclusion": 9}

    if config.dataset in ["csabstract"]:
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as methods or resutls, and the last several sentences might play a rhetorical role as conclusions. "
    elif config.dataset == "csabstruct":
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as method, and the last several sentences might play a rhetorical role as result. "
    elif config.dataset == "nicta_piboso":
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background, the middle several sentences might play a rhetorical role as population or intervention, and the last several sentences might play a rhetorical role as outcome. "
    elif config.dataset == "pubmed_20k":
        note = "Note that in the paragraph, the first several sentences might play a rhetorical role as background or objective, the middle several sentences might play a rhetorical role as method, and the last several sentences might play a rhetorical role as result. "
    elif config.dataset == "art_coresc":
        note = ""

    test_df = load_data(config, model_retrieve, "test", retrieve_df_train)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto")
    model = model.to(device)

    gold_labels, pred_labels = preprocess_function(model, test_df, note, config.dataset, tokenizer, config.max_length, labels_to_ids, config, device)

    precision_micro, recall_micro, f1_score_micro, precision_macro, recall_macro, f1_score_macro, precision_weighted, recall_weighted, f1_score_weighted, classification_report_ = evaluation(gold_labels, pred_labels)
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

    parser.add_argument('--dataset', type=str, default="csabstract",
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
