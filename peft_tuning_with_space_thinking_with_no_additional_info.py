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
import pickle


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
    if dataset in ["art_coresc"]:
        dataset_items = []

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
                        illustrations += "The paragraph is \"" + " ".join(
                            i['sentences']) + "\". Select from rhetorical labels including background, motivation, hypothesis, goal, objective, method, result, observation, experiment, conclusion"
                        for sentence, label in zip(i['sentences'], i['labels']):
                            illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + label.lower() + ">"
                    illustrations += " <End>\n"
                break
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
        dataset_items.append({"samples": illustrations, "sentences": paragraph, "labels": [[label.lower()] for label in true_labels]})
        return dataset_items
    elif dataset == "csabstract":
        dataset_items = []
        whole_paragraph = " ".join(paragraph)
        results = model_retrieve.search(whole_paragraph, top_k=500, threshold=0.1)
        # illustrations = "<Start> " + note
        illustrations = "<Start> "
        if len(results) == 0:
            print("results == 0")

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
                            i['sentences']) + "\". Select from rhetorical labels including background, objective, method, result and conclusion"
                        for sentence, label in zip(i['sentences'], i['labels']):
                            illustrations += ", the sentence \"" + sentence + "\" plays rhetorical role in the paragraph as <" + label.lower().rstrip("s") + ">"
                        illustrations += " <End>\n"
                break
        dataset_items.append({"samples": illustrations, "sentences": paragraph, "labels": [[label.lower().rstrip("s")] for label in true_labels]})
        return dataset_items
    elif dataset == "pubmed_20k":
        dataset_items = []

        whole_paragraph = " ".join(paragraph)
        results = model_retrieve.search(whole_paragraph, top_k=500, threshold=0.1)
        # illustrations = "<Start> " + note
        illustrations = "<Start> "
        if len(results) == 0:
            print("results == 0")

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

                break
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

def load_data(config, model_retrieve, mode, retrieve_df_train):
    data = load_json("data_llm/" + config.dataset + "/" + mode + ".jsonl")
    if mode == "train":
        data = data[:5000]
    elif mode == "test":
        data = data[:1000]
    elif mode == "dev":
        data = data[:1000]
    print("mode: ", len(data))
    df = pd.DataFrame(data=construct_dataset_single(data, model_retrieve, retrieve_df_train, config.dataset))
    return df


def preprocess_function(examples, note, dataset, tokenizer, max_length, labels_to_ids, config):
    label_space = len(labels_to_ids)
    batch_size = len(examples["sentences"])
    model_inputs = {}
    model_inputs["input_ids"] = []
    model_inputs["attention_mask"] = []
    model_inputs["labels"] = []
    sequential_prompt = False
    long_seq_count = 0

    for i in range(batch_size):
        samples = examples["samples"][i]
        inputs = examples["sentences"][i]
        labels = examples["labels"][i]
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
            elif dataset == "art_coresc":
                query = "The paragraph is \"" + whole_paragraph + "\". Select from rhetorical labels including background, motivation, hypothesis, goal, objective, method, observation, result, experiment, conclusion"

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
                print(len(model_inputs["labels"]))
            print("Long sequence count: ", long_seq_count)
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
                label_ids = [-100] * len(label_ids_initial) + [-100] * len(single_sentence_ids)

                all_input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids)) + input_ids
                all_attention_masks = [0] * (max_length - len(attention_mask)) + attention_mask
                input_text = tokenizer.decode(input_ids)
                if len(all_input_ids) > max_length:
                    print(input_text)
                    input_text = input_text.split("<End>")[1]
                    # print(input_text)
                    tokenized_text = tokenizer.tokenize(input_text)
                    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                    attention_mask = [1] * len(input_ids)

                    all_input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids)) + input_ids
                    all_attention_masks = [0] * (max_length - len(attention_mask)) + attention_mask
                    if len(all_input_ids) > max_length:
                        # print(tokenizer.decode(all_input_ids))
                        continue
                    model_inputs["input_ids"].append(all_input_ids[:max_length])
                    label_one_hot = [0] * label_space
                    for i in range(len(label)):
                        label_one_hot[labels_to_ids[label[i]]] = 1
                    model_inputs["labels"].append(label_one_hot)
                    model_inputs["attention_mask"].append(all_attention_masks[:max_length])

                else:
                    model_inputs["input_ids"].append(all_input_ids[:max_length])
                    label_one_hot = [0] * label_space
                    for i in range(len(label)):
                        label_one_hot[labels_to_ids[label[i]]] = 1
                    model_inputs["labels"].append(label_one_hot)
                    model_inputs["attention_mask"].append(all_attention_masks[:max_length])
                    # print("input prompt: ", f"{tokenizer.decode(all_input_ids[:max_length])}")

    # print("long_seq_count: ", long_seq_count)
                #
                # tokenized_label = tokenizer.tokenize(", ".join(label))
                # single_label_ids = tokenizer.convert_tokens_to_ids(tokenized_label)
                # input_ids += single_label_ids
                # attention_mask += [1] * len(single_label_ids)
                # label_ids += single_label_ids
                # tokenized_text = tokenizer.tokenize(">")
                # single_sentence_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                # input_ids += single_sentence_ids
                # attention_mask += [1] * len(single_sentence_ids)
                # label_ids += [-100] * len(single_sentence_ids)
    return model_inputs


def pred_single_label(output, id, threshold):
    probs = np.array(output)
    predictions = []
    for idx in range(probs.shape[0]):
        output = []
        for l in range(0, probs.shape[1]):
            if probs[idx, l] >= threshold:
                if l == id:
                    output.append(1)
                    break
        if len(output) == 0:
            # predict = np.argmax(probs[idx])
            # if predict == id:
            #     output.append(1)
            # else:
            output.append(0)

        predictions.extend(output)

    return predictions

def find_best_thresholds(output, labels, labels_to_id, default_threshold):
    thresholds = {}
    for label, id in labels_to_id.items():
        best_f1 = -1
        for threshold in [x * 0.1 for x in range(0, 10)]:
            predictions = pred_single_label(output, id, threshold)
            f1 = f1_score([1 if singlelabels[id] == 1 else 0 for singlelabels in labels], predictions)
            if f1 > best_f1:
                best_f1 = f1
                if threshold == 0:
                    thresholds[label] = default_threshold
                else:
                    thresholds[label] = threshold
    print(thresholds)
    return thresholds

def save_div(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0


def evaluation(true_labels, predicted_labels):

    print("number of true labels: ", len(true_labels))
    print("number of predicted labels: ", len(predicted_labels))

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

def evaluation_single_label(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    true_labels = np.argmax(true_labels, axis=1)
    predicted_labels = np.argmax(predicted_labels, axis=1)

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


def predict_thresholds(logits, thresholds, vocab, default_threshold):
    inv_vocab = {v:k for k,v in vocab.items()}
    probs = np.array(logits)
    predictions = []
    for idx in range(probs.shape[0]):
        output = [0] * len(vocab)
        for l in range(0, probs.shape[1]):
            if inv_vocab[l] in thresholds:
                if probs[idx, l] >= thresholds[inv_vocab[l]]:
                    output[l]=1
            else:
                if probs[idx, l] >= default_threshold:
                    output[l]=1
        if 1 not in output:
            output[np.argmax(probs[idx]).item()]=1
        else:
            output = output
        predictions.append(output)

    return predictions

def predict(logits, threshold):
    probs = np.array(logits)
    predictions = []
    for idx in range(probs.size(0)):
        output = []
        for l in range(0, probs.size(1)):
            if probs[idx, l] >= threshold:
                output.append(l)
        if len(output) == 0:
            output.append(torch.argmax(probs[idx]).item())
        else:
            output = output

        predictions.append(output)

    return predictions

def cosine_sim(x1, x2, eps=1e-15, temperature=1):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)

def pairwise_concatenate(matrix1, matrix2, n1, n2):
    idx = torch.cartesian_prod(torch.arange(n1), torch.arange(n2))
    concatenated_pairs = torch.cat((matrix1[idx[:, 0]], matrix2[idx[:, 1]]), dim=1)
    return concatenated_pairs

### multi-label
## check if the shape of current_labels is N x C, where C is the number of classes.
def contrastive_loss(current_logits, current_labels, memory_bank, memory_label, device, model, contrastive_mode, multi_label=False):
    con_loss = 0
    class_num = current_labels.shape[1]
    memory_bank = torch.from_numpy(memory_bank)
    memory_bank = memory_bank.to(device)
    memory_label = torch.from_numpy(memory_label)
    memory_label = memory_label.to(device).float()
    current_labels = torch.from_numpy(current_labels)
    current_labels = current_labels.to(device).float()

    for i in range(class_num):
        pos_idx = torch.where(memory_label[:, i] == 1)[0]
        neg_idx = torch.where(memory_label[:, i] != 1)[0]
        current_logits_idx = torch.where(current_labels[:, i] == 1)[0]
        if len(pos_idx) == 0 or len(current_logits_idx) == 0:
            continue
        positive_logits = memory_bank[pos_idx, :]
        negative_logits = memory_bank[neg_idx, :]
        size = negative_logits.shape[0] + 1
        # distinguish whether it is multi-class problem or multi-label problem
        if multi_label:
            if contrastive_mode == "weighcon":
                n1 = current_labels.shape[0]
                n2 = memory_label.shape[0]
                sim = model.weightfunction(pairwise_concatenate(current_labels, memory_label, n1, n2)).reshape(n1, n2)
                pos_weight = sim[current_logits_idx, :][:, pos_idx]
                neg_weight = 1 - sim[current_logits_idx, :][:, neg_idx]
            elif contrastive_mode == "herocon":
                dist = hamming_distance_by_matrix(current_labels[current_logits_idx], memory_label)
                pos_weight = 1 - dist[current_logits_idx, :][:, pos_idx] / class_num
                neg_weight = dist[current_logits_idx, :][:, neg_idx]
        else:
            pos_weight = 1
            neg_weight = 1
        # print(pos_weight.shape)
        pos_dis = torch.exp(cosine_sim(current_logits[current_logits_idx], positive_logits)) * pos_weight
        neg_dis = torch.exp(cosine_sim(current_logits[current_logits_idx], negative_logits)) * neg_weight
        denominator = neg_dis.sum(1) + pos_dis
        con_loss += -1 * torch.mean(torch.log(pos_dis/denominator))
    return con_loss


def hamming_distance_by_matrix(current_label, memory_label):
    return torch.matmul(current_label, (1 - memory_label).T) + torch.matmul(1 - current_label, memory_label.T)

def update_memory_bank(train_dataloader, device, model):
    memory_bank = []

    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs, outputs_con = model(**batch)
        memory_bank.extend(outputs_con.detach().cpu().numpy())

    memory_bank = np.array(memory_bank)
    return memory_bank

def pairwise_concatenate(self, matrix, n):
    idx = torch.cartesian_prod(torch.arange(n), torch.arange(n))
    concatenated_pairs = torch.cat((matrix[idx[:, 0]], matrix[idx[:, 1]]), dim=1)
    return concatenated_pairs

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
    train_df = load_data(config, model_retrieve, "train", retrieve_df_train)
    val_df = load_data(config, model_retrieve, "dev", retrieve_df_train)
    test_df = load_data(config, model_retrieve, "test", retrieve_df_train)


    train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_df))
    val_dataset = Dataset.from_pandas(pd.DataFrame(data=val_df))
    test_dataset = Dataset.from_pandas(pd.DataFrame(data=test_df))


    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    model = space_thinking_llm(config, num_labels=len(labels_to_ids))


    train_processed_dataset = train_dataset.map(
        lambda batch: preprocess_function(batch, note, config.dataset, tokenizer, config.max_length, labels_to_ids, config),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    val_processed_dataset = val_dataset.map(
        lambda batch: preprocess_function(batch, note, config.dataset, tokenizer, config.max_length, labels_to_ids, config),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    test_processed_dataset = test_dataset.map(
        lambda batch: preprocess_function(batch, note, config.dataset, tokenizer, config.max_length, labels_to_ids, config),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataloader = DataLoader(
        train_processed_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

    val_dataloader = DataLoader(
        val_processed_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

    test_dataloader = DataLoader(
        test_processed_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)

    early_stopper = EarlyStopper(patience=3, min_delta=10)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.75)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    model = model.to(device)
    BCELoss = torch.nn.BCELoss()
    best_val_loss = -1
    best_val_f1 = 0

    print("initialize memory bank")
    model.eval()
    if config.start_from_memory_bank is not None:
        with open(config.start_from_memory_bank_path, 'rb') as f:

            memories = pickle.load(f)
            memory_bank = memories[0]
            memory_labels = memories[1]
    else:
        sample_idx = 0
        memory_bank = []
        memory_labels = []

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs, outputs_con = model(**batch)
            memory_bank.extend(outputs_con.detach().cpu().numpy())
            sample_idx += len(batch["labels"])
            memory_labels.extend(batch["labels"].detach().cpu().numpy())
            memories = [memory_bank, memory_labels]
            with open(config.start_from_memory_bank_path, 'wb') as f:
                pickle.dump(memories, f)

    memory_bank = np.array(memory_bank)
    memory_labels = np.array(memory_labels)
    print("start tuning")
    model.train()

    for epoch in range(config.num_epochs):
        print("epoch: ", epoch)
        model.train()
        train_total_loss = 0.0
        train_con_total_loss = 0.0
        sample_idx = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs, outputs_con = model(**batch)
            # loss = BCELoss(outputs, batch["labels"].to(torch.bfloat16))
            loss = BCELoss(outputs, batch["labels"].float())
            if sample_idx < config.bank_size:
                sub_memory_bank = memory_bank[sample_idx - config.bank_size:]
                if sample_idx != 0:
                    sub_memory_bank = np.concatenate((sub_memory_bank, memory_bank[:sample_idx]), axis=0)
                sub_memory_label = memory_labels[sample_idx - config.bank_size:]
                if sample_idx != 0:
                    sub_memory_label = np.concatenate((sub_memory_label, memory_labels[:sample_idx]), axis=0)
            else:
                sub_memory_bank = memory_bank[sample_idx - config.bank_size:sample_idx]
                sub_memory_label = memory_labels[sample_idx - config.bank_size:sample_idx]

            con_loss = contrastive_loss(outputs_con, batch["labels"].detach().cpu().numpy(), sub_memory_bank, sub_memory_label, device, model, config.contrastive_mode)
            loss = loss + 0.1 * con_loss

            train_total_loss += loss.item()
            train_con_total_loss += con_loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            for i in range(sample_idx, sample_idx + len(batch["labels"])):
                memory_bank[i] = outputs_con.detach().cpu().numpy()[i - sample_idx]
            sample_idx += len(batch["labels"])

            if step != 0 and step % config.update_memory_bank_steps == 0:
                model.eval()
                memory_bank = update_memory_bank(train_dataloader, device, model)
                model.train()

        print("train loss: ", train_total_loss)
        print("train con loss: ", train_con_total_loss)
        model.eval()
        all_val_outputs = []
        all_val_labels = []
        val_total_loss = 0
        for step, batch in enumerate(tqdm(val_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs, out_con = model(**batch)
            # loss = BCELoss(outputs, batch["labels"].to(torch.bfloat16))
            loss = BCELoss(outputs, batch["labels"].float())
            val_total_loss += loss.item()
            output_logits = outputs.cpu().detach().float().numpy()
            output_logits = np.round(output_logits, 2)
            output_logits = output_logits.tolist()
            all_val_outputs.extend(output_logits)
            all_val_labels.extend(batch["labels"].cpu().numpy())

        thresholds = find_best_thresholds(all_val_outputs, all_val_labels, labels_to_ids, config.default_threshold)
        predictions = predict_thresholds(all_val_outputs, thresholds, labels_to_ids, config.default_threshold)
        # predictions = predict(all_val_outputs, config.default_threshold)
        # print(predictions)
        precision_micro, recall_micro, f1_score_micro, precision_macro, recall_macro, f1_score_macro, precision_weighted, recall_weighted, f1_score_weighted, classification_report_ = evaluation(all_val_labels, predictions)
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
        print("val loss", val_total_loss)

        if best_val_f1 != 0 and best_val_f1 < f1_score_micro:
            print("saving best model...")
            torch.save(dict(model=model.state_dict(), config=config), "best_model.mdl")
            best_val_f1 = f1_score_micro
            print("best val f1: ", best_val_f1)
            best_thresholds = thresholds

        if best_val_f1 == 0:
            torch.save(dict(model=model.state_dict(), config=config), "best_model.mdl")
            best_val_f1 = f1_score_micro
            print("best val f1: ", best_val_f1)
            best_thresholds = thresholds
            print("saving best model...")
            torch.save(dict(model=model.state_dict(), config=config), "best_model.mdl")

        if early_stopper.early_stop(val_total_loss):
            break

        # memory_bank = update_memory_bank(train_dataloader, device, model)

    all_test_outputs = []
    all_test_labels = []
    model.eval()
    model.load_state_dict(torch.load("best_model.mdl")['model'])
    test_total_loss = 0.0
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs, outputs_con = model(**batch)
        # loss = BCELoss(outputs, batch["labels"].to(torch.bfloat16))
        loss = BCELoss(outputs, batch["labels"].float())

        test_total_loss += loss.item()
        output_logits = outputs.cpu().detach().float().numpy()
        output_logits = np.round(output_logits, 2)
        output_logits = output_logits.tolist()
        all_test_outputs.extend(output_logits)
        all_test_labels.extend(batch["labels"].cpu().numpy())

    print("testing results: ")
    predictions = predict_thresholds(all_test_outputs, best_thresholds, labels_to_ids, config.default_threshold)
    # predictions = predict(all_test_outputs, config.default_threshold)
    precision_micro, recall_micro, f1_score_micro, precision_macro, recall_macro, f1_score_macro, precision_weighted, recall_weighted, f1_score_weighted, classification_report_ = evaluation(all_test_labels, predictions)
    # precision_micro, recall_micro, f1_score_micro, precision_macro, recall_macro, f1_score_macro, precision_weighted, recall_weighted, f1_score_weighted, classification_report_ = evaluation_single_label(all_test_labels, all_test_outputs)

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
    print(labels_to_ids)

    print("val loss: ", test_total_loss)

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

    parser.add_argument('--max_length', type=int, default=1200,
                        help='maximum input length')

    parser.add_argument('--dataset', type=str, default="pubmed_20k",
                        help='name of the dataset')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    parser.add_argument('--lr', type=float, default=2e-5,
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

    parser.add_argument('--start_from_memory_bank_path', type=str,
                        help='name of the dataset')

    parser.add_argument('--start_from_memory_bank', type=bool,
                        help='name of the dataset')

    parser.add_argument('--contrastive_mode', type=str, help="mode of contrastive learning - choose from herocon and weighcon")

    config = parser.parse_args()
    print("config", config)
    train_llm(config)
