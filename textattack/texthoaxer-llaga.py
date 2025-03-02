import argparse
import os
import certifi
import numpy as np
from pathlib import Path
from scipy.special import softmax
np.random.seed(1234)
import pickle
import dataloader
from train_classifier import Model
from itertools import zip_longest
import criteria
import random
random.seed(0)
import csv
import math
import sys
csv.field_size_limit(sys.maxsize)

import tensorflow_hub as hub
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
# tf.compat.v1.disable_eager_execution()
import ssl
from tqdm import tqdm
import json

# import packages for llaga
sys.path.append(os.path.abspath('/scratch/qz2086/TrustGLM/Baselines/LLaGA'))
from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math
from sentence_transformers import SentenceTransformer


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_path = '/scratch/qz2086/hard-label-attack/sentence-encoder'
        self.embed = hub.load(module_path)
        print('Sentence encoder loaded')

        # Enable GPU memory growth for TensorFlow 2.x
        gpus = tf.config.list_physical_devices('GPU')
        print("Available GPUs:", gpus)
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for GPUs")
            except RuntimeError as e:
                print(f"Failed to enable memory growth: {e}")


    def semantic_sim(self, sents1, sents2):

        input1 = tf.convert_to_tensor(sents1, dtype=tf.string)
        input2 = tf.convert_to_tensor(sents2, dtype=tf.string)

        # Use the embed model's default signature
        default_signature = self.embed.signatures['default']
        sts_encode1 = tf.nn.l2_normalize(default_signature(input1)['default'], axis=1)
        sts_encode2 = tf.nn.l2_normalize(default_signature(input2)['default'], axis=1)

        # Calculate cosine similarity
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
        return sim_scores



class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()


        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):

        self.model.eval()


        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []

        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):


    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)


            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):

    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_rang_min = 0
        text_range_max = len_text
    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
            list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))

    return semantic_sims


def get_predictor_results(new_text, orig_batch, predictor, tokenizer, stop_str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        # unpack input batch
        input_ids = orig_batch[0].to(device)
        sbert_embedding = orig_batch[1]
        # print('original graph embedding shape', graph_embedding.shape, flush = True)
        structure_emb = orig_batch[2]
        # print("original structure emebdding shape", structure_emb.shape, flush = True)
        graph = orig_batch[3].to(device)
        do_sample = orig_batch[4]
        temperature = orig_batch[5]
        num_beams = orig_batch[6]
        max_new_tokens = orig_batch[7]
        use_cache = orig_batch[8]
        idx = orig_batch[9]
        graph = orig_batch[10]
        mask = orig_batch[11]
        
        # replace old word embedding with the new one
        new_text = " ".join(new_text)
        new_text_embedding = get_sbert_embedding(new_text)
        new_sbert_embedding = sbert_embedding
        new_sbert_embedding[idx] = new_text_embedding

        
        masked_graph_emb = new_sbert_embedding[graph[mask]]
        s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
        graph_emb = torch.zeros((s, n, d))
        graph_emb[mask] = masked_graph_emb

        new_graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
        new_graph_emb = new_graph_emb.half().cuda()

        # get output from model
        output_ids = predictor(input_ids, graph_emb=new_graph_emb, graph=graph, do_sample = do_sample,temperature = temperature, num_beams = num_beams, max_new_tokens = max_new_tokens, use_cache=use_cache)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        output_label = outputs.strip()

    return output_label

    
def get_attack_result(new_text, orig_batch, predictor, orig_label, tokenizer, stop_str):
    
    if isinstance(new_text[0], str):
        output_label = get_predictor_results(new_text, orig_batch, predictor, tokenizer, stop_str)
        pr = np.array([orig_label != output_label], dtype=int)
        return pr
    elif isinstance(new_text[0], list):
        pr = []
        for t in new_text:
            output_label = get_predictor_results(new_text, orig_batch, predictor, tokenizer, stop_str)
            pr_t = 1 if orig_label != output_label else 0
            pr.append(pr_t)
        pr = np.array(pr)
        return pr


def soft_threshold(alpha, beta):
    if beta > alpha:
        return beta - alpha
    elif beta < -alpha:
        return beta + alpha
    else:
        return 0


def texthoaxer_attack(fuzz_val, top_k_words, allowed_qrs, input_ids, batch, true_label, orig_label, text, predictor,tokenizer, stop_str, 
            stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32,embed_func = '',budget=1000):

    
    text_ls = text.split()

    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0, 0, 0
    else:
        
        word_idx_dict={}
        with open(embed_func, 'r') as ifile:
            for index, line in enumerate(ifile):
                word = line.strip().split()[0]
                word_idx_dict[word] = index


        embed_file=open(embed_func)
        embed_content=embed_file.readlines()


        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]

        words_perturb_idx= []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                words_perturb_embed.append([float(num) for num in embed_content[ word_idx_dict[word] ].strip().split()[1:]])

        words_perturb_embed_matrix = np.asarray(words_perturb_embed)
        words_perturb_embed_matrix = np.atleast_2d(words_perturb_embed_matrix)


        synonym_words,synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms


        qrs = 0
        num_changed = 0
        flag = 0
        th = 0
        while qrs < len(text_ls):
            random_text = text_ls[:]
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                if i >= th:
                    break
            pr = get_attack_result(random_text, batch, predictor, orig_label, tokenizer, stop_str)
            qrs+=1
            th +=1
            if th > len_text:
                break
            if np.sum(pr)>0:
                flag = 1
                break
        old_qrs = qrs
        while qrs < old_qrs + 2500 and flag == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result(random_text, batch, predictor, orig_label, tokenizer, stop_str)
            qrs+=1
            if np.sum(pr)>0:
                flag = 1
                break

        if flag == 1:
            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed+=1
            print(changed)

            while True:
                choices = []

                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        qrs+=1
                        pr = get_attack_result(new_text, batch, predictor, orig_label, tokenizer, stop_str)
                        if np.sum(pr) > 0:
                            choices.append((i,semantic_sims[0]))


                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]
                        # pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        pr = get_attack_result(new_text, batch, predictor, orig_label, tokenizer, stop_str)
                        qrs+=1
                        if pr[0] == 0:
                            break
                        random_text[choices[i][0]] = text_ls[choices[i][0]]

                if len(choices) == 0:
                    break

            changed_indices = []
            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed_indices.append(i)
                    num_changed+=1
            print(str(num_changed)+" "+str(qrs))
            # r_b = batch
            # r_b['desc'] = random_text
            random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]


            if qrs > budget:
                return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                    orig_label, get_predictor_results(random_text, batch, predictor, tokenizer, stop_str), qrs, random_sim, random_sim

            if num_changed == 1:
                return ' '.join(random_text), 1, 1, \
                    orig_label, get_predictor_results(random_text, batch, predictor, tokenizer, stop_str), qrs, random_sim, random_sim


            best_attack = random_text
            best_sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)

            gamma = 0.3*np.ones([words_perturb_embed_matrix.shape[0], 1])
            l1 = 0.1
            l2_lambda = 0.1
            for t in range(100):

                theta_old_text = best_attack
                sim_old= best_sim 
                old_adv_embed = []
                for idx in words_perturb_doc_idx:
                    old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[theta_old_text[idx]]].strip().split()[1:]])
                old_adv_embed_matrix = np.asarray(old_adv_embed)
                old_adv_embed_matrix = np.atleast_2d(old_adv_embed_matrix)

                theta_old = old_adv_embed_matrix-words_perturb_embed_matrix
                theta_old = np.atleast_2d(theta_old)
               
                u_vec = np.random.normal(loc=0.0, scale=1,size=theta_old.shape)
                theta_old_neighbor = theta_old+0.5*u_vec
                theta_old_neighbor = np.atleast_2d(theta_old_neighbor)

                theta_perturb_dist = np.sum((theta_old_neighbor)**2, axis=1)
                nonzero_ele = np.nonzero(np.linalg.norm(theta_old,axis = -1))[0].tolist()
                perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])

                theta_old_neighbor_text = text_ls[:]
                for perturb_idx in range(len(nonzero_ele)):
                    perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                    word_dict_idx = words_perturb_idx[perturb_word_idx]
                    
                    perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_old_neighbor[perturb_word_idx]
                    syn_feat_set = []
                    for syn in synonyms_all[perturb_word_idx][1]:
                        syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                        syn_feat_set.append(syn_feat)
                    syn_feat_set = np.asarray(syn_feat_set)
                    syn_feat_set = np.atleast_2d(syn_feat_set)
                    perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                    perturb_syn_order = np.argsort(perturb_syn_dist)
                    replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                    
                    theta_old_neighbor_text[synonyms_all[perturb_word_idx][0]] = replacement
                    # pr = get_attack_result([theta_old_neighbor_text], predictor, orig_label, batch_size)
                    pr = get_attack_result(theta_old_neighbor_text, batch, predictor, orig_label, tokenizer, stop_str)
                    qrs+=1

                    if qrs > budget:
                        sim = best_sim[0]
                        max_changes = 0
                        for i in range(len(text_ls)):
                            if text_ls[i]!=best_attack[i]:
                                max_changes+=1
                        # b_a = batch
                        # b_a['desc'] = best_attack
                        return ' '.join(best_attack), max_changes, len(changed_indices), \
                            orig_label, get_predictor_results(best_attack, batch, predictor, tokenizer, stop_str), qrs, sim, random_sim

                    if np.sum(pr)>0:
                        break

                if np.sum(pr)>0:
                    sim_new = calc_sim(text_ls, [theta_old_neighbor_text], -1, sim_score_window, sim_predictor)
                    derivative = (sim_old-sim_new)/0.5

                    g_hat = derivative*u_vec

                    theta_new = theta_old-0.3*(g_hat+2*l2_lambda*theta_old)

                    if sim_new > sim_old:
                        best_attack = theta_old_neighbor_text
                        best_sim = sim_new

                    theta_perturb_dist = np.sum((theta_new)**2, axis=1)
                    nonzero_ele = np.nonzero(np.linalg.norm(theta_new,axis = -1))[0].tolist()
                    perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])

                    theta_new_text = text_ls[:]
                    for perturb_idx in range(len(nonzero_ele)):
                        perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                        word_dict_idx = words_perturb_idx[perturb_word_idx]
                        
                        perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_new[perturb_word_idx]
                        syn_feat_set = []
                        for syn in synonyms_all[perturb_word_idx][1]:
                            syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                            syn_feat_set.append(syn_feat)

                        perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                        perturb_syn_order = np.argsort(perturb_syn_dist)
                        replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                        
                        theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement
                        # pr = get_attack_result([theta_new_text], predictor, orig_label, batch_size)
                        pr = get_attack_result(theta_new_text, batch, predictor, orig_label, tokenizer, stop_str)
                        qrs+=1

                        if qrs > budget:
                            sim = best_sim[0]
                            max_changes = 0
                            for i in range(len(text_ls)):
                                if text_ls[i]!=best_attack[i]:
                                    max_changes+=1
                            # b_att = batch
                            # b_att['desc'] = best_attack
                            return ' '.join(best_attack), max_changes, len(changed_indices), \
                                orig_label, get_predictor_results(best_attack, batch, predictor, tokenizer, stop_str), qrs, sim, random_sim

                        if np.sum(pr)>0:
                            break
                    if np.sum(pr)>0:
                        sim_theta_new = calc_sim(text_ls, [theta_new_text], -1, sim_score_window, sim_predictor)
                        if sim_theta_new > best_sim:
                            best_attack = theta_new_text
                            best_sim = sim_theta_new

                    if np.sum(pr)>0:

                        gamma_old_text = theta_new_text
                        gamma_sim_full = calc_sim(text_ls, [gamma_old_text], -1, sim_score_window, sim_predictor)
                        gamma_old_adv_embed = []
                        for idx in words_perturb_doc_idx:
                            gamma_old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[gamma_old_text[idx]]].strip().split()[1:]])
                        gamma_old_adv_embed_matrix = np.asarray(gamma_old_adv_embed)

                        gamma_old_pert= gamma_old_adv_embed_matrix-words_perturb_embed_matrix
                        gamma_old_pert_divided =gamma_old_pert/gamma
                        perturb_gradient = []
                        for i in range(gamma.shape[0]):
                            idx = words_perturb_doc_idx[i]
                            replaceback_text = gamma_old_text[:]
                            replaceback_text[idx] = text_ls[idx]
                            replaceback_sims = calc_sim(text_ls, [replaceback_text], -1, sim_score_window, sim_predictor)
                            gradient_2 = soft_threshold(l1,gamma[i][0])
                            gradient_1 = -((gamma_sim_full-replaceback_sims)/(gamma[i]+1e-4))[0]
                            gradient = gradient_1+gradient_2
                            gamma[i]=gamma[i]-0.05*gradient


                        theta_new = gamma_old_pert_divided * gamma
                        theta_perturb_dist = np.sum((theta_new)**2, axis=1)
                        nonzero_ele = np.nonzero(np.linalg.norm(theta_new,axis = -1))[0].tolist()
                        perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])
                        theta_new_text = text_ls[:]
                        for perturb_idx in range(len(nonzero_ele)):
                            perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                            word_dict_idx = words_perturb_idx[perturb_word_idx]
                            
                            perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_new[perturb_word_idx]
                            syn_feat_set = []
                            for syn in synonyms_all[perturb_word_idx][1]:
                                syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                                syn_feat_set.append(syn_feat)

                            perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                            perturb_syn_order = np.argsort(perturb_syn_dist)
                            replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                            
                            theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement
                            pr = get_attack_result(theta_new_text, batch, predictor, orig_label, tokenizer, stop_str)
                            qrs+=1

                            if qrs > budget:
                                sim = best_sim[0]
                                max_changes = 0
                                for i in range(len(text_ls)):
                                    if text_ls[i]!=best_attack[i]:
                                        max_changes+=1
                                return ' '.join(best_attack), max_changes, len(changed_indices), \
                                    orig_label, get_predictor_results(best_attack, batch, predictor, tokenizer, stop_str), qrs, sim, random_sim

                            if np.sum(pr)>0:
                                break

                    
                        if np.sum(pr)>0:
                            sim_theta_new = calc_sim(text_ls, [theta_new_text], -1, sim_score_window, sim_predictor)
                            if sim_theta_new > best_sim:
                                best_attack = theta_new_text
                                best_sim = sim_theta_new

            sim = best_sim[0]
            max_changes = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=best_attack[i]:
                    max_changes+=1
            return ' '.join(best_attack), max_changes, len(changed_indices), \
                  orig_label, get_predictor_results(best_attack, batch, predictor, tokenizer, stop_str), qrs, sim, random_sim

            

        else:
            print("Not Found")
            return '', 0,0, orig_label, orig_label, 0, 0, 0

# helper functions for llaga
def get_sbert_embedding(texts, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN', 'graphprompter', 'llaga'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default="counter-fitted-vectors.txt",
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--atk_output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    ## model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
    parser.add_argument("--fuzz",
                        default=0,
                        type=int,
                        help="Word Pruning Value")
    parser.add_argument("--top_k_words",
                        default=1000000,
                        type=int,
                        help="Top K Words")
    parser.add_argument("--allowed_qrs",
                        default=100000,
                        type=int,
                        help="Allowerd qrs")

    
    parser.add_argument("--budget",
                        type=int,
                        required=True,
                        help="Number of Budget Limit")

    parser.add_argument("--graphllm_config_file", 
                        type = str,
                        help = "config file for graphllm")
    parser.add_argument("--sampling_portion", 
                        default = 0.1,
                        required = True,
                        type = float,
                        help = "porportion of test data to be attacked")



    args = parser.parse_args()

    with open(args.graphllm_config_file, 'r') as file:
        graphllm_config = json.load(file)

    print(graphllm_config)

    for key, value in graphllm_config.items():
        if hasattr(args, key):
            print(f"Warning: Attribute '{key}' already exists in args. Skipping this key from config.")
        else:
            setattr(args, key, value)

    if os.path.exists(args.atk_output_dir) and os.listdir(args.atk_output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.atk_output_dir))
    else:
        os.makedirs(args.atk_output_dir, exist_ok=True)

    print("Building LLaGA Model...")

    if args.target_model == 'llaga':
        # step 1: load victim model
        disable_torch_init()
        
        model_name = get_model_name_from_path(args.model_path)
        print(f"Loaded from {args.model_path}. Model Base: {args.model_base}")
        tokenizer, model, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                            cache_dir=args.cache_dir)
        model = model.to(torch.float16).cuda()


        # step 2: load data 
        if args.dataset == "arxiv_sup":
            data_dir = "/scratch/qz2086/TrustGLM/Baselines/LLaGA/dataset/sup/ogbn-arxiv_sup"
        elif args.dataset == "products_sup":
            data_dir = "/scratch/qz2086/TrustGLM/Baselines/LLaGA/dataset/sup/ogbn-products_sup"
        elif args.dataset == "pubmed_sup":
            data_dir = "/scratch/qz2086/TrustGLM/Baselines/LLaGA/dataset/sup/pubmed_sup"
        elif args.dataset == "cora_sup":
            data_dir = "/scratch/qz2086/TrustGLM/Baselines/LLaGA/dataset/sup/cora_sup"
        else:
            print(f"{args.dataset} not exists")


        pretrained_emb = torch.load(os.path.join(data_dir, f"{args.pretrained_embedding_type}_x.pt"))
        print(f'pretrained emb shape: {pretrained_emb.shape}')

        structure_emb = torch.load(
                f"/scratch/qz2086/TrustGLM/Baselines/LLaGA/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")
        print(f'structure emb shape: {structure_emb.shape}')

        prompt_file = args.prompt_file
        print(f"Load from {prompt_file}\n")
        lines = open(prompt_file, "r").readlines()

        
        # load data
        data_path = os.path.join(data_dir, f"processed_data.pt")
        data = torch.load(data_path)

        
        # answer file
        answers_file = os.path.expanduser(args.output_dir)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        if "tmp" not in answers_file and os.path.exists(answers_file):
            line_number = len(open(answers_file, 'r').readlines())
            print(f"{answers_file} already exists! it has {line_number} lines!!")
            if line_number >= len(lines):
                return
            lines = lines[line_number:]
            ans_file = open(answers_file, "a")
        else:
            ans_file = open(answers_file, "w")

        questions = [json.loads(q) for q in lines]

    # predictor = model.inference
    with torch.inference_mode():
        predictor = model.generate
        print("LLaGA Model built!")

        idx2word = {}
        word2idx = {}
        sim_lis=[]

        print("Building vocab...")
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1

        print("Building cos sim matrix...")
        if args.counter_fitting_cos_sim_path:
            print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
            with open(args.counter_fitting_cos_sim_path, "rb") as fp:
                sim_lis = pickle.load(fp)
        else:
            print('Start computing the cosine similarity matrix!')
            embeddings = []
            with open(args.counter_fitting_embeddings_path, 'r') as ifile:
                for line in ifile:
                    embedding = [float(num) for num in line.strip().split()[1:]]

                    embeddings.append(embedding)

            embeddings = np.array(embeddings,dtype='float64')
            embeddings = embeddings[:30000]


            print(embeddings.T.shape)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.asarray(embeddings / norm, "float64")
            cos_sim = np.dot(embeddings, embeddings.T)

        print("Cos sim import finished!")

        use = USE(args.USE_cache_path)
        print('cache path')
        orig_failures = 0.
        adv_failures = 0.
        avg=0.
        tot = 0
        changed_rates = []
        nums_queries = []
        orig_texts = []
        adv_texts = []
        true_labels = []
        new_labels = []
        wrds=[]
        s_queries=[]
        f_queries=[]
        success=[]
        results=[]
        fails=[]
        final_sims = []
        random_sims = []
        random_changed_rates = []
        print('before log dir')

        log_dir = "results_hard_label/"+args.target_model+"/"+args.dataset
        res_dir = "results_hard_label/"+args.target_model+"/"+args.dataset
        log_file = "results_hard_label/"+args.target_model+"/"+args.dataset+"log.txt"
        result_file = "results_hard_label/"+args.target_model+"/"+args.dataset+"results_final.csv"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        stop_words_set = criteria.get_stopwords()
        print('Start attacking!')

        # sample an arbitraty number of data for testing
        total_batches = len(questions)
        print('total test data size:', total_batches, flush = True)
        num_to_attack = math.ceil(args.sampling_portion * total_batches)
        print(print('sampled number to attack:', num_to_attack, flush = True))
        sample_indices = sorted(random.sample(range(total_batches), num_to_attack))

        for line in tqdm(questions, desc = 'processing samples'):
            # if idx % 20 == 0:
            # print(line, flush = True)
            idx = line["id"]
            if idx not in sample_indices:
                print('skipped')
                continue
            print(len(success), flush = True)
            print(np.mean(changed_rates), flush = True)
            
            print('currently attacking node:', idx, flush = True)
            text = data.raw_texts[idx]
            label_idx = data.y[idx]
            true_label = data.label_texts[label_idx]
            print(text, flush = True)
            print(true_label, flush = True)
            # print(true_label)

            qs = line["conversations"][0]['value']
            cur_prompt = qs
            print("prompt:", cur_prompt)
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            # print("input ids", input_ids, flush = True)
            if not isinstance(line['graph'][0], list):
                line['graph'] = [line['graph']]

            graph = torch.LongTensor(line['graph'])
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask]]
            s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((s, n, d))
            graph_emb[mask] = masked_graph_emb

            batch = (input_ids, pretrained_emb, structure_emb, # graph_emb
                    graph.cuda(), # graph
                    True, # do_sample
                    args.temperature,
                    args.num_beams,
                    1024, # max_new_tokens
                    True, idx,graph, mask)

            if structure_emb is not None:
                graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            output_ids = predictor(input_ids, graph_emb=graph_emb.half().cuda(), graph=graph.cuda(), do_sample=True,temperature=args.temperature,
                                        num_beams=args.num_beams,
                                        max_new_tokens=1024, # max_new_tokens
                                        use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            original_label = outputs.strip()
            
            # do the attack
            new_text, num_changed, random_changed, orig_label, \
            new_label, num_queries, sim, random_sim = texthoaxer_attack(args.fuzz,args.top_k_words,args.allowed_qrs,input_ids, batch, true_label, original_label, text, predictor, tokenizer, stop_str,
                                                stop_words_set,
                                                word2idx, idx2word, sim_lis, sim_predictor=use,
                                                sim_score_threshold=args.sim_score_threshold,
                                                import_score_threshold=args.import_score_threshold,
                                                sim_score_window=args.sim_score_window,
                                                synonym_num=args.synonym_num,
                                                batch_size=args.batch_size,embed_func = args.counter_fitting_embeddings_path,budget=args.budget)

            if true_label != orig_label:
                orig_failures += 1
            else:
                nums_queries.append(num_queries)

            if true_label != new_label:
                adv_failures += 1

            changed_rate = 1.0 * num_changed / len(text)
            random_changed_rate = 1.0 * random_changed / len(text)
            if true_label == orig_label and true_label != new_label:
                temp=[]
                s_queries.append(num_queries)
                success.append(idx)
                changed_rates.append(changed_rate)
                orig_texts.append(' '.join(text))
                adv_texts.append(new_text)
                true_labels.append(true_label)
                new_labels.append(new_label)
                random_changed_rates.append(random_changed_rate)
                random_sims.append(random_sim)
                final_sims.append(sim)
                temp.append(idx)
                temp.append(orig_label)
                temp.append(new_label)
                temp.append(' '.join(text))
                temp.append(new_text)
                temp.append(num_queries)
                temp.append(random_sim)
                temp.append(sim)
                temp.append(changed_rate * 100)
                temp.append(random_changed_rate * 100)
                results.append(temp)
                print("Attacked: "+str(idx))
            if true_label == orig_label and true_label == new_label:
                f_queries.append(num_queries)
                temp1=[]
                temp1.append(idx)
                temp1.append(' '.join(text))
                temp1.append(new_text)
                temp1.append(num_queries)
                fails.append(temp1)
        
       
        message =  'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
                'avg changed rate: {:.3f}%, num of queries: {:.1f}, random_sims: {:.3f}, final_sims : {:.3f} \n'.format(
                                                                        (1-orig_failures/1000)*100,
                                                                        (1-adv_failures/1000)*100,
                                                                        np.mean(random_changed_rates)*100,
                                                                        np.mean(changed_rates)*100,
                                                                        np.mean(nums_queries),
                                                                        np.mean(random_sims),
                                                                        np.mean(final_sims))
        print(message)
        

        log=open(log_file,'a')
        log.write(message)
        with open(result_file,'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(results)

    

        with open(os.path.join(args.atk_output_dir, 'adversaries.txt'), 'w') as ofile:
            for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
                ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

if __name__ == "__main__":
    main()





