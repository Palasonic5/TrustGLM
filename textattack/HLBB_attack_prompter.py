import argparse
import os
import numpy as np
from pathlib import Path
from scipy.special import softmax
np.random.seed(1234)
import pickle
import dataloader
# from train_classifier import Model
from itertools import zip_longest
import criteria
import random
random.seed(0)
import csv
import math
import sys
csv.field_size_limit(sys.maxsize)
# import tensorflow as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import tensorflow as tf
#tf.disable_v2_behavior()
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

# for victim model
from graphprompter.src.utils.seed import seed_everything
from graphprompter.src.config import parse_args_llama
from graphprompter.src.utils.ckpt import _reload_best_model
from graphprompter.src.model import load_model, llama_model_path
from graphprompter.src.dataset import load_dataset
from graphprompter.src.utils.evaluate import eval_funcs
from graphprompter.src.utils.collate import collate_funcs


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
        # Convert input sentences to tensors
        input1 = tf.convert_to_tensor(sents1, dtype=tf.string)
        input2 = tf.convert_to_tensor(sents2, dtype=tf.string)

        default_signature = self.embed.signatures['default']
        sts_encode1 = tf.nn.l2_normalize(default_signature(input1)['default'], axis=1)
        sts_encode2 = tf.nn.l2_normalize(default_signature(input2)['default'], axis=1)

        # Calculate cosine similarity
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
        return sim_scores


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
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
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.
    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
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
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


# It calculates semantic similarity between two text inputs.
# text_ls (list): First text input either original text input or previous text.
# new_texts (list): Updated text inputs.
# idx (int): Index of the word that has been changed.
# sim_score_window (int): The number of words to consider around idx. If idx = -1 consider the whole text.
def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):

    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    # Compute the starting and ending indices of the window.
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
    # Calculate semantic similarity using USE.
    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
            list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))

    return semantic_sims


def get_attack_result(new_text, orig_batch, predictor, orig_label):
    with torch.no_grad():
        new_batch = orig_batch
        if isinstance(new_text[0], str):
            text = " ".join(new_text)
            new_batch['desc'] = [text]
            output = predictor(new_batch)['pred'][0]
            print(orig_label, flush = True)
            print(output, flush = True)
            pr = np.array([orig_label != output], dtype=int)
            return pr
        elif isinstance(new_text[0], list):
            pr = []
            for t in new_text:
                new_text = " ".join(t)
                new_batch['desc'] = [new_text]
                output = predictor(new_batch)['pred'][0]
                print(orig_label, flush = True)
                print(output, flush = True)
                pr_t = 1 if orig_label != output else 0
                pr.append(pr_t)
            pr = np.array(pr)
            return pr


# It changes the input text at the specified index.
# rand_idx (int): Index to be mutated.
# text_ls (list): Original text.
# pos_ls (list): POS tage list.
# new_attack (list): The changed text during genetic optimization.
# best_attack (list): The best attack until now.
# remaining_indices (list): The indices in text input different from original input.
# synonyms_dict (dict): Synonym dict for each word.
# orig_label (int): Original prediction of the target model.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# batch_size (int): batch size.
def mutate(rand_idx, text_ls, pos_ls, batch, new_attack, best_attack, remaining_indices,
           synonyms_dict, old_syns, orig_label, sim_score_window,
           predictor, sim_predictor, batch_size):

    # Calculates the semantic similarity before mutation.
    random_text = new_attack[:]
    syns = synonyms_dict[text_ls[rand_idx]]
    prev_semantic_sims = calc_sim(text_ls, [best_attack], rand_idx, sim_score_window, sim_predictor)
    # Gives Priority to Original Word
    orig_word = 0
    if random_text[rand_idx] != text_ls[rand_idx]:

        temp_text = random_text[:]
        temp_text[rand_idx] = text_ls[rand_idx]
        # print(temp_text)
        pr = get_attack_result(temp_text, batch, predictor, orig_label)
        semantic_sims = calc_sim(text_ls, [temp_text], rand_idx, sim_score_window, sim_predictor)
        if np.sum(pr) > 0:
            orig_word = 1
            return temp_text, 1  #(updated_text, queries_taken)

    # If replacing with original word does not yield adversarial text, then try to replace with other synonyms.
    if orig_word == 0:
        final_mask = []
        new_texts = []
        final_texts = []

        # Replace with synonyms.
        for syn in syns:

            # Ignore the synonym already present at position rand_idx.
            if syn == best_attack[rand_idx]:
                final_mask.append(0)
            else:
                final_mask.append(1)
            temp_text = random_text[:]
            temp_text[rand_idx] = syn
            new_texts.append(temp_text[:])

        # Filter out mutated texts that: (1) are not having same POS tag of the synonym, (2) lowers Semantic Similarity and (3) Do not satisfy adversarial criteria.
        synonyms_pos_ls = [criteria.get_pos(new_text[max(rand_idx - 4, 0):rand_idx + 5])[min(4, rand_idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[rand_idx] for new_text in new_texts]
        pos_mask = np.array(criteria.pos_filter(pos_ls[rand_idx], synonyms_pos_ls))
        semantic_sims = calc_sim(text_ls, new_texts, rand_idx, sim_score_window, sim_predictor)
        pr = get_attack_result(new_texts, batch, predictor, orig_label)
        # print('new texts', new_texts, flush = True)
        final_mask = np.asarray(final_mask)

        sem_filter = semantic_sims >= prev_semantic_sims
        sem_filter = sem_filter.numpy()
        sem_filter = np.full(pos_mask.shape, sem_filter)
        prediction_filter = pr > 0
        print(prediction_filter, flush = True)
        final_mask = final_mask*sem_filter
        final_mask = final_mask*prediction_filter
        final_mask = final_mask*pos_mask
        sem_vals = final_mask*semantic_sims
        print(final_mask, flush = True)
        print(sem_vals, flush = True)

        for i in range(len(sem_vals)):
            if sem_vals[i] > 0:
                final_texts.append((new_texts[i], sem_vals[i]))

        # Return mutated text with best semantic similarity.
        final_texts.sort(key =  lambda x : x[1])
        final_texts.reverse()
        print('final texts', final_texts)

        if len(final_texts) > 0:
            #old_syns[rand_idx].append(final_texts[0][0][rand_idx])
            return final_texts[0][0], len(new_texts)
        else:
            return [], len(new_texts)

# It generates children texts from the parent texts using crossover.
# population_size (int): Size of population used.
# population (list): The population currently in the optimization process.
# parent1_idx (int): The index of parent text input 1.
# parent2_idx (int): The index of parent text input 2.
# text_ls (list): Original text.
# best_attack (list): The best attack until now in the optimization.
# max_changes (int): The number of words substituted in the best_attack.
# changed_indices (list): The indices in text input different from original input.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# orig_label (int): Original prediction of the target model.
# batch_size (int): batch size.
def crossover(batch, population_size, population, parent1_idx, parent2_idx,
              text_ls, best_attack, max_changes, changed_indices,
              sim_score_window, sim_predictor,
              predictor, orig_label, batch_size):

    childs = []
    changes = []

    # Do crossover till population_size-1.
    for i in range(population_size-1):

        # Generates new child.
        p1 = population[parent1_idx[i]]
        p2 = population[parent2_idx[i]]

        assert len(p1) == len(p2)
        new_child = []
        for j in range(len(p1)):
            if np.random.uniform() < 0.5:
                new_child.append(p1[j])
            else:
                new_child.append(p2[j])
        change = 0
        cnt = 0
        mismatches = 0
        # Filter out crossover child which (1) Do not improve semantic similarity, (2) Have number of words substituted
        # more than the current best_attack.
        for k in range(len(changed_indices)):
            j = changed_indices[k]
            if new_child[j] == text_ls[j]:
                change+=1
                cnt+=1
            elif new_child[j] == best_attack[j]:
                change+=1
                cnt+=1
            elif new_child[j] != best_attack[j]:
                change+=1
                prev_semantic_sims = calc_sim(text_ls, [best_attack], j, sim_score_window, sim_predictor)
                semantic_sims = calc_sim(text_ls, [new_child], j, sim_score_window, sim_predictor)
                if semantic_sims >= prev_semantic_sims:
                    mismatches+=1
                    cnt+=1
        if cnt==change and mismatches<=max_changes:
            childs.append(new_child)
        changes.append(change)
    if len(childs) == 0:
        return [], 0

    # Filter out childs whoch do not satisfy the adversarial criteria.
    pr = get_attack_result(childs, batch, predictor, orig_label)
    print('pr in crossover', pr, flush = True)
    final_childs = [childs[i] for i in range(len(pr)) if pr[i] > 0]
    return final_childs, len(final_childs)

def attack(fuzz_val, top_k_words, qrs, sample_index, batch, true_label,
           predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):

    # first check the prediction of the original text
    orig_label = predictor(batch)['pred'][0]
    # print(batch)
    text_ls = batch['desc'][0].split()
    print('true label', true_label)
    print('original label', orig_label)
    # orig_label = torch.argmax(orig_probs)
    # orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0, 0, 0
    else:

        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}
        # get the pos and verb tense info
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)
        # find synonyms and make a dict of synonyms of each word.
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
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

        # STEP 1: Random initialisation.
        qrs = 0
        num_changed = 0
        flag = 0
        th = 0
        # Try substituting a random index with its random synonym.
        while qrs < len(text_ls):
            random_text = text_ls[:]
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                if i >= th:
                    break
            pr = get_attack_result(random_text, batch, predictor, orig_label)

            qrs+=1
            th +=1
            if th > len_text:
                break
            if np.sum(pr)>0:
                flag = 1
                break
        old_qrs = qrs
        # If adversarial text is not yet generated try to substitute more words than 30%.
        while qrs < old_qrs + 2500 and flag == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result(random_text, batch, predictor, orig_label)
            qrs+=1
            if np.sum(pr)>0:
                flag = 1
                break

        if flag == 1:
            #print("Found "+str(sample_index))
            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed+=1
            print(changed)

            # STEP 2: Search Space Reduction i.e.  Move Sample Close to Boundary
            while True:
                choices = []

                # For each word substituted in the original text, change it with its original word and compute
                # the change in semantic similarity.
                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        qrs+=1
                        pr = get_attack_result(new_text, batch, predictor, orig_label)
                        if np.sum(pr) > 0:
                            # print(semantic_sims)
                            choices.append((i,semantic_sims))

                # Sort the relacements by semantic similarity and replace back the words with their original
                # counterparts till text remains adversarial.
                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]
                        pr = get_attack_result(new_text, batch, predictor, orig_label)
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

            random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)
            random_t_b = batch
            random_t_b['desc'] = random_text
            if num_changed == 1:
                return ' '.join(random_text), 1, 1, \
                    orig_label, predictor(random_t_b)['pred'][0], qrs, random_sim, random_sim
            population_size = 30
            population = []
            old_syns = {}
            max_replacements = defaultdict(int)
            # STEP 3: Genetic Optimization
            # Genertaes initial population by mutating the substituted indices.
            for i in range(len(changed_indices)):
                txt, mut_qrs = mutate(changed_indices[i], text_ls, pos_ls, batch, random_text, random_text, changed_indices,
                                synonyms_dict, old_syns, orig_label, sim_score_window,
                                predictor, sim_predictor, batch_size)
                qrs+=mut_qrs
                if len(txt)!=0:
                    population.append(txt)
            print('line 733 poluation', population, flush = True)
            max_iters = 100
            pop_count = 0
            attack_same = 0
            old_best_attack = random_text[:]
            new_b = batch
            new_b['desc'] = random_text
            if len(population) == 0:
                return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                            orig_label, predictor(new_b)['pred'][0], qrs, random_sim, random_sim

            ## Genetic Optimization
            for _ in range(max_iters):
                max_changes = len_text
                print('max_changes', max_changes, flush = True)
                len_changed = [len(txt) for txt in population]
                print('len of changed texts', len_changed, flush = True)
                # Find the best_attack text in the current population.
                for txt in population:
                    changes = 0
                    for i in range(len(changed_indices)):
                        j = changed_indices[i]
                        if txt[j] != text_ls[j]:
                            changes+=1
                    if changes <= max_changes:
                        pr = get_attack_result(txt, batch, predictor, orig_label)
                        if pr[0] > 0:
                            max_changes = changes
                            best_attack = txt
                            print('use content in population as new best attack')
                        else:
                            print('population content is not an adversarial example, skipped')
                try:
                    fl = old_best_attack == best_attack
                    print('make sure old best is not the same as current best',fl, flush = True)
                except:
                    print('tried, but code failed')
                print('best attack', best_attack, flush = True)
                pr = get_attack_result(best_attack, batch, predictor, orig_label)
                print(pr, flush = True)
                # Check that it is adversarial.
                assert pr[0] > 0
                flag = 0

                # If the new best attack is the same as the old best attack for last 15 consecutive iterations tham
                # stop optimization.
                for i in range(len(changed_indices)):
                    k = changed_indices[i]
                    if best_attack[k] != old_best_attack[k]:
                        flag = 1
                        break
                if flag == 1:
                    attack_same = 0
                else:
                    attack_same+=1

                if attack_same >= 15:
                    best_attack_b = batch
                    best_attack_b['desc'] = [" ".join(best_attack)]
                    sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)
                    return ' '.join(best_attack), max_changes, len(changed_indices), \
                         orig_label, predictor(best_attack_b)['pred'][0], qrs, sim, random_sim

                old_best_attack = best_attack[:]

                #print(str(max_changes)+" After Genetic")

                # If only 1 input word substituted return it.
                if max_changes == 1:
                    best_a_b = batch
                    best_a_b['desc'] = [" ".join(best_attack)]
                    sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)
                    return ' '.join(best_attack), max_changes, len(changed_indices), \
                         orig_label, predictor(best_a_b)['pred'][0], qrs, sim, random_sim


                # Sample two parent input propotional to semantic similarity.
                sem_scores = calc_sim(text_ls, population, -1, sim_score_window, sim_predictor)
                sem_scores = np.asarray(sem_scores)
                scrs = softmax(sem_scores)

                parent1_idx = np.random.choice(len(population), size = population_size-1, p = scrs)
                parent2_idx = np.random.choice(len(population), size = population_size-1, p = scrs)


                ## Crossover
                final_childs, cross_qrs = crossover(batch, population_size, population, parent1_idx, parent2_idx,
                                         text_ls, best_attack, max_changes, changed_indices, sim_score_window, sim_predictor,
                                         predictor, orig_label, batch_size)
                qrs+=cross_qrs
                population = []
                indices_done = []

                # Randomly select indices for mutation from the changed indices. The changed indices contains indices
                # which has not been replaced by original word.
                indices = np.random.choice(len(changed_indices), size = min(len(changed_indices), len(final_childs)))
                for i in range(len(indices)):
                    child = final_childs[i]
                    j = indices[i]
                    # If the index has been substituted no need to mutate.
                    if text_ls[changed_indices[j]] == child[changed_indices[j]]:
                        population.append(child)
                        indices_done.append(j)
                        continue
                    txt = []
                    # Mutate the childs obtained after crossover on the random index.
                    if max_replacements[changed_indices[j]] <= 25:
                        txt, mut_qrs = mutate(changed_indices[j], text_ls, pos_ls, batch, child, child, changed_indices,
                                            synonyms_dict, old_syns, orig_label, sim_score_window,
                                            predictor, sim_predictor, batch_size)
                    qrs+=mut_qrs
                    indices_done.append(j)

                    # If the input has been mutated successfully add to population for nest generation.
                    if len(txt)!=0:
                        max_replacements[changed_indices[j]] +=1
                        population.append(txt)
                if len(population) == 0:
                    pop_count+=1
                else:
                    pop_count = 0

                # If length of population is zero for 15 consecutive iterations return.
                if pop_count >= 15:
                    sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)
                    bestatt = batch
                    bestatt['desc']  = [" ".join(best_attack)]
                    return ' '.join(best_attack), len(changed_indices), \
                         max_changes, orig_label, predictor(bestatt)['pred'][0], qrs, sim, random_sim

                # Add best adversarial attack text also to next population.
                population.append(best_attack)
            sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)
            bestat = batch
            bestat['desc'] = [" ".join(best_attack)]
            return ' '.join(best_attack), max_changes, len(changed_indices), \
                  orig_label, predictor(bestat)['pred'][0], qrs, sim, random_sim

        else:
            print("Not Found")
            return '', 0,0, orig_label, orig_label, 0, 0, 0


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN', 'graphprompter'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
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

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.6,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--atk_batch_size",
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
    parser.add_argument("--target_dataset",
                        default="imdb",
                        type=str,
                        help="Dataset Name")
    parser.add_argument("--fuzz",
                        default=0,
                        type=int,
                        help="Word Pruning Value")
    parser.add_argument("--top_k_words",
                        default=1000000,
                        type=int,
                        help="Top K Words")
    parser.add_argument("--allowed_qrs",
                        default=1000000,
                        type=int,
                        help="Allowerd qrs")

    parser.add_argument("--graphllm_config_file", 
                        type = str,
                        help = "config file for graphllm")



    args = parser.parse_args()

    with open(args.graphllm_config_file, 'r') as file:
        graphllm_config = json.load(file)

    print(graphllm_config)

    for key, value in graphllm_config.items():
        if hasattr(args, key):
            print(f"Warning: Attribute '{key}' already exists in args. Skipping this key from config.")
        else:
            setattr(args, key, value)


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

# preparing dataset
    # get data to attack
    # texts, labels = dataloader.read_corpus(args.dataset_path,csvf=False)
    # data = list(zip(texts, labels))
    # data = data[:args.data_size] # choose how many samples for adversary
    # print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'graphprompter':
        seed = args.seed
        seed_everything(seed=args.seed)
        print(args)
        # Step 1: load dataset
        print(args.gpu)
        dataset = load_dataset[args.dataset]()
        idx_split = dataset.get_idx_split()

        # Step 2: Build Node Classification Dataset
        test_dataset = [dataset[i] for i in idx_split['test']]
        collate_fn = collate_funcs[args.dataset](dataset.graph)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

        # Step 3: Build Model
        args.llm_model_path = llama_model_path[args.llm_model_name]
        model = load_model[args.model_name](graph=dataset.graph, graph_type=dataset.graph_type, prompt=dataset.prompt, args=args)

        # Step 4: Load Best Model Checkpoint
        model = _reload_best_model(model, args)
        model.eval()
        print('Graphllm loaded successfully!')

    #==================================
    
    predictor = model.inference
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
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
        # load pre-computed cosine similarity matrix if provided
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
        embeddings = np.array(embeddings)
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)

    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
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
    log_dir = "results_hard_label/"+args.target_model+"/"+args.target_dataset
    res_dir = "results_hard_label/"+args.target_model+"/"+args.target_dataset
    log_file = "results_hard_label/"+args.target_model+"/"+args.target_dataset+"/log.txt"
    result_file = "results_hard_label/"+args.target_model+"/"+args.target_dataset+"/results_final.csv"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    stop_words_set = criteria.get_stopwords()
    print('Start attacking!', flush = True)

    for idx, batch in tqdm(enumerate(test_loader), desc = 'Processing samples'):
        # print(idx)
        # print(batch)
        print(str(idx)+" Samples Done", flush = True)
        print(str(len(success)) + "Success attacks", flush = True)
        print(np.mean(changed_rates), flush = True)
        if idx % 20 == 0 and idx > 0 :
            break
        true_label = batch['label'][0]
        print(true_label)
        text = batch['desc'][0].split()

        new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim = attack(args.fuzz,args.top_k_words,args.allowed_qrs,
                                            idx,batch, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, sim_lis , sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.atk_batch_size)

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        if true_label != new_label:
            adv_failures += 1
            #f_queries.append(num_queries)

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

    message = 'For target model using TFIDF {} on dataset window size {} with WP val {} top words {} qrs {} : ' \
              'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, random_sims: {:.3f}%, final_sims : {:.3f}% \n'.format(args.target_model,
                                                                      args.sim_score_window,
                                                                      args.fuzz,
                                                                      args.top_k_words,args.allowed_qrs,
                                                                     (1-orig_failures/1000)*100,
                                                                     (1-adv_failures/1000)*100,
                                                                     np.mean(random_changed_rates)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries),
                                                                     np.mean(random_sims),
                                                                     np.mean(final_sims))
    print(message)
    print(orig_failures)

    log=open(log_file,'a')
    log.write(message)
    with open(result_file,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

    print(avg)
    print(len(f_queries))
    print(f_queries)

    with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

if __name__ == "__main__":
    main()





