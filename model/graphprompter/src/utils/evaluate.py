import json
import pandas as pd
import re
import argparse
import random

random.seed(42)
def get_accuracy_cora(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    
    for pred, label in zip(df['pred'], df['label']):
        print(f'pred: {pred}')
        matches = re.findall(classes_regex, pred)
        print(f'matches: {matches}')
        print(f'label: {label}')
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


def get_accuracy_pubmed(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_citeseer(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_arxiv(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row)) + '\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        print(f'prediction: {pred}')

        # Remove everything after the first open parenthesis (if any) for cleaner matching
        clean_pred = re.sub(r'\(.*\)', '', pred.strip())
        clean_label = re.sub(r'\(.*\)', '', label.strip())
        print(clean_label)
        matches = re.findall(r"cs\.[a-zA-Z]{2}", clean_pred)

        if len(matches) > 0 and clean_label == matches[0]:
            correct += 1
            print('correct')
        print(f'gt: {clean_label}')
        print('\n')
    print(len(df))
    return correct / len(df)


def get_accuracy_sports(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)
def get_accuracy_computers(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)
def get_accuracy_photo(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)
def get_accuracy_products(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
    classes = ["Home & Kitchen",'Health & Personal Care',
               'Beauty',
               'Sports & Outdoors',
               'Books',
               'Patio, Lawn & Garden',
               'Toys & Games',
               'CDs & Vinyl',
               'Cell Phones & Accessories',
               'Grocery & Gourmet Food',
               'Arts, Crafts & Sewing',
               'Clothing, Shoes & Jewelry',
               'Electronics',
               'Movies & TV',
               'Software',
               'Video Games',
               'Automotive',
               'Pet Supplies',
               'Office Products',
               'Industrial & Scientific',
               'Musical Instruments',
               'Tools & Home Improvement',
               'Magazine Subscriptions',
               'Baby Products',
               'NaN',
               'Appliances',
               'Kitchen & Dining',
               'Collectibles & Fine Art',
               'All Beauty',
               'Luxury Beauty',
               'Amazon Fashion',
               'Computers',
               'All Electronics',
               'Purchase Circles',
               'MP3 Players & Accessories',
               'Gift Cards',
               'Office & School Supplies',
               'Home Improvement',
               'Camera & Photo',
               'GPS & Navigation',
               'Digital Music',
               'Car Electronics',
               'Baby',
               'Kindle Store',
               'Buy a Kindle',
               'Furniture & Decor',
               '#508510']
    # if mode == 'increment':
        # list = [0.1, 0.3, 0.5, 1, 1,5, 2]
        # result = {}
        # for ratio in list:
        #     added_num = len(classes) * ratio
        #     if added_num <= len(in_domain_classes):
        #         added_list = in_domain_classes[:added_num]
        #     else:
        #         added_list = in_domain_classes
        #     random.shuffle(added_list, seed = 42)
        #     classes_new = classes + added_list
        #     random.shuffle(classes_new, seed = 42)
        #     print('incremental random shuffle result for ratio = ', ratio, flush = True)
        #     print('shuffled classes list', flush = True)
        #     print(classes, flush = True)

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1
    this_result = correct/len(df)
    # print(ratio, flush = True)
    print(this_result, flush = True)
    # result[ratio] = this_result
    # print(result, flush = True)
    return this_result

    # random.shuffle(classes, seed = 42)
    # print('cross-domain random shuffle result')
    # print('shuffled classes list:')
    # print(classes)
    # 'Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Method', 'Reinforcement_Learning', 'Rule_Learning', 'Theory', 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 'Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts','Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography',

    

    # return correct/len(df)

eval_funcs = {
    'cora_sup': get_accuracy_cora,
    'citeseer': get_accuracy_citeseer,
    'pubmed_sup': get_accuracy_pubmed,
    'arxiv_sup': get_accuracy_arxiv,
    'products_sup': get_accuracy_products,
    'cora_semi': get_accuracy_cora,
    'pubmed_semi': get_accuracy_pubmed,
    'arxiv_semi': get_accuracy_arxiv,
    'products_semi': get_accuracy_products,
    "sports_semi": get_accuracy_sports,
    "sports_sup": get_accuracy_sports,
    "computers_semi": get_accuracy_computers,
    "computers_sup": get_accuracy_computers,
    "photo_semi": get_accuracy_photo,
    "photo_sup": get_accuracy_photo,
}