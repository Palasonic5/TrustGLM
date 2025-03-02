import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from pecos.utils import smat_util
import numpy as np
import random

class ProductsSemiDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/qz2086/hard-label-attack/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        # in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    # 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    # 'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        selected_categories = categories
        # print('selected')
        # random.seed(42)
        # random.shuffle(selected_categories)
        # print('self shuffle', flush = True)
        # print(selected_categories, flush = True)
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"

        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/qz2086/hard-label-attack/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/qz2086/hard-label-attack/graphprompter/dataset/split/semi_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]

        sup_np_filename = f'/scratch/qz2086/hard-label-attack/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(sup_np_filename, allow_pickle=True).item()
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}
        # # Load the saved indices
        # with open('dataset/tape_products/split/train_indices.txt', 'r') as file:
        #     train_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/val_indices.txt', 'r') as file:
        #     val_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/test_indices.txt', 'r') as file:
        #     test_indices = [int(line.strip()) for line in file]
        # return {'train': train_indices, 'val': val_indices, 'test': test_indices}


class ProductsSupDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/qz2086/hard-label-attack/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
    #     in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    # 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    # 'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        selected_categories = categories
        print('all categories + in domain classes, placed later')
        # random.shuffle(selected_categories, seed = 42)
        # print(' shuffle', flush = True)
        # print(selected_categories, flush = True)
        # 格式化类别为1) Category, 2) Category, ... 的形式
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/qz2086/hard-label-attack/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/qz2086/hard-label-attack/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}
        # # Load the saved indices
        # with open('dataset/tape_products/split/train_indices.txt', 'r') as file:
        #     train_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/val_indices.txt', 'r') as file:
        #     val_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/test_indices.txt', 'r') as file:
        #     test_indices = [int(line.strip()) for line in file]
        # return {'train': train_indices, 'val': val_indices, 'test': test_indices}

if __name__ == '__main__':
    dataset = ProductsDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[22], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
