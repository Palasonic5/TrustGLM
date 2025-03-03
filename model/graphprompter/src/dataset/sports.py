import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from pecos.utils import smat_util
import numpy as np
import random

class SportsSemiDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/jl11523/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.prompt = "\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing'? \n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
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
        return ['/scratch/jl11523/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/jl11523/graphprompter/dataset/split/semi_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        sup_np_filename = f'/scratch/jl11523/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(sup_np_filename, allow_pickle=True).item()
        test_ids = [int(i) for i in loaded_data_dict['test']]

        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}



class SportsSupDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.prompt = "\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing'? \n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
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

class SportsSupDatasetS(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
        random.seed(42)
        random.shuffle(categories)
        str_categories = ",".join(categories)
        self.prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class SportsSupDatasetLN50(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
        in_domain_class = ['CDs & Vinyl', 'Africa', 'Science, Nature & How It Works', 'Mysteries & Detectives', 'Car Electronics', 'Data StorageHome & Kitchen', 'Animals']
        # str_categories = ",".join(in_domain_class + categories)
        str_categories = ",".join(categories + in_domain_class)
        self.prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class SportsSupDatasetLN100(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
        in_domain_class = ['CDs & Vinyl', 'Africa', 'Science, Nature & How It Works', 'Mysteries & Detectives', 'Car Electronics', 'Data StorageHome & Kitchen', 'Animals', 'Video Games', 'Office Products', 'label 25', 'Magazine Subscriptions', 'Appliances', 'Action & Adventure']
        str_categories = ",".join(categories + in_domain_class)
        self.prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class SportsSupDatasetST(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
            random.shuffle(categories)
            str_categories = ",".join(categories)
            prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class SportsSupDatasetLNT(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
                # 定义类列表
            classes = [
                # Children's labels
                'Literature & Fiction', 'Animals', 'Growing Up & Facts of Life', 'Humor',
                'Cars, Trains & Things That Go', 'Fairy Tales, Folk Tales & Myths',
                'Activities, Crafts & Games', 'Science Fiction & Fantasy', 'Classics',
                'Mysteries & Detectives', 'Action & Adventure', 'Geography & Cultures',
                'Education & Reference', 'Arts, Music & Photography', 'Holidays & Celebrations',
                'Science, Nature & How It Works', 'Early Learning', 'Biographies', 'History',
                "Children's Cookbooks", 'Religions', 'Sports & Outdoors', 'Comics & Graphic Novels',
                'Computers & Technology',

                # Photo labels
                'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video',
                'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes',
                'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography',

                # History labels
                'World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa',
                'Ancient Civilizations', 'Middle East', 'Historical Study & Educational Resources',
                'Australia & Oceania', 'Australia & Oceania ', 'Arctic & Antarctica',

                # Computers labels
                "Monitors", "Networking Products", "Laptop Accessories",
                "Computer Accessories & Peripherals", "Tablet Accessories",
                "Tablet Replacement Parts", "Servers", "Computer Components",
                "Computers & Tablets", "Data Storage"

                # Products labels
                "Home & Kitchen", 
                "Health & Personal Care", 
                "Beauty", 
                "Sports & Outdoors", 
                "Books", 
                "Patio, Lawn & Garden", 
                "Toys & Games", 
                "CDs & Vinyl", 
                "Cell Phones & Accessories", 
                "Grocery & Gourmet Food", 
                "Arts, Crafts & Sewing", 
                "Clothing, Shoes & Jewelry", 
                "Electronics", 
                "Movies & TV", 
                "Software", 
                "Video Games", 
                "Automotive", 
                "Pet Supplies", 
                "Office Products", 
                "Industrial & Scientific", 
                "Musical Instruments", 
                "Tools & Home Improvement", 
                "Magazine Subscriptions", 
                "Baby Products", 
                "label 25", 
                "Appliances", 
                "Kitchen & Dining", 
                "Collectibles & Fine Art", 
                "All Beauty", 
                "Luxury Beauty", 
                "Amazon Fashion", 
                "Computers", 
                "All Electronics", 
                "Purchase Circles", 
                "MP3 Players & Accessories", 
                "Gift Cards", 
                "Office & School Supplies", 
                "Home Improvement", 
                "Camera & Photo", 
                "GPS & Navigation", 
                "Digital Music", 
                "Car Electronics", 
                "Baby", 
                "Kindle Store", 
                "Buy a Kindle", 
                "Furniture & D&#233;cor,", 
                "#508510"
            ]
            shuffled_classes = classes[:]
            random.shuffle(shuffled_classes)
            str_categories=",".join(shuffled_classes[:13]+categories)
            prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class SportsSupDatasetCLNT(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
                # 定义类列表
            classes = [
                # cora labels
                "Case_Based",
                "Genetic_Algorithms",
                "Neural_Networks",
                "Probabilistic_Methods",
                "Reinforcement_Learning",
                "Rule_Learning",
                "Theory",

                # pubmed labels
                "Diabetes Mellitus Experimental",
                "Diabetes Mellitus Type1",
                "Diabetes Mellitus Type2",

                "cs.NA(Numerical Analysis)", 
                "cs.MM(Multimedia)", 
                "cs.LO(Logic in Computer Science)", 
                "cs.CY(Computers and Society)", 
                "cs.CR(Cryptography and Security)", 
                "cs.DC(Distributed, Parallel, and Cluster Computing)", 
                "cs.HC(Human-Computer Interaction)", 
                "cs.CE(Computational Engineering, Finance, and Science)", 
                "cs.NI(Networking and Internet Architecture)", 
                "cs.CC(Computational Complexity)", 
                "cs.AI(Artificial Intelligence)", 
                "cs.MA(Multiagent Systems)", 
                "cs.GL(General Literature)", 
                "cs.NE(Neural and Evolutionary Computing)", 
                "cs.SC(Symbolic Computation)", 
                "cs.AR(Hardware Architecture)", 
                "cs.CV(Computer Vision and Pattern Recognition)", 
                "cs.GR(Graphics)", 
                "cs.ET(Emerging Technologies)", 
                "cs.SY(Systems and Control)", 
                "cs.CG(Computational Geometry)", 
                "cs.OH(Other Computer Science)", 
                "cs.PL(Programming Languages)", 
                "cs.SE(Software Engineering)", 
                "cs.LG(Machine Learning)", 
                "cs.SD(Sound)", 
                "cs.SI(Social and Information Networks)", 
                "cs.RO(Robotics)", 
                "cs.IT(Information Theory)", 
                "cs.PF(Performance)", 
                "cs.CL(Computational Complexity)", 
                "cs.IR(Information Retrieval)", 
                "cs.MS(Mathematical Software)", 
                "cs.FL(Formal Languages and Automata Theory)", 
                "cs.DS(Data Structures and Algorithms)", 
                "cs.OS(Operating Systems)", 
                "cs.GT(Computer Science and Game Theory)", 
                "cs.DB(Databases)", 
                "cs.DL(Digital Libraries)", 
                "cs.DM(Discrete Mathematics)"]
            shuffled_classes = classes[:]
            random.shuffle(shuffled_classes)
            str_categories=",".join(shuffled_classes[:13]+categories)
            prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class SportsSupDatasetLNC50(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
        out_domain_class = [
                "cs.AR(Hardware Architecture)", 
                "cs.NE(Neural and Evolutionary Computing)", 
                "cs.CC(Computational Complexity)", 
                "cs.MM(Multimedia)", 
                "Reinforcement_Learning", 
                "cs.OS(Operating Systems)", 
                "cs.CV(Computer Vision and Pattern Recognition)"
            ]
        str_categories = ",".join(out_domain_class + categories)
        # str_categories = ",".join(categories + out_domain_class)
        self.prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class SportsSupDatasetLNC100(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
        out_domain_class = [
                "cs.AR(Hardware Architecture)", 
                "cs.NE(Neural and Evolutionary Computing)", 
                "cs.CC(Computational Complexity)", 
                "cs.MM(Multimedia)", 
                "Reinforcement_Learning", 
                "cs.OS(Operating Systems)", 
                "cs.CV(Computer Vision and Pattern Recognition)", 
                "Diabetes Mellitus Type2", 
                "cs.SY(Systems and Control)", 
                "cs.HC(Human-Computer Interaction)", 
                "cs.OH(Other Computer Science)", 
                "cs.MA(Multiagent Systems)", 
                "cs.LO(Logic in Computer Science)"
            ]
        str_categories = ",".join(out_domain_class + categories)
        # str_categories = ",".join(categories + out_domain_class)
        self.prompt = f"\nQuestion: Please predict the most appropriate category for this product. Choose from the following categories: {str_categories}? \n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 13
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/amazon-sports/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_amazon-sports.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

if __name__ == '__main__':
    dataset = ProductsDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[22], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
