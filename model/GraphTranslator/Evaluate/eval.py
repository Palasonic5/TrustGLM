import re
import pandas as pd

def legality_rate(node2pred, dataset):
    if dataset == "cora":
        patterns = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                'Probabilistic_Methods', 'Reinforcement_Learning', 
                'Rule_Learning', 'Theory']
    
        label_map = {
        "Case_Based": 0,
        "Genetic_Algorithms": 1,
        "Neural_Networks": 2,
        "Probabilistic_Methods": 3,
        "Reinforcement_Learning": 4,
        "Rule_Learning": 5,
        "Theory": 6
        }
    elif dataset == "pubmed":
        patterns = ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type1', 'Diabetes Mellitus Type2']
    
        label_map = {
            'Diabetes Mellitus Experimental': 0,
            'Diabetes Mellitus Type1': 1,
            'Diabetes Mellitus Type2': 2
        }
    elif dataset == "products":
        patterns = [
            'Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books',
            'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food',
            'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games',
            'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments',
            'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances',
            'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers',
            'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies',
            'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby',
            'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510'
        ]

        label_map = {
            'Home & Kitchen': 0, 'Health & Personal Care': 1, 'Beauty': 2, 'Sports & Outdoors': 3, 'Books': 4,
            'Patio, Lawn & Garden': 5, 'Toys & Games': 6, 'CDs & Vinyl': 7, 'Cell Phones & Accessories': 8, 'Grocery & Gourmet Food': 9,
            'Arts, Crafts & Sewing': 10, 'Clothing, Shoes & Jewelry': 11, 'Electronics': 12, 'Movies & TV': 13, 'Software': 14, 'Video Games': 15,
            'Automotive': 16, 'Pet Supplies': 17, 'Office Products': 18, 'Industrial & Scientific': 19, 'Musical Instruments': 20,
            'Tools & Home Improvement': 21, 'Magazine Subscriptions': 22, 'Baby Products': 23, 'label 25': 24, 'Appliances': 25,
            'Kitchen & Dining': 26, 'Collectibles & Fine Art': 27, 'All Beauty': 28, 'Luxury Beauty': 29, 'Amazon Fashion': 30, 'Computers': 31,
            'All Electronics': 32, 'Purchase Circles': 33, 'MP3 Players & Accessories': 34, 'Gift Cards': 35, 'Office & School Supplies': 36,
            'Home Improvement': 37, 'Camera & Photo': 38, 'GPS & Navigation': 39, 'Digital Music': 40, 'Car Electronics': 41, 'Baby': 42,
            'Kindle Store': 43, 'Buy a Kindle': 44, 'Furniture & Decor': 45, '#508510': 46
        }
    elif "arxiv" in dataset:
        patterns = [    "cs.NA(Numerical Analysis)", 
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
                        "cs.DM(Discrete Mathematics)"
                    ]
        label_map = {
            "cs.NA(Numerical Analysis)": 0,
            "cs.MM(Multimedia)": 1,
            "cs.LO(Logic in Computer Science)": 2,
            "cs.CY(Computers and Society)": 3,
            "cs.CR(Cryptography and Security)": 4,
            "cs.DC(Distributed, Parallel, and Cluster Computing)": 5,
            "cs.HC(Human-Computer Interaction)": 6,
            "cs.CE(Computational Engineering, Finance, and Science)": 7,
            "cs.NI(Networking and Internet Architecture)": 8,
            "cs.CC(Computational Complexity)": 9,
            "cs.AI(Artificial Intelligence)": 10,
            "cs.MA(Multiagent Systems)": 11,
            "cs.GL(General Literature)": 12,
            "cs.NE(Neural and Evolutionary Computing)": 13,
            "cs.SC(Symbolic Computation)": 14,
            "cs.AR(Hardware Architecture)": 15,
            "cs.CV(Computer Vision and Pattern Recognition)": 16,
            "cs.GR(Graphics)": 17,
            "cs.ET(Emerging Technologies)": 18,
            "cs.SY(Systems and Control)": 19,
            "cs.CG(Computational Geometry)": 20,
            "cs.OH(Other Computer Science)": 21,
            "cs.PL(Programming Languages)": 22,
            "cs.SE(Software Engineering)": 23,
            "cs.LG(Machine Learning)": 24,
            "cs.SD(Sound)": 25,
            "cs.SI(Social and Information Networks)": 26,
            "cs.RO(Robotics)": 27,
            "cs.IT(Information Theory)": 28,
            "cs.PF(Performance)": 29,
            "cs.CL(Computational Complexity)": 30,
            "cs.IR(Information Retrieval)": 31,
            "cs.MS(Mathematical Software)": 32,
            "cs.FL(Formal Languages and Automata Theory)": 33,
            "cs.DS(Data Structures and Algorithms)": 34,
            "cs.OS(Operating Systems)": 35,
            "cs.GT(Computer Science and Game Theory)": 36,
            "cs.DB(Databases)": 37,
            "cs.DL(Digital Libraries)": 38,
            "cs.DM(Discrete Mathematics)": 39
        }
    
    print("Total class number:", len(patterns))
    assert len(patterns) == len(label_map), "patterns and label_map should have the same size"

    # count = 0
    node2digitallabel = {}

    for node, pred_list in node2pred.items():
        matches = set()
        for pred in pred_list:
            for pattern in patterns:
                if re.search(rf'\b{re.escape(pattern)}\b', pred, re.IGNORECASE):
                    matches.add(label_map[pattern])
        
        node2digitallabel[int(node)] = list(matches)
    return node2digitallabel



def read_data(label_file, pred_file):
    df_node2label = pd.read_csv(label_file)
    node2label = dict(zip(df_node2label['node_id'], df_node2label['digital_label']))
    df_pred = pd.read_csv(pred_file, sep='\t', names=['node', 'summary', 'pred'])
    node2pred = {}
    for _, row in df_pred.iterrows():
        node = int(row.iloc[0])
        node2pred[node] = [row.iloc[2].split(".")[0].strip()]
    return node2label, node2pred

def evaluation(label_file, pred_file, dataset):
    node2label, node2pred = read_data(label_file, pred_file)
    node2digitallabel = legality_rate(node2pred, dataset)

    acc_count = 0
    total_count = 0
    count = 0
    longer_count = 0
    for node, pred_list in node2digitallabel.items():
        count += 1
        label = node2label[node]
        if len(pred_list) == 1:
            total_count += 1
            if label == pred_list[0]:
                acc_count += 1
        else:
            longer_count += 1

    accuracy = round(100 * acc_count / count, 2) if count > 0 else 0.0
    print(f"Accuracy: {accuracy}%")
    print("number of samples", count)
    print("longer count:", longer_count)



if __name__ == '__main__':
    label_file = "/scratch/xs2334/TrustGLM/model/GraphTranslator/data/cora/cora_test_0120.csv"
    pred_file = "/scratch/xs2334/TrustGLM/model/GraphTranslator/data/cora/pred_test_0120.txt"
    evaluation(label_file, pred_file, dataset)
