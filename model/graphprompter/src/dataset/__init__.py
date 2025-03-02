from graphprompter.src.dataset.cora import CoraSupDataset, CoraSemiDataset
from graphprompter.src.dataset.citeseer import CiteseerDataset
from graphprompter.src.dataset.pubmed import PubmedSupDataset,PubmedSemiDataset
from graphprompter.src.dataset.arxiv import ArxivSupDataset, ArxivSemiDataset
from graphprompter.src.dataset.products import ProductsSupDataset, ProductsSemiDataset
from graphprompter.src.dataset.sports import SportsSemiDataset, SportsSupDataset
from graphprompter.src.dataset.photo import PhotoSemiDataset, PhotoSupDataset
from graphprompter.src.dataset.computers import ComputersSemiDataset, ComputersSupDataset

load_dataset = {
    'cora_sup': CoraSupDataset,
    'pubmed_sup': PubmedSupDataset,
    'arxiv_sup': ArxivSupDataset,
    'products_sup': ProductsSupDataset,
    'cora_semi': CoraSemiDataset,
    'pubmed_semi': PubmedSemiDataset,
    'arxiv_semi': ArxivSemiDataset,
    'products_semi': ProductsSemiDataset,
    'citeseer': CiteseerDataset,
    'sports_semi': SportsSemiDataset,
    'sports_sup': SportsSupDataset,
    'photo_semi': PhotoSemiDataset,
    'photo_sup': PhotoSupDataset,
    'computers_semi': ComputersSemiDataset,
    'computers_sup': ComputersSupDataset,
}
