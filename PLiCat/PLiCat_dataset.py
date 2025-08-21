import pandas as pd
import torch
from torch.utils.data import Dataset
from esm.tokenization import EsmSequenceTokenizer
from torch.nn.utils.rnn import pad_sequence
from typing import List


# Custom dataset
class SequenceDataset(Dataset):
    def __init__(self, csv_file, device='cuda'):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = EsmSequenceTokenizer()
        self.device = device

    def __len__(self):
        return len(self.data)

    def label_to_vector(self, label_str):
        return torch.tensor([int(x) for x in label_str.split()], dtype=torch.float)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['Sequence']
        label_vector = self.label_to_vector(self.data.iloc[idx]['Label'])
        input_ids = self.tokenizer.encode(sequence)  # No truncation, no padding
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': label_vector,
            'sequence': sequence
        }


# Dynamic padding within batch
def collate_fn_dynamic_padding(batch):
    # Extract input_ids list from batch
    input_ids_list = [item['input_ids'] for item in batch]
    tokenizer = EsmSequenceTokenizer()
    # Pad to the longest sequence in the batch
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Attention mask: 1 for real tokens, 0 for padding
    attention_masks = (input_ids_padded != tokenizer.pad_token_id).long()

    # Stack labels
    labels = torch.stack([item['labels'] for item in batch])

    # Original sequence strings
    sequences = [item['sequence'] for item in batch]

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks,
        'labels': labels,
        'sequence': sequences
    }


class DiseaseDataset(Dataset):
    """Custom sequence dataset class"""

    def __init__(self, sequences: List[str]):
        """
        Initialize dataset

        Args:
            sequences: List of string sequences
        """
        self.sequences = sequences

    def __len__(self):
        """Return dataset size"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a single sample"""
        return self.sequences[idx]


# Custom dataset
class BindingDataset(Dataset):
    def __init__(self, csv_file, device='cuda'):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = EsmSequenceTokenizer()
        self.device = device

    def __len__(self):
        return len(self.data)

    def label_to_vector(self, label_str):
        return torch.tensor([int(x) for x in label_str.split()], dtype=torch.float)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['protein_Sequence']
        lipid_label_vector = self.label_to_vector(self.data.iloc[idx]['lipid_label'])
        site_label_vector = self.label_to_vector(self.data.iloc[idx]['site_label'])
        binding_site = self.data.iloc[idx]['complex_Binding_Site_Residues_(Re-numbered)']
        input_ids = self.tokenizer.encode(sequence)  # No truncation, no padding
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'lipid_label': lipid_label_vector,
            'site_label': site_label_vector,
            'sequence': sequence,
            'binding_site': binding_site
        }


# Dynamic padding within batch
def binding_collate_fn_dynamic_padding(batch):
    # Extract input_ids list from batch
    input_ids_list = [item['input_ids'] for item in batch]
    tokenizer = EsmSequenceTokenizer()
    # Pad to the longest sequence in the batch
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Attention mask: 1 for real tokens, 0 for padding
    attention_masks = (input_ids_padded != tokenizer.pad_token_id).long()

    # Stack labels
    lipid_labels = torch.stack([item['lipid_label'] for item in batch])
    site_labels = torch.stack([item['site_label'] for item in batch])

    # Original sequence strings
    sequences = [item['sequence'] for item in batch]
    binding_sites = [item['binding_site'] for item in batch]

    return {
        'input_ids': input_ids_padded,
        'lipid_label': lipid_labels,
        'site_label': site_labels,
        'sequence': sequences,
        'binding_site': binding_sites,
        'attention_mask': attention_masks
    }
