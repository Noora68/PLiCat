import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from esm.tokenization import EsmSequenceTokenizer

# Custom dataset loader to read sequences and labels from a CSV file
class SequenceDataset(Dataset):
    def __init__(self, csv_file, max_len=500, device='cuda'):
        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(csv_file)

        self.tokenizer = EsmSequenceTokenizer()

        # Store max sequence length
        self.max_len = max_len

    def __len__(self):
        # Return total number of samples
        return len(self.data)

    def label_to_vector(self, label_str):
        """
        Convert label string into a vector of integers.
        Example: '1 0 1 0' -> tensor([1,0,1,0])
        """
        return torch.tensor([int(x) for x in label_str.split()], dtype=torch.float)

    def __getitem__(self, idx):
        # Get sequence string from DataFrame
        sequence = self.data.iloc[idx]['Sequence']

        # Get label string and convert to vector
        label_vector = self.label_to_vector(self.data.iloc[idx]['Label'])

        # Encode the protein into token IDs
        input_ids = self.tokenizer.encode(sequence)

        # Truncate if sequence is too long
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Apply padding if sequence is shorter than max_len
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': label_vector,
            'sequence': sequence
        }
