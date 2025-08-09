import torch
import torch.nn as nn
from transformers import BertModel
from esm.models.esmc import ESMC

# Define the LPR model (a hybrid model combining ESMC and BERT as the main framework + a custom classification head)
# Purpose: For classification tasks on biological sequence data such as protein sequences
class LPR(nn.Module):
    def __init__(self, esmc_unfreeze_last_n=6, bert_unfreeze_last_n=6, num_classes=9):
        """
        Initialize the LPR model

        Parameters:
            esmc_unfreeze_last_n: int
                Number of final transformer blocks in ESMC to unfreeze for fine-tuning (default: 6)
                (Helps adapt to downstream tasks while preserving pretrained knowledge)
            bert_unfreeze_last_n: int
                Number of final transformer layers in BERT to unfreeze (default: 6)
            num_classes: int
                Number of classes for the classification task (default: 9)
        """
        super(LPR, self).__init__()

        self.esmc_unfreeze_last_n = esmc_unfreeze_last_n
        self.bert_unfreeze_last_n = bert_unfreeze_last_n
        self.num_classes = num_classes

        # Load pretrained ESMC model (esmc_300m = 300M parameter version)
        self.esmc = ESMC.from_pretrained("esmc_300m")
        # Convert ESMC model to float32 precision (balances precision and computation efficiency)
        self.esmc = self.esmc.to(torch.float32)

        # Freeze all ESMC parameters initially (to preserve pretrained knowledge)
        for p in self.esmc.parameters():
            p.requires_grad = False  # False = no gradient update

        # Unfreeze the last `esmc_unfreeze_last_n` transformer blocks for fine-tuning
        total_esmc_layers = len(self.esmc.transformer.blocks)  # Get total number of layers
        start_layer = max(0, total_esmc_layers - self.esmc_unfreeze_last_n)

        # Start unfreezing from the last `esmc_unfreeze_last_n` layers
        for i in range(start_layer, total_esmc_layers):
            for param in self.esmc.transformer.blocks[i].parameters():
                param.requires_grad = True  # True = enable gradient update
        # Unfreeze the embedding layer when fully released
        if start_layer == 0:
            for param in self.esmc.embed.parameters():
                param.requires_grad = True

        # Dimensionality reduction: ESMC output (960-d) → BERT input (768-d)
        self.Linear960_768 = nn.Sequential(
            nn.Linear(in_features=960, out_features=768, bias=True),
            nn.GELU(),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        )

        # Load pretrained BERT model (base version, uncased)
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze all BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last `bert_unfreeze_last_n` BERT encoder layers for fine-tuning
        total_bert_layers = len(self.bert.encoder.layer)  # Get total number of BERT encoder layers
        start_layer = max(0, total_bert_layers - self.bert_unfreeze_last_n)

        # Start unfreezing from the last `bert_unfreeze_last_n` layers
        for i in range(start_layer, total_bert_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Unfreeze the embedding layer when fully released
        if start_layer == 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = True

        # Classification head: uses the BERT [CLS] token output for final classification
        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.GELU(),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
            nn.Linear(768, num_classes)  # Output logits for final class prediction
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass (defines data flow)

        Parameters:
            input_ids: torch.Tensor
                Tokenized input sequences, shape: [batch_size, sequence_length]
            attention_mask: torch.Tensor
                Attention mask (1 for valid tokens, 0 for padding), same shape as input_ids

        Returns:
            logits: torch.Tensor
                Raw classification outputs (logits), shape: [batch_size, num_classes]
            cls_output: torch.Tensor
                BERT [CLS] token features, shape: [batch_size, 768]
        """

        # 1. Feature extraction with ESMC
        output = self.esmc(input_ids, attention_mask)

        embeddings = output.embeddings  # Shape: (batch_size, sequence_length, 960)

        # Project to 768-d for BERT
        embeddings = self.Linear960_768(embeddings)

        # Generate position encodings (required by BERT)
        batch_size, seq_len, _ = embeddings.size()  # Get batch size and sequence length

        # Create position indices [0, 1, ..., seq_len-1], expand to all samples in batch
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embeddings.device) \
            .unsqueeze(0).expand(batch_size, seq_len)

        # 2. Further encoding with BERT
        outputs = self.bert(
            inputs_embeds=embeddings,  # Use custom embeddings instead of input_ids
            attention_mask=attention_mask,  # Attention mask
            position_ids=position_ids,  # Position encodings
            return_dict=True  # Return output as a dictionary
        )

        # Extract [CLS] token output (first token used as sequence-level representation)
        cls_output = outputs.last_hidden_state[:, 0]

        # 3. Classification head for final output
        logits = self.classifier(cls_output)

        # Returns regression values and classification embedding vectors
        return logits, cls_output
