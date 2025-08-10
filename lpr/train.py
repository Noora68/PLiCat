from torch.utils.data import DataLoader
from weights import compute_class_weights
from LPRdataset import SequenceDataset
import json
from LPRmodel import LPR
from LPRtrain import train_kfold_model
from utils import summarize_kfold_results

# Instantiate the dataset
dataset = SequenceDataset(csv_file = './data/train_dataset.csv')

# Loading label mapping
with open("./data/lipid_9_category.json", "r") as f:
    category_mapping = json.load(f)

# Perform ten-fold cross validation
histories = train_kfold_model(
    model_class=LPR,
    dataset=dataset,
    class_weights=compute_class_weights,
    category_mapping=category_mapping,
    k=10,
    num_epochs=100,
    lr=2e-5,
    batch_size=16,
    early_stop_patience=5,
    save_dir="lpr-models/kfold"
)

# Save the average training results of ten-fold cross validation
summarize_kfold_results(save_dir="lpr-models/kfold", k_folds=10)
