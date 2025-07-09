# data_loader.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SkeletonDataset(Dataset):
    """A simple dataset class for loading .npy files."""
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path, mmap_mode='r')
        self.label = np.load(label_path, mmap_mode='r')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        return torch.from_numpy(data_numpy).float(), torch.tensor(label).long()

def get_data_loaders(config, device):
    """Creates train and test data loaders."""
    # Load full language features (label + context)
    label_embeds = torch.from_numpy(np.load(config['text_feat_path'])).float()
    context_embeds = torch.from_numpy(np.load(config['context_text_feat_path'])).float()
    # Stack them: shape becomes [Num_classes, D_text, 2]
    full_language_features = torch.stack([label_embeds, context_embeds], dim=-1).to(device)

    # Create datasets
    train_dataset = SkeletonDataset(config['train_list_path'], config['train_label_path'])
    test_dataset = SkeletonDataset(config['test_list_path'], config['test_label_path'])
    
    # Create data loaders
    data_loaders = {
        'train': DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=16,
        ),
        'test': DataLoader(
            dataset=test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=16,
        )
    }
    
    unseen_labels_tensor = torch.tensor(config['unseen_labels']).long().to(device)

    return data_loaders, full_language_features, unseen_labels_tensor