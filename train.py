# train.py
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict

from config import ex
from model import DVTA
from utils import setup_seed, create_ground_truth_matrix, adjust_learning_rate
from logger import Logger 
from data_loader import get_data_loaders 
import os
from encoders.gcn.st_gcn import Model
from KLLoss import KLLoss

class Trainer:
    @ex.capture
    def __init__(self, _config):
        self.config = _config
        self.device = torch.device(f"cuda:{self.config['device_id']}" if torch.cuda.is_available() else "cpu")
        setup_seed(self.config['seed'])

        # Initialize logger
        self.logger = Logger(self.config['log_path'])
        self.logger.info("Configuration loaded:")
        self.logger.info(self.config)

        # Load data
        self.data_loader, self.full_language_features, self.unseen_labels_tensor = get_data_loaders(self.config, self.device)
        self.logger.info("Data loaders and language features created successfully.")

        # Load model components
        self.skeleton_encoder = self.load_skeleton_encoder()
        self.model = self.load_dvta_model()
        self.optimizer = self.load_optimizer()
        self.loss_fn = KLLoss().to(self.device)

        self.logger.info("Model:")
        self.logger.info(self.model)
        
        self.best_acc = -1
        self.best_epoch = -1

    def load_skeleton_encoder(self):
        if self.config['track'] == 'custom' and self.config['skeleton_encoder_path']:
            # Load pre-trained GCN model here for the custom track
            encoder = Model(**self.config['gcn_params'])
            print(f"Loading skeleton encoder from {self.config['skeleton_encoder_path']}")
            encoder.load_state_dict(torch.load(self.config['skeleton_encoder_path'], map_location="cpu"))
            encoder.to(self.device)
            self.logger.info("Pre-trained skeleton encoder loaded.")
            return encoder
        else:
            # SOTA track uses pre-extracted features, no encoder needed during training
            self.logger.info("Using pre-extracted skeleton features. No encoder loaded.")
            return None

    def load_dvta_model(self):
        model = DVTA(
            skeleton_dim=self.config['skeleton_feat_dim'],
            text_dim=self.config['text_feat_dim'],
            temperature=self.config['temperature'],
            leaky_sigmoid_alpha=self.config['leaky_sigmoid_alpha']
        ).to(self.device)
        return model

    def load_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

    def train_one_epoch(self, epoch):
        if self.skeleton_encoder:
            self.skeleton_encoder.eval() # Encoder is frozen
        self.model.train()
        
        total_loss = 0.0
        lr = self.config['learning_rate']
        adjust_learning_rate(self.optimizer, epoch, self.config['epochs'], lr_min=0, lr_max=lr)

        for skeleton_data, labels in tqdm(self.data_loader['train'], desc=f"Epoch {epoch+1}/{self.config['epochs']}"):
            skeleton_data = skeleton_data.to(self.device)
            labels = labels.to(self.device)
            
            # Get features
            if self.skeleton_encoder:
                with torch.no_grad():
                    skeleton_features = self.skeleton_encoder(skeleton_data)
            else: # Pre-extracted features
                skeleton_features = skeleton_data
            
            # Get corresponding text features for the batch
            seen_text_features = self.full_language_features[labels]
            
            # Forward pass through DVTA
            sim_da_v2t, sim_da_t2v, sim_aa_v2t, sim_aa_t2v = self.model(skeleton_features, seen_text_features, is_inference=False)
            
            # Combine scores as per original code logic
            final_sim_v2t = sim_da_v2t + sim_aa_v2t
            final_sim_t2v = sim_da_t2v + sim_aa_t2v
            
            # Create ground truth
            ground_truth = create_ground_truth_matrix(labels).to(self.device)
            
            # Calculate loss
            loss_v2t = self.loss_fn(final_sim_v2t, ground_truth)
            loss_t2v = self.loss_fn(final_sim_t2v, ground_truth.t())
            loss = (loss_v2t + loss_t2v) / 2
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.data_loader['train'])
        self.logger.info(f"Epoch [{epoch+1}] Training Loss: {avg_loss:.4f}")

    def evaluate(self, epoch):
        if self.skeleton_encoder:
            self.skeleton_encoder.eval()
        self.model.eval()
    
        acc_list = []
        unseen_text_features = self.full_language_features[self.unseen_labels_tensor]
        with torch.no_grad():
            for skeleton_data, labels in tqdm(self.data_loader['test'], desc="Evaluating"):
                skeleton_data = skeleton_data.to(self.device)
                labels = labels.to(self.device)

                if self.skeleton_encoder:
                    skeleton_features = self.skeleton_encoder(skeleton_data)
                else:
                    skeleton_features = skeleton_data
                
                # Get similarity scores for all unseen classes
                final_sim_scores = self.model(skeleton_features, unseen_text_features, is_inference=True)
                
                # Get predictions
                _, pred_indices = torch.max(final_sim_scores, 1)
                predictions = self.unseen_labels_tensor[pred_indices]
                
                correct_predictions = (predictions == labels).float().mean().item()
                acc_list.append(correct_predictions)

        acc_list = torch.tensor(acc_list)
        accuracy = acc_list.mean()
        self.logger.info(f"Epoch {epoch+1} Test Accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_epoch = epoch + 1
            self.save_model()
            self.logger.info(f"New best accuracy! Model saved to {self.config['model_save_path']}")
        
        self.logger.info(f"Current Best Accuracy: {self.best_acc * 100:.2f}% at Epoch {self.best_epoch}")

    def save_model(self):
        os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, self.config['model_save_path'])

    def start(self):
        self.logger.info("Starting training...")
        for epoch in range(self.config['epochs']):
            self.train_one_epoch(epoch)
            self.evaluate(epoch)
        self.logger.info("Training finished.")
        self.logger.info(f"Final Best Accuracy: {self.best_acc * 100:.2f}% at Epoch {self.best_epoch}")

@ex.automain
def main(_config):
    trainer = Trainer(_config)
    trainer.start()