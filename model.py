# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class LeakySigmoid(nn.Module):
    """
    LeakySigmoid activation function used in the Augmented Alignment module.
    It mitigates the vanishing gradient problem for negative inputs.
    """
    def __init__(self, leakiness=0.01):
        super().__init__()
        self.leakiness = leakiness

    def forward(self, x):
        return torch.where(x > 0, torch.sigmoid(x), self.leakiness * torch.exp(x * self.leakiness))

class DeepMetricNetwork(nn.Module):
    """
    The Deep Metric Network (DMN) for the Augmented Alignment (AA) module.
    It learns a similarity score between skeleton and text features.
    Corresponds to the network 'E' in the paper and GlobalDiscriminator_KL_sim in the original code.
    """
    def __init__(self, in_features, leaky_sigmoid_alpha=0.01):
        super().__init__()
        # The original code's in_feature was 768*2
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )
        self.leaky_sigmoid = LeakySigmoid(leaky_sigmoid_alpha)

    def forward(self, skeleton_feat, text_feat):
        combined_feat = torch.cat((skeleton_feat, text_feat), dim=-1)
        score = self.net(combined_feat)
        return self.leaky_sigmoid(score)

class VisualProjector(nn.Module):
    """
    The deep visual projector for skeleton features with a residual connection.
    This corrects the logic that was broken by using nn.Sequential.
    """
    def __init__(self, skeleton_dim, text_dim):
        super().__init__()
        self.l0 = nn.Linear(skeleton_dim, 256)
        self.l1 = nn.Linear(256, 384)
        self.l2 = nn.Linear(384, 512)
        self.l3 = nn.Linear(512, text_dim)

    def forward(self, visual_feat):
        residual = visual_feat
        x = F.relu(self.l0(visual_feat) + residual) 
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        projected_visual = self.l3(x)
        return projected_visual

class DirectAlignment(nn.Module):
    """
    The Direct Alignment (DA) module, including the deep visual projector
    and the Semantic Description Enhancement (SDE) attention mechanism.
    """
    def __init__(self, skeleton_dim, text_dim):
        super().__init__()
        self.visual_projector = VisualProjector(skeleton_dim, text_dim)
        
        self.text_projector = nn.Linear(text_dim, text_dim)

    def forward(self, skeleton_feat, text_feat, is_inference=False):
        projected_skeleton = self.visual_projector(skeleton_feat)

        # SDE: Use skeleton features to attend over text features (label + context)
        if not is_inference:
            attention_scores = torch.bmm(projected_skeleton.unsqueeze(1), text_feat).squeeze(1)
            attention_weights = torch.softmax(attention_scores, dim=1)
            augmented_text = (text_feat * attention_weights.unsqueeze(1)).sum(dim=2)
        else:
            B = skeleton_feat.shape[0]
            U = text_feat.shape[0]
            skeleton_expanded = repeat(projected_skeleton, 'b c -> b u c', u=U)
            text_expanded = repeat(text_feat, 'u c s -> b u c s', b=B)
            attention_scores = torch.einsum('buc, bucs -> bus', skeleton_expanded, text_expanded)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            augmented_text = (text_expanded * attention_weights.unsqueeze(-2)).sum(dim=-1)

        projected_text = self.text_projector(augmented_text)
        return projected_skeleton, projected_text

class DVTA(nn.Module):
    """
    The main Dual Visual-Text Alignment (DVTA) model.
    It combines Direct Alignment (DA) and Augmented Alignment (AA).
    """
    def __init__(self, skeleton_dim, text_dim, temperature, leaky_sigmoid_alpha):
        super().__init__()
        self.direct_alignment = DirectAlignment(skeleton_dim, text_dim)
        self.augmented_alignment = DeepMetricNetwork(in_features=text_dim * 2, leaky_sigmoid_alpha=leaky_sigmoid_alpha)

        self.scaling_factors = nn.Parameter(torch.ones(2) * temperature)

    def forward(self, skeleton_feat, text_feat, is_inference=False):
        # Normalize raw input features
        skeleton_feat = F.normalize(skeleton_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=1) # dim=1 for [B, C, 2] or [U, C, 2]

        # --- 1. Direct Alignment (DA) ---
        skel_feat_da, text_feat_da = self.direct_alignment(skeleton_feat, text_feat, is_inference)
        
        skel_feat_da_norm = F.normalize(skel_feat_da, dim=-1)
        text_feat_da_norm = F.normalize(text_feat_da, dim=-1)
        
        temperature_da = self.scaling_factors[0]
        temperature_aa = self.scaling_factors[1]
        if not is_inference: # --- Training Path ---
            # DA Cosine Similarity Score
            sim_da_v2t = (skel_feat_da_norm @ text_feat_da_norm.t()) / temperature_da
            sim_da_t2v = sim_da_v2t.t()
            
            # --- 2. Augmented Alignment (AA) ---
            # Detach to prevent gradients from flowing back from AA to DA projectors
            skel_feat_aa = skel_feat_da.detach()
            text_feat_aa = text_feat_da.detach()
            
            B = skel_feat_aa.shape[0]
            skel_expanded = repeat(skel_feat_aa, 'b c -> b l c', l=B)
            text_expanded = repeat(text_feat_aa, 'l c -> b l c', b=B)
            
            # AA Similarity Score from Deep Metric Network
            sim_aa_v2t = self.augmented_alignment(skel_expanded, text_expanded).squeeze(-1) / temperature_aa
            sim_aa_t2v = sim_aa_v2t.t()

            return sim_da_v2t, sim_da_t2v, sim_aa_v2t, sim_aa_t2v
        
        else: # --- Inference Path ---

            # DA Cosine Similarity Score (Inference)
            sim_da = torch.einsum('bc,buc->bu', skel_feat_da_norm, text_feat_da_norm) / temperature_da
            
            # AA Similarity Score (Inference)
            B, U, C = text_feat_da.shape
            skel_expanded = repeat(skel_feat_da, 'b c -> b u c', u=U)
            sim_aa = self.augmented_alignment(skel_expanded, text_feat_da).squeeze(-1) / temperature_aa
            
            final_sim = sim_da + sim_aa
            return final_sim