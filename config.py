import os
from sacred import Experiment

ex = Experiment("DVTA", save_git_info=False)

@ex.config
def base_config():
    # --- General Settings ---
    seed = 0
    device_id = 5  # GPU device ID to use
    
    # --- Experiment Tracking ---
    # 'sota': For comparing with state-of-the-art methods using their specific data splits.
    # 'custom': For running experiments on custom/random splits.
    track = "sota" 
    
    # --- Dataset & Paths ---
    # IMPORTANT: Change 'data_root' to the absolute path of your data directory.
    data_root = "/home1/kuangjidong/Methods/SMIE/data" 
    dataset_name = "ntu120"  # 'ntu60', 'ntu120', 'pku'
    output_root = "output"

    # --- SOTA Track Settings (if track == 'sota') ---
    # Define the number of unseen classes for SOTA splits
    # 'ntu60': '5', '12' | 'ntu120': '10', '24' 
    sota_unseen_split = "24" 
    
    # --- Custom Track Settings (if track == 'custom') ---
    custom_unseen_split = "6" # Example: '1', '2', ..., '9'

    # --- Model Hyperparameters ---
    skeleton_feat_dim = 256  # Dimension of visual features from skeleton encoder
    text_feat_dim = 768      # Dimension of text features from language model
    leaky_sigmoid_alpha = 0.01 # Leakiness for the LeakySigmoid in the AA module
    
    # --- Training Hyperparameters ---
    epochs = 100
    batch_size = 128
    if dataset_name == "ntu60":
        learning_rate = 1e-6  
    elif dataset_name == "ntu120":
        learning_rate = 5e-6
    if dataset_name == "pku":
        learning_rate = 1e-5
    weight_decay = 0
    temperature = 0.1 # Learnable temperature parameter tau
    
    if track == "sota":
        # SOTA-specific paths and unseen labels
        sota_splits_unseen_labels = {
            'ntu60': {'5': [10,11,19,26,56], '12': [3,5,9,12,15,40,42,47,51,56,58,59]},
            'ntu120': {'10': [4,13,37,43,49,65,88,95,99,106], '24': [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]},
        }
        unseen_labels = sota_splits_unseen_labels[dataset_name][str(sota_unseen_split)]
        data_path_prefix = os.path.join(data_root, 'sota', f'split_{sota_unseen_split}')
        train_list_path = os.path.join(data_path_prefix, 'train.npy')
        train_label_path = os.path.join(data_path_prefix, 'train_label.npy')
        test_list_path = os.path.join(data_path_prefix, 'test.npy')
        test_label_path = os.path.join(data_path_prefix, 'test_label.npy')
        
        # Skeleton encoder is not used in SOTA track as features are pre-extracted
        skeleton_encoder_path = None 
        
        # Output directory for the experiment
        exp_name = f"{dataset_name}_split_{sota_unseen_split}"
        output_dir = os.path.join(output_root, track, exp_name)
    
    else: # track == 'custom'
        # Custom split paths and unseen labels
        custom_splits_unseen_labels = {
            'ntu60': {
                '1': [4,19,31,47,51], '2': [12,29,32,44,59], '3': [7,20,28,39,58]
            },
            'ntu120': {
                '4': [3, 18, 26, 38, 41, 60, 87, 99, 102, 110],
                '5': [5, 12, 14, 15, 17, 42, 67, 82, 100, 119],
                '6': [6, 20, 27, 33, 42, 55, 71, 97, 104, 118]
            },
            'pku':{
                   '7': [1, 9, 20, 34, 50],
                   '8': [3, 14, 29, 31, 49],
                   '9': [2, 15, 39, 41, 43]
            }
        }
        unseen_labels = custom_splits_unseen_labels[dataset_name][str(custom_unseen_split)]
        data_path_prefix = os.path.join(data_root, 'zeroshot', dataset_name, f'split_{custom_unseen_split}')
        train_list_path = os.path.join(data_path_prefix, 'seen_train_data.npy')
        train_label_path = os.path.join(data_path_prefix, 'seen_train_label.npy')
        test_list_path = os.path.join(data_path_prefix, 'unseen_data.npy')
        test_label_path = os.path.join(data_path_prefix, 'unseen_label.npy')

        # Path to the pre-trained skeleton encoder (e.g., Shift-GCN)
        skeleton_encoder_path = f"./encoders/gcn/model/split_{custom_unseen_split}.pt"

        # Output directory for the experiment
        exp_name = f"{dataset_name}_split_{custom_unseen_split}"
        output_dir = os.path.join(output_root, track, exp_name)

    # Paths for language embeddings

    text_feat_path = os.path.join(data_root, "language", f"{dataset_name}_embeddings.npy")
    context_text_feat_path = os.path.join(data_root, "language", f"{dataset_name}_cont_embeddings.npy")
    
    # Final log and model save paths
    log_path = os.path.join(output_dir, "run.log")
    model_save_path = os.path.join(output_dir, "best_model.pt")
    
    # GCN parameters (only for custom track)
    gcn_params = {
        "in_channels": 3,
        "hidden_channels": 16,
        "hidden_dim": 256,
        "dropout": 0.5,
        "graph_args": {"layout": 'ntu-rgb+d', "strategy": 'spatial'},
        "edge_importance_weighting": True
    }