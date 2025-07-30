from lib.CLIC_dataset import build_trainloader
import os
import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch
from lib.loss import get_loss_functions
import torch.optim as optim

import NeuralCompression.neuralcompression.functional as ncF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from lib.illm import get_teacher_encoder


def extract_teacher_features(batch_size, quality=3, teacher_model_path=r"D:\IOT_project\models\teacher", 
                           img_dir=r"D:\IOT_project\Datasets\CLIC\val2017",
                           output_dir=r"D:\IOT_project\CLIC_activations\ILLM_Q3_B128_torch"):
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get teacher encoder
    encoder = get_teacher_encoder(quality=quality, path=teacher_model_path)
    
    # Feature storage for hints
    teacher_feats = {}
    
    # Hook function
    def get_teacher_hook(name):
        def hook(module, input, output):
            teacher_feats[name] = output.detach().cpu()
        return hook
    
    # Register hooks on desired hint layers
    encoder.blocks[0].register_forward_hook(get_teacher_hook('hint1'))
    encoder.blocks[1].register_forward_hook(get_teacher_hook('hint2'))
    encoder.blocks[2].register_forward_hook(get_teacher_hook('hint3'))
    encoder.blocks[3].register_forward_hook(get_teacher_hook('hint4'))
    encoder.blocks[4].register_forward_hook(get_teacher_hook('hint5'))
    
    # Build data loader
    train_loader = build_trainloader(batch_size=batch_size, img_dir=img_dir)
    
    # Move encoder to device
    encoder.to(device)
    
    for batch_idx, images in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            images, (_, _) = ncF.pad_image_to_factor(images, 64)
            images = images.to(device)
            
            current_batch_activations = {}
            current_batch_activations['images'] = images.detach().cpu()
            
            y_final = encoder(images)
            current_batch_activations['layer6'] = y_final.detach().cpu()
            
            # Store teacher features
            current_batch_activations['layer1'] = teacher_feats['hint1'].detach().cpu()
            current_batch_activations['layer2'] = teacher_feats['hint2'].detach().cpu()
            current_batch_activations['layer3'] = teacher_feats['hint3'].detach().cpu()
            current_batch_activations['layer4'] = teacher_feats['hint4'].detach().cpu()
            current_batch_activations['layer5'] = teacher_feats['hint5'].detach().cpu()
            
            # Save current batch activations
            torch.save(
                current_batch_activations, 
                os.path.join(output_dir, f'batch_{batch_idx:04d}.pt')
            )
