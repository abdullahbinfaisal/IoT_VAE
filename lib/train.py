import torch
from tqdm import tqdm
from lib.loss import get_loss_functions
import NeuralCompression.neuralcompression.functional as ncF
import matplotlib.pyplot as plt
from lib.visualizer import save_reconstruction_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(student, teacher_decoder, optimizer, dataloader, hyperparams, epoch=None, save_dir="reconstructions"):
    
    # Initialize Loss Functions
    msssim_loss, vgg_perceptual, distillation_loss = get_loss_functions()
    vgg_perceptual = vgg_perceptual.to(device)
    
    # Track Losses
    student.train()
    running_loss = 0.0
    total_hint1_loss = 0.0
    total_hint2_loss = 0.0
    total_hint3_loss = 0.0
    total_hint4_loss = 0.0
    total_hint5_loss = 0.0
    total_latent_loss = 0.0
    total_ssim_loss = 0.0
    total_perc_loss = 0.0

    # Send to GPU
    student.to(device)
    teacher_decoder.to(device)
    
    
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch if epoch is not None else ''}")

    for i, batch in loop:
        
        x = batch["images"]
        x = x.to(device)
        # Padding Correction        
        x, (_, _) = ncF.pad_image_to_factor(x, 64) # 64 is ILLM Q3's pad factor
         
        optimizer.zero_grad()

        y = student.block1(x)
        
        y = y.to(device)
        z1 = batch['layer1'].to(device)
        z2 = batch['layer2'].to(device)
        z3 = batch['layer3'].to(device)
        z4 = batch['layer4'].to(device)
        z5 = batch['layer5'].to(device)
        z6 = batch['layer6'].to(device)
        
        
        hint1_loss = distillation_loss(y, z1)

        y = student.block2(y)
        hint2_loss = distillation_loss(y, z2)
        
        y = student.block3(y)
        hint3_loss = distillation_loss(y, z3)
        
        y = student.block4(y)
        hint4_loss = distillation_loss(y, z4)
        
        y = student.block5(y)
        hint5_loss = distillation_loss(y, z5)
        
        y = student.block6(y)
        latent_loss = distillation_loss(y, z6)

        
        y = teacher_decoder(y)

        perc_loss = vgg_perceptual(x, y)
        ssim_loss = msssim_loss(x, y)

        # Cumulative Loss
        loss = (
            hyperparams['alpha_hint1'] * hint1_loss +
            hyperparams['alpha_hint2'] * hint2_loss +
            hyperparams['alpha_hint3'] * hint3_loss +
            hyperparams['alpha_hint4'] * hint4_loss +
            hyperparams['alpha_hint5'] * hint5_loss +
            hyperparams['beta_latent'] * latent_loss +
            hyperparams['gamma_msssim'] * ssim_loss +
            hyperparams['gamma_perc'] * perc_loss
        )

        # Backprop and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate loss values
        running_loss += loss.item() * x.size(0)
        total_hint1_loss += hint1_loss.item() * x.size(0)
        total_hint2_loss += hint2_loss.item() * x.size(0)
        total_hint3_loss += hint3_loss.item() * x.size(0)
        total_hint4_loss += hint4_loss.item() * x.size(0)
        total_hint5_loss += hint5_loss.item() * x.size(0)
        total_latent_loss += latent_loss.item() * x.size(0)
        total_ssim_loss += ssim_loss.item() * x.size(0)
        total_perc_loss += perc_loss.item() * x.size(0)

    # Average losses
    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    
    avg_hint1_loss = total_hint1_loss / dataset_size
    avg_hint2_loss = total_hint2_loss / dataset_size
    avg_hint3_loss = total_hint3_loss / dataset_size
    avg_hint4_loss = total_hint4_loss / dataset_size
    avg_hint5_loss = total_hint5_loss / dataset_size
    
    avg_latent_loss = total_latent_loss / dataset_size
    avg_ssim_loss = total_ssim_loss / dataset_size
    avg_perc_loss = total_perc_loss / dataset_size
    
    
    print("Component-Wise Loss")
    print("Hint 1: ", avg_hint1_loss)
    print("Hint 2: ", avg_hint2_loss)
    print("Hint 3: ", avg_hint3_loss)
    print("Hint 4: ", avg_hint4_loss)
    print("Hint 5: ", avg_hint5_loss)
    print("Latent Loss: ", avg_latent_loss)
    print("SSIM Loss: ", avg_ssim_loss)
    print("VGG Loss: ", avg_perc_loss)
    

    save_reconstruction_comparison(x, y, epoch, save_dir=save_dir)

    return {
        "hint1_loss": avg_hint1_loss,
        "hint2_loss": avg_hint2_loss,
        "hint3_loss": avg_hint3_loss,
        "hint4_loss": avg_hint4_loss,
        "hint5_loss": avg_hint5_loss,
        "latent_loss": avg_latent_loss,
        "ssim_loss": avg_ssim_loss,
        "perc_loss": avg_perc_loss,
        "epoch_loss": epoch_loss
    }
