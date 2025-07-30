import matplotlib.pyplot as plt
import os

def save_reconstruction_comparison(x, y, epoch, save_dir="reconstructions"):
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare images for visualization
    x_vis = x[0].detach().cpu().permute(1, 2, 0)
    x_recon_vis = y[0].detach().cpu().permute(1, 2, 0)
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    axs[0].imshow(x_vis)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # Plot reconstructed image
    axs[1].imshow(x_recon_vis)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis('off')
    
    # Set main title
    plt.suptitle(f"Reconstruction at Epoch {epoch}")
    
    # Save the figure
    save_path = os.path.join(save_dir, f"reconstruction_epoch_{epoch:04d}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free memory