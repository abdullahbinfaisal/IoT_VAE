import torch
import matplotlib.pyplot as plt

class LossLogger:
    def __init__(self):
        self.loss_names = [
            'hint1_loss', 'hint2_loss', 'hint3_loss', 'hint4_loss', 'hint5_loss',
            'latent_loss', 'ssim_loss', 'perc_loss', 'epoch_loss'
        ]
        self.losses = {name: [] for name in self.loss_names}

    def log(self, loss_dict):
        # Verify all required keys are present in the dictionary
        missing_keys = set(self.loss_names) - set(loss_dict.keys())
        extra_keys = set(loss_dict.keys()) - set(self.loss_names)
        
        if missing_keys:
            raise ValueError(f"Missing loss keys in dictionary: {missing_keys}")
        if extra_keys:
            print(f"Warning: Extra loss keys in dictionary that won't be logged: {extra_keys}")
        
        for name in self.loss_names:
            self.losses[name].append(loss_dict[name])

    def plot(self):
        plt.figure(figsize=(12, 8))
        for name, values in self.losses.items():
            plt.plot(values, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Losses over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save(self, path):
        torch.save(self.losses, path)

    def load(self, path):
        loaded_losses = torch.load(path)
        # Verify all required keys are present in the loaded dictionary
        missing_keys = set(self.loss_names) - set(loaded_losses.keys())
        if missing_keys:
            raise ValueError(f"Loaded file missing some loss keys: {missing_keys}")
        self.losses = {name: loaded_losses[name] for name in self.loss_names}