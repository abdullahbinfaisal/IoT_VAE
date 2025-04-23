from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow_compression_2.signal_conv_slim import SignalConv2D_slim
from tensorflow_compression_2.gdn_slim_plus import GDN
EntropyModelClass = tfc.entropy_models.PowerLawEntropyModel


def load_image(filename):
    """Loads a JPEG image file."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(string, channels=3)
    # tell TF it’s a 3‑D tensor [H, W, 3]
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def quantize_image(image):
    image = tf.clip_by_value(image, 0, 1)  # Ensure the image values are between 0 and 1
    image = tf.round(image * 255)           # Quantize to the range [0, 255]
    image = tf.cast(image, tf.uint8)        # Cast to uint8 for 8-bit representation
    return image


def save_image(filename, image):
    """Saves an image to a JPEG file."""
    image = quantize_image(image)           # 0–255 uint8
    string = tf.image.encode_jpeg(image)    # JPEG instead of PNG
    return tf.io.write_file(filename, string)


# Create a proper Keras model
class SlimCAE(tf.keras.Model):
    def __init__(self, switch_list, total_filters_num):
        super(SlimCAE, self).__init__()
        self.switch_list = switch_list
        self.total_filters_num = total_filters_num
        self.entropy_models = [
            EntropyModelClass(coding_rank=2) for _ in switch_list
        ]
        
        # Initialize analysis (encoder) layers
        self.analysis_conv_layers = []
        self.analysis_gdn_layers = []
        
        for i in range(len(switch_list)):
            # Convolutional layers - 3 per switch
            layer_set = []
            layer_set.append(SignalConv2D_slim(
                total_filters_num, (9, 9), corr=True, strides_down=4, padding="same_zeros",
                use_bias=True, activation=None, name=f"analysis_conv0_{i}"))
            layer_set.append(SignalConv2D_slim(
                total_filters_num, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=True, activation=None, name=f"analysis_conv1_{i}"))
            layer_set.append(SignalConv2D_slim(
                total_filters_num, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=False, activation=None, name=f"analysis_conv2_{i}"))
            self.analysis_conv_layers.append(layer_set)
            
            # GDN layers - 3 per switch
            gdn_set = []
            gdn_set.append(GDN(name=f"analysis_gdn0_{i}"))
            gdn_set.append(GDN(name=f"analysis_gdn1_{i}"))
            gdn_set.append(GDN(name=f"analysis_gdn2_{i}"))
            self.analysis_gdn_layers.append(gdn_set)
        
        # Initialize synthesis (decoder) layers
        self.synthesis_conv_layers = []
        self.synthesis_gdn_layers = []
        
        for i in range(len(switch_list)):
            # Convolutional layers - 3 per switch
            layer_set = []
            layer_set.append(SignalConv2D_slim(
                total_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=None, name=f"synthesis_conv0_{i}"))
            layer_set.append(SignalConv2D_slim(
                total_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=None, name=f"synthesis_conv1_{i}"))
            layer_set.append(SignalConv2D_slim(
                3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
                use_bias=True, activation=None, name=f"synthesis_conv2_{i}"))
            self.synthesis_conv_layers.append(layer_set)
            
            # Inverse GDN layers - 3 per switch
            gdn_set = []
            gdn_set.append(GDN(inverse=True, name=f"synthesis_gdn0_{i}"))
            gdn_set.append(GDN(inverse=True, name=f"synthesis_gdn1_{i}"))
            gdn_set.append(GDN(inverse=True, name=f"synthesis_gdn2_{i}"))
            self.synthesis_gdn_layers.append(gdn_set)
        
    def analysis_transform(self, tensor_in):
        tensor_encoder = []
        for i, _switch in enumerate(self.switch_list):
            # --- conv0 + GDN0 ----------
            # Conv0 outputs shape [B, H, W, _switch]
            tensor = self.analysis_conv_layers[i][0](tensor_in, 3, _switch)
            # pad _before_ GDN to full channels
            tensor = tf.pad(
                tensor,
                [[0, 0], [0, 0], [0, 0],
                [0, self.total_filters_num - _switch]],
                "CONSTANT")
            # now GDN always sees 'total_filters_num' channels
            tensor_gdn = self.analysis_gdn_layers[i][0](
                tensor, i, _switch)

            # --- conv1 + GDN1 ----------
            tensor = self.analysis_conv_layers[i][1](
                tensor_gdn, _switch, _switch)
            tensor = tf.pad(
                tensor,
                [[0, 0], [0, 0], [0, 0],
                [0, self.total_filters_num - _switch]],
                "CONSTANT")
            tensor_gdn = self.analysis_gdn_layers[i][1](
                tensor, i, _switch)

            # --- conv2 + GDN2 ----------
            tensor = self.analysis_conv_layers[i][2](
                tensor_gdn, _switch, _switch)
            tensor = tf.pad(
                tensor,
                [[0, 0], [0, 0], [0, 0],
                [0, self.total_filters_num - _switch]],
                "CONSTANT")
            tensor_gdn = self.analysis_gdn_layers[i][2](
                tensor, i, _switch)

            tensor_encoder.append(tensor_gdn)
        return tensor_encoder
    
    def synthesis_transform(self, tensor_encoder):
        """Builds the synthesis transform (decoder)."""
        tensor_decoder = []
        
        for i, _switch in enumerate(self.switch_list):
            # --- first pad + IGDN0 + deconv0 ----------
            # pad up to full channels before IGDN
            tensor = tf.pad(
                tensor_encoder[i],
                [[0, 0], [0, 0], [0, 0],
                 [0, self.total_filters_num - _switch]],
                "CONSTANT")
            tensor_igdn = self.synthesis_gdn_layers[i][0](
                tensor, i, _switch)
            tensor = self.synthesis_conv_layers[i][0](
                tensor_igdn, _switch, _switch)
            
            # --- second pad + IGDN1 + deconv1 ----------
            tensor = tf.pad(
                tensor,
                [[0, 0], [0, 0], [0, 0],
                 [0, self.total_filters_num - _switch]],
                "CONSTANT")
            tensor_igdn = self.synthesis_gdn_layers[i][1](
                tensor, i, _switch)
            tensor = self.synthesis_conv_layers[i][1](
                tensor_igdn, _switch, _switch)
            
            # --- third pad + IGDN2 + deconv2 ----------
            tensor = tf.pad(
                tensor,
                [[0, 0], [0, 0], [0, 0],
                 [0, self.total_filters_num - _switch]],
                "CONSTANT")
            tensor_igdn = self.synthesis_gdn_layers[i][2](
                tensor, i, _switch)
            tensor = self.synthesis_conv_layers[i][2](
                tensor_igdn, _switch, 3)
            
            tensor_decoder.append(tensor)
        
        return tensor_decoder
        
    def call(self, inputs, training=False):
        # Encoder
        y = self.analysis_transform(inputs)

        # Apply entropy models (quantize + likelihood)
        y_tilde = []
        likelihoods = []
        for i, _switch in enumerate(self.switch_list):
            _y_tilde, _likelihoods = self.entropy_models[i](
                y[i]
            )
            y_tilde.append(_y_tilde)
            likelihoods.append(_likelihoods)

        # Decoder
        x_tilde = self.synthesis_transform(y_tilde)
        return x_tilde, y, y_tilde, likelihoods



def train(last_step, lmbdas):
    """Trains the model with an optimized data pipeline and graph-compiled training step."""
    # Set logging level
    logger = tf.get_logger()
    logger.setLevel('INFO')

    # Build input pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    def preprocess(x):
        shape = tf.shape(x)[:2]
        ps = args.patchsize
        return tf.cond(
            tf.reduce_all(shape >= [ps, ps]),
            lambda: tf.image.random_crop(x, [ps, ps, 3]),
            lambda: tf.image.resize(x, [ps, ps])
        )

    train_dataset = (
        tf.data.Dataset
          .list_files(args.train_glob, shuffle=True)
          .shuffle(buffer_size=len(glob.glob(args.train_glob)))
          .repeat()
          .interleave(
              lambda f: tf.data.Dataset.from_tensors(f)
                        .map(load_image, num_parallel_calls=AUTOTUNE),
              cycle_length=args.preprocess_threads,
              num_parallel_calls=AUTOTUNE
          )
          .map(preprocess, num_parallel_calls=AUTOTUNE)
          .batch(args.batchsize, drop_remainder=True)
          .prefetch(AUTOTUNE)
    )

    num_pixels = args.batchsize * args.patchsize ** 2
    model = SlimCAE(args.switch_list, args.num_filters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            x_tildes, ys, y_tildes, likelihoods = model(x, training=True)
            losses, bpps, mses = [], [], []
            for i in range(len(args.switch_list)):
                bpp = tf.reduce_sum(tf.math.log(likelihoods[i])) / (-np.log(2) * num_pixels)
                mse = tf.reduce_mean(tf.square(x - x_tildes[i]))
                loss = lmbdas[i] * mse + bpp
                losses.append(loss)
                bpps.append(bpp)
                mses.append(mse)
            total_loss = tf.add_n(losses)
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        optimizer.apply_gradients(grads_and_vars)
        return total_loss, losses, bpps, mses, ys

    # TensorBoard and checkpoint setup
    log_dir = os.path.join(args.checkpoint_dir, "logs")
    summary_writer = tf.summary.create_file_writer(log_dir)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint_dir, max_to_keep=5)

    # Restore from latest checkpoint
    latest = manager.latest_checkpoint
    if latest:
        ckpt.restore(latest)
        print(f"Restored from {latest}")
        step = int(latest.split('-')[-1])
    else:
        step = 0

    # Training loop
    while step < last_step:
        for batch in train_dataset:
            loss, losses, bpps, mses, ys = train_step(batch)
            if step % 1 == 0:
                print(f"Step: {step}, Loss: {loss.numpy():.6f}")
                with summary_writer.as_default():
                    tf.summary.scalar("total_loss", loss, step=step)
                    for i in range(len(losses)):
                        tf.summary.scalar(f"loss_{i}", losses[i], step=step)
                        tf.summary.scalar(f"bpp_{i}", bpps[i], step=step)
                        tf.summary.scalar(f"mse_{i}", mses[i] * 255**2, step=step)
                        tf.summary.histogram(f"hist_y_{i}", ys[i], step=step)
            if step % 1000 == 0:
                save_path = manager.save(checkpoint_number=step)
                print(f"Checkpoint saved: {save_path}")
            step += 1
            if step >= last_step:
                break

    # Final checkpoint
    save_path = manager.save(checkpoint_number=step)
    print(f"Final checkpoint saved: {save_path}")
    return model

def evaluate(last_step):
    """Evaluate the model for test dataset"""
    
    # Create model
    model = SlimCAE(args.switch_list, args.num_filters)
    
    # Restore the model from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
    status = checkpoint.restore(latest_checkpoint)
    status.expect_partial()  # Suppress warnings about optimizer variables
    
    # Process all images
    imagesList = os.listdir(args.inputPath)
    
    # Initialize metric scores
    bpp_estimate_total = [0.0] * len(args.switch_list)
    mse_total = [0.0] * len(args.switch_list)
    psnr_total = [0.0] * len(args.switch_list)
    msssim_total = [0.0] * len(args.switch_list)
    msssim_db_total = [0.0] * len(args.switch_list)
    
    for image in imagesList:
        img_path = os.path.join(args.inputPath, image)
        x = load_image(img_path)
        x = tf.expand_dims(x, 0)
        x.set_shape([1, None, None, 3])
        
        # Forward pass through the model
        x_tilde, _, _, likelihoods = model(x, training=False)
        
        # Get number of pixels
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), tf.float32)
        
        # Scale to 0-255 range for metrics
        x_255 = x * 255
        
        for i, _switch in enumerate(args.switch_list):
            # Calculate bpp
            bpp = tf.reduce_sum(tf.math.log(likelihoods[i])) / (-np.log(2) * num_pixels)
            
            # Prepare reconstructed image
            x_rec = tf.clip_by_value(x_tilde[i], 0, 1)
            x_rec_255 = tf.round(x_rec * 255)
            
            # Calculate metrics
            mse = tf.reduce_mean(tf.square(x_255 - x_rec_255))
            psnr = tf.image.psnr(x_rec_255, x_255, 255)
            msssim = tf.image.ssim_multiscale(x_rec_255, x_255, 255)
            msssim_db = -10 * tf.math.log(1 - msssim) / tf.math.log(10.0)
            
            # Accumulate metrics
            bpp_estimate_total[i] += bpp
            mse_total[i] += mse
            psnr_total[i] += psnr[0]
            msssim_total[i] += msssim[0]
            msssim_db_total[i] += msssim_db[0]
            
            print(f"Image {image}, Switch {i}")
            print(f"  BPP: {bpp:.4f}")
            print(f"  PSNR: {psnr[0]:.2f}")
            print(f"  MS-SSIM: {msssim[0]:.4f}")
    
    # Calculate averages
    num_images = len(imagesList)
    avg_bpp = [b / num_images for b in bpp_estimate_total]
    avg_mse = [m / num_images for m in mse_total] 
    avg_psnr = [p / num_images for p in psnr_total]
    avg_msssim = [m / num_images for m in msssim_total]
    avg_msssim_db = [m / num_images for m in msssim_db_total]
    
    # Save results if needed
    if args.evaluation_name is not None:
        result_file = f"{args.evaluation_name}_{last_step}.txt"
        with open(result_file, 'w') as f:
            f.write(f'Avg_bpp_estimate: {avg_bpp}\n')
            f.write(f'Avg_mse: {avg_mse}\n')
            f.write(f'Avg_psnr: {avg_psnr}\n')
            f.write(f'Avg_msssim: {avg_msssim}\n')
            f.write(f'Avg_msssim_db: {avg_msssim_db}\n')
    
    return avg_bpp, avg_psnr


def train_loop():
    """Search the optimal RD points in a slimmable compressive autoencoder"""
    
    # Initial RD tradeoffs
    lmbdas = args.lmbda.copy()
    
    # Train SlimCAE as stage 1
    model = train(args.last_step, lmbdas)
    
    # Evaluate the model with validation dataset
    bpp, psnr = evaluate(args.last_step)
    
    # Logs for lambda scheduling
    lambda_log = [list(lmbdas)]
    grad_flag_log = []
    grad_current_log = []
    
    # Train SlimCAE with lambda scheduling as stage 2
    for i in range(len(lmbdas) - 1):
        if bpp[i] == bpp[i+1]:  # Avoid division by zero
            continue
            
        grad_flag = (psnr[i] - psnr[i+1]) / (bpp[i] - bpp[i+1])
        factor = 1
        m = 1
        
        while True:
            # Create new lambda values
            new_lmbdas = lmbdas.copy()
            for j in range(i+1, len(new_lmbdas)):
                new_lmbdas[j] *= 0.8
                
            lambda_log.append(list(new_lmbdas))
            lmbdas = new_lmbdas  # Update lambdas
            
            # Train more steps
            next_step = args.last_step + 20
            model = train(next_step, lmbdas)
            args.last_step = next_step
            
            # Evaluate again
            bpp, psnr = evaluate(args.last_step)
            
            if bpp[i] == bpp[i+1]:  # Avoid division by zero
                break
                
            grad_current = (psnr[i] - psnr[i+1]) / (bpp[i] - bpp[i+1])
            grad_flag_log.append(grad_flag)
            grad_current_log.append(grad_current)
            
            if grad_current > grad_flag:
                break
            else:
                if m == 1:
                    factor_flag = grad_flag - grad_current
                elif m < 7:
                    factor = (grad_flag - grad_current) / factor_flag
                    factor_flag = grad_flag - grad_current
                else:
                    break
                grad_flag = grad_current
                m += 1
    
    # Save logs
    save_log(os.path.join(args.checkpoint_dir, 'lmbdaslog.csv'), lambda_log)
    save_log(os.path.join(args.checkpoint_dir, 'grad_flag_log.csv'), grad_flag_log)
    save_log(os.path.join(args.checkpoint_dir, 'grad_current_log.csv'), grad_current_log)


def save_log(filename, data):
    """Save log data to a CSV file"""
    with open(filename, 'w', newline='') as myfile:
        writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "command", choices=["train", "evaluate", "train_lambda_schedule"],
        help="What to do: 'train' loads training data and trains "
             "a new model. 'evaluate' loads a pre-trained model and "
             "evaluates on a given dataset. 'train_lambda_schedule' "
             "means train a new model with lambda scheduling. ")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
        "--num_filters", type=int, default=128,
        help="Number of filters per layer.")
    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--train_glob", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG format.")
    parser.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    parser.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    parser.add_argument(
        "--lambda", nargs="+", type=float, default=[512], dest="lmbda",
        help="Lambdas for rate-distortion tradeoff point.")
    parser.add_argument(
        "--last_step", type=int, default=800000,
        help="Train up to this number of steps.")
    parser.add_argument(
        "--preprocess_threads", type=int, default=6,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    parser.add_argument(
        "--switch_list", nargs="+", type=int, default=[64],
        help="Number of filters per layer.")
    parser.add_argument(
        "--evaluation_name", type=str, default='./searchRDpoints/One',
        help="the name of evaluation results txt file.")
    parser.add_argument(
        "--inputPath", type=str, default=None,
        help="Directory where to evaluation dataset.")
    parser.add_argument(
        "--train_jointly", action="store_true",
        help="Train all the variables together.")

    args = parser.parse_args()
    
    if args.command == "train":
        train(args.last_step, args.lmbda)
    elif args.command == "train_lambda_schedule":
        train_loop()
    elif args.command == "evaluate":
        if args.inputPath is None:
            raise ValueError("Need input path for evaluation.")
        evaluate(args.last_step)