- Setup a conda environment with python 3.6
- `pip install tensorflow==1.13.1` 
- `pip install tensorflow-gpu==1.13.1` // If u are on GPU
- `pip install numpy==1.16.4`
- Run the following Command to Train:

python SlimCAE.py -v --train_glob="F:\Spring25\IoT\Slimmable_VAE\Kodak_Raw\*.png" --patchsize 240 --num_filter 192 --switch_list 192 144 96 72 48 --train_jointly --lambda 2048 1024 512 256 128 --last_step 1000000 --checkpoint_dir "F:\Spring25\IoT\Slimmable_VAE\models" train

- Run the following Command to Test:
python SlimCAE.py --num_filters 192 --switch_list 192 144 96 72 48 --checkpoint_dir "F:\Spring25\IoT\Slimmable_VAE\models" --inputPath "F:/Spring25/IoT/Slimmable_VAE/Kodak_Raw/" --evaluation_name "F:/Spring25/IoT/Slimmable_VAE/Eval_1" evaluate