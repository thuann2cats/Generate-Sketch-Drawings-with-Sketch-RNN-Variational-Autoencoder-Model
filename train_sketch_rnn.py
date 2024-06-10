import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Any
import os, math
import random
import argparse

from utils import StrokesDataset
from models import SketchRNN, SketchRNNDecoder, SketchRNNEncoder

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def save_hyperparams(variable_dict: dict, folder_path: str, filename="config.json"):
    """Saving the hyparameters of the encoder and the decoder"""
    
    # Open the file for writing in JSON format
    with open(os.path.join(folder_path, filename), 'w') as file:
        import json
        json.dump(variable_dict, file, indent=4)  # Indent for readability
        
          
def generate_one_sketch(model, valid_loader, savefig_file_path, device):
    # get a sample from valid_loader
    # need: valid_loader
    # output: 1 sample of input_stroke, shape (seq_len, 5) & mask, shape (seq_len - 1, )
    input_stroke_sample: torch.Tensor
    mask_sample: torch.Tensor
    random_idx = np.random.choice(len(valid_loader.dataset))
    input_stroke_sample, mask_sample = valid_loader.dataset[random_idx]
    seq_len, _ = input_stroke_sample.shape
    assert input_stroke_sample.shape == (seq_len, 5)
    assert mask_sample.shape == (seq_len - 1, )
    
    input_stroke_sample = input_stroke_sample.to(device)
    mask_sample = mask_sample.to(device)
    
    # pass the sample through the model
    generated_stroke_sequence: torch.Tensor
    generated_stroke_sequence = model.sample(input_stroke_sample, mask_sample)
    
    # output: stroke sequence (generated_length, 5)
    generated_length, _ = generated_stroke_sequence.shape
    assert generated_stroke_sequence.shape == (generated_length, 5)
    
    # plot the stroke sequence

    # Get the directory path
    directory, filename = os.path.split(savefig_file_path)

    # Extract filename without extension
    base_filename, extension = os.path.splitext(filename)

    # Create the modified filename
    modified_filename = base_filename + "-input"

    # Combine the modified filename with directory and extension
    modified_path = os.path.join(directory, modified_filename + extension)
    # input tensor (generated_length, 5), png file name
    SketchRNN.plot(input_stroke_sample, modified_path)
    SketchRNN.plot(generated_stroke_sequence, savefig_file_path)
    # output: savefig

def train_sketchRNN(
    model: "SketchRNN",
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    folder_path: str,
    device,
    optimizer_func=torch.optim.Adam,
    lr=0.005,
    save_every_n_epochs=1,
    new_line_every_n_batches=100,
    sample_every_n_batches=100,
    tensorboard_folder="tensorboard_logging",
    log_file_name="training_log.txt",
    sampling_folder="sampled_sketches",
    saved_encoder_subfolder="saved_encoders",
    saved_decoder_subfolder="saved_decoders",
    saved_encoder_starting_idx=1,
    saved_decoder_starting_idx=1
):
    import os, sys
    # import torch.utils.tensorboard
    from torch.utils.tensorboard import SummaryWriter

    num_batches = len(train_loader)

    model.to(device)
    f = open(os.path.join(folder_path, log_file_name), 'w')
    tensorboard_writer = SummaryWriter(os.path.join(folder_path, tensorboard_folder))
    optimizer = optimizer_func(model.parameters(), lr=lr)
    decoder_folder_path = os.path.join(folder_path, saved_decoder_subfolder)
    encoder_folder_path = os.path.join(folder_path, saved_encoder_subfolder)
    next_encoder_idx = saved_encoder_starting_idx
    next_decoder_idx = saved_decoder_starting_idx

    for epoch in range(1, num_epochs+1):
        for batch_idx, (input_strokes, masks) in enumerate(train_loader):
            
            if batch_idx % sample_every_n_batches == 0:
                sampling_folder_path = os.path.join(folder_path, sampling_folder)
                savefig_filename_fullpath = os.path.join(sampling_folder_path, f"before_epoch{epoch}_batch{batch_idx}.png")
                generate_one_sketch(model=model, valid_loader=valid_loader, savefig_file_path=savefig_filename_fullpath, device=device)
            
            input_strokes = input_strokes.to(device)
            masks = masks.to(device)
            model.train()
            
            batch_size, longest_len, _ = input_strokes.shape
            input_strokes = input_strokes.transpose(0, 1)
            masks = masks.transpose(0, 1)
            assert input_strokes.shape == (longest_len, batch_size, 5)
            assert masks.shape == (longest_len - 1, batch_size)
            
            model.zero_grad()
            
            loss_strokes, loss_p, loss_kl = model(input_strokes, masks)
            
            loss = torch.mean(loss_kl * model.weight_KL + (loss_p + loss_strokes))
            
            mean_loss_construction = torch.mean(loss_p + loss_strokes)
            mean_loss_kl = torch.mean(loss_kl)
        
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()
            
            # Get training stats
            stats = f"Epoch [{epoch:3}/{num_epochs:3}], Step [{batch_idx:4}/{num_batches:4}], Loss: {loss.item():.4f}, Loss/R: {mean_loss_construction.item():.4f}, Loss/KL: {mean_loss_kl.item():.4f}"
            num_batches_so_far = (epoch - 1) * num_batches + batch_idx
            tensorboard_writer.add_scalar(tag="Train/Loss/Total", scalar_value=loss.item(), global_step=num_batches_so_far)
            tensorboard_writer.add_scalar(tag="Train/Loss/Reconstruction", scalar_value=mean_loss_construction.item(), global_step=num_batches_so_far)
            tensorboard_writer.add_scalar(tag="Train/Loss/KL", scalar_value=mean_loss_kl.item(), global_step=num_batches_so_far)
            
            # Track gradients
            for name, param in model.named_parameters():
                if param.requires_grad:
                    tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.mean(), num_batches_so_far)
            
            # Print training stats (on same line)
            print('\r' + stats, end="")
            sys.stdout.flush()
            
            
            # Print training stats (on different line)
            if batch_idx % new_line_every_n_batches == 0:
                print('\r' + stats)
                # Print training staistics to file.
                f.write(stats + '\n')
                f.flush()
                
                
        # Save the weights
        if epoch % save_every_n_epochs == 0:
            torch.save(model.decoder.state_dict(), os.path.join(decoder_folder_path, f"decoder-{next_decoder_idx}.pt"))
            next_decoder_idx += 1
            torch.save(model.encoder.state_dict(), os.path.join(encoder_folder_path, f"encoder-{next_encoder_idx}.pt"))
            next_encoder_idx += 1
            
    tensorboard_writer.close()        
    f.close()
    


def main():
    
    def validate_learning_rate(value):
        """Custom validation function to check for float <= 2.0."""
        value = float(value)
        if value > 2.0:
            raise argparse.ArgumentError(f"{value} is not a valid learning rate (must be <= 2.0).")
        return value
    
    parser = argparse.ArgumentParser(description="Train a Sketch-RNN model.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the training data (.npz file).")
    parser.add_argument("--folder_path", type=str, required=True,
                        help="Path to the folder for saving model checkpoints and logs.")
    
    
    parser.add_argument("--num_epochs", type=int, default=70,
                    help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--lr", type=validate_learning_rate, default=0.001,
                        help="Learning rate for the optimizer (must be <= 2.0).")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for training.")
    parser.add_argument("--encoder_lstm_hidden_size", type=int, default=256, choices=range(64, 2049), metavar="64-2048",
                    help="Hidden size of the encoder LSTM (must be between 64 and 2048).")
    parser.add_argument("--decoder_lstm_hidden_size", type=int, default=512, choices=range(64, 2049), metavar="64-2048",
                        help="Hidden size of the decoder LSTM (must be between 64 and 2048).")
    parser.add_argument("--d_z", type=int, default=128, choices=range(64, 1025), metavar="64-1024",
                        help="Dimensionality of the latent space (must be between 64 and 1024).")
    parser.add_argument("--num_Gaussians", type=int, default=20, choices=range(1, 128), metavar="1-127",
                        help="Number of Gaussian mixtures in the decoder (must be between 1 and 127).")
    
    def validate_weight_KL(value):
        """Custom validation function to check for float <= 2.0."""
        value = float(value)
        if value > 3.0:
            raise argparse.ArgumentError(f"{value} is not a valid weight_KL (must be <= 3.0).")
        return value

    parser.add_argument("--weight_KL", type=validate_weight_KL, default=0.5,
                    help="Weight of the KL divergence loss term (must be <= 3.0).")

    parser.add_argument("--log_file_name", type=str, default="training_log.txt",
                        help="Name of the log file (automatically saved inside 'folder_path').")
    parser.add_argument("--encoder_folder_name", type=str, default="saved_encoders",
                        help="Folder name for saving encoder checkpoints (automatically saved inside 'folder_path').")
    parser.add_argument("--decoder_folder_name", type=str, default="saved_decoders",
                        help="Folder name for saving decoder checkpoints (automatically saved inside 'folder_path').")
    parser.add_argument("--tensorboard_folder_name", type=str, default="tensorboard_logging",
                        help="Folder name for TensorBoard logging (automatically saved inside 'folder_path').")
    parser.add_argument("--sampled_sketches_folder_name", type=str, default="sampled_sketches",
                        help="Folder name to save sampled sketches (at the beginning of each batch - this folder is automatically saved inside 'folder_path').")
    parser.add_argument("--optimizer_func", type=str, default="adam", choices=["adam", "rmsprop"],  # Restrict choices
                        help="Optimizer function (adam or rmsprop).")
    

    
    parser.add_argument("--load_encoder_checkpoint_file", type=str,
                    help="Path to a pre-trained encoder checkpoint file (optional).")
    parser.add_argument("--load_decoder_checkpoint_file", type=str,
                        help="Path to a pre-trained decoder checkpoint file (optional).")


    max_seq_length = 200
    
    args = parser.parse_args()
    
    # Accessing arguments by name
    num_epochs=args.num_epochs
    batch_size = args.batch_size
    path_to_npz_file = args.data_file
    encoder_lstm_hidden_size = args.encoder_lstm_hidden_size
    decoder_lstm_hidden_size = args.decoder_lstm_hidden_size
    d_z = args.d_z
    num_Gaussians = args.num_Gaussians
    weight_KL = args.weight_KL
    folder_path = args.folder_path
    log_file_name = args.log_file_name
    encoder_folder_name = args.encoder_folder_name
    decoder_folder_name = args.decoder_folder_name
    tensorboard_folder_name = args.tensorboard_folder_name
    sampled_sketches_folder_name = args.sampled_sketches_folder_name
    if args.optimizer_func.lower() == "adam":
        optimizer_func = torch.optim.Adam
    elif args.optimizer_func.lower() == "rmsprop":
        optimizer_func = torch.optim.RMSprop
    lr = args.lr
    load_encoder_checkpoint_file = args.load_encoder_checkpoint_file
    load_decoder_checkpoint_file = args.load_decoder_checkpoint_file
    
    ### Saving the hyparameters of the encoder and the decoder
    
    # Create the folder if it doesn't exist (optional)
    os.makedirs(folder_path, exist_ok=True)  # Creates folder if it doesn't exist

    # save these variables into a file
    variable_dict = {
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "path_to_npz_file": path_to_npz_file,
        "encoder_lstm_hidden_size": encoder_lstm_hidden_size,
        "decoder_lstm_hidden_size": decoder_lstm_hidden_size,
        "d_z": d_z,
        "num_Gaussians": num_Gaussians,
        "weight_KL": weight_KL,
        "folder_path": folder_path,
        "log_file_name": log_file_name,
        "encoder_folder_name": encoder_folder_name,
        "decoder_folder_name": decoder_folder_name,
        "tensorboard_folder_name": tensorboard_folder_name,
        "sampled_sketches_folder_name": sampled_sketches_folder_name,
        "lr": lr,
        "num_epochs": num_epochs,
    }
    if load_encoder_checkpoint_file is not None:
        variable_dict["load_encoder_checkpoint_file"] = load_encoder_checkpoint_file
    if load_decoder_checkpoint_file is not None:
        variable_dict["load_decoder_checkpoint_file"] = load_decoder_checkpoint_file
        
    save_hyperparams(variable_dict, folder_path)
    
    ### Preparing train & test loaders

    dataset = np.load(str(path_to_npz_file), encoding='latin1', allow_pickle=True)

    # Create training dataset
    train_dataset = StrokesDataset(dataset['train'], max_seq_length)
    # Create validation dataset
    valid_dataset = StrokesDataset(dataset['valid'], max_seq_length, train_dataset.scale)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

    ### Preparing the modules
    
    encoder = SketchRNNEncoder(lstm_hidden_size=encoder_lstm_hidden_size, d_z=d_z)
    decoder = SketchRNNDecoder(lstm_hidden_size=decoder_lstm_hidden_size, d_z=d_z, num_Gaussians=num_Gaussians)
    
    ### load encoder and decoder checkpoints (if specified)
    for file, module in ((load_encoder_checkpoint_file, encoder), (load_decoder_checkpoint_file, decoder)):
        if file:
            module_state_dict = torch.load(file)
            module.load_state_dict(module_state_dict)
            
    sketch_rnn_model = SketchRNN(encoder=encoder, decoder=decoder, weight_KL=weight_KL)
    
    ### Create the folders to save encoder and decoder later

    # Combine path components for subfolder
    encoder_folder_path = os.path.join(folder_path, encoder_folder_name)
    decoder_folder_path = os.path.join(folder_path, decoder_folder_name)

    # Create the subfolder if it doesn't exist
    try:
        os.makedirs(encoder_folder_path, exist_ok=True)
        os.makedirs(decoder_folder_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating folder: {e}")
        
    ### Initialize other components of training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")


    ### training

    normal_repr = torch.Tensor.__repr__ 
    torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"  

    train_sketchRNN(
        model=sketch_rnn_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=num_epochs,
        folder_path=folder_path,
        device=device,
        optimizer_func=optimizer_func,
        lr=lr,
        save_every_n_epochs=1,
        new_line_every_n_batches=100,
        sample_every_n_batches=100,
        tensorboard_folder=tensorboard_folder_name,
        log_file_name=log_file_name,
        saved_encoder_subfolder=encoder_folder_name,
        saved_decoder_subfolder=decoder_folder_name,
        sampling_folder=sampled_sketches_folder_name,
    )


if __name__ == "__main__":
    main()
    