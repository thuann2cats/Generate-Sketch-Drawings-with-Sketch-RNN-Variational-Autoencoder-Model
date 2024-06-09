import numpy as np
import torch
import os, math


class SketchRNNEncoder(torch.nn.Module):
    """Encoder module for the Sketch RNN model, consisting of a bidirection LSTM to learn a representation of the ground-truth stroke sequence for a particular class (object, such as bicycle). The representation is then projected via a Linear layer into mu and sigma, which represents a distribution over the latent space, from which a "z" latent vector would be sampled to feed into the decoder.
    """ 
    
    def __init__(self, lstm_hidden_size: int = 512, d_z: int = 128):
        """
        Encoder module for the Sketch RNN model, consisting of a bidirection LSTM to learn a representation of the ground-truth stroke sequence for a particular class (object, such as bicycle). The representation is then projected via a Linear layer into mu and sigma, which represents a distribution over the latent space, from which a "z" latent vector would be sampled to feed into the decoder.
        
        Args:
            lstm_hidden_size (int, optional): Hidden size of the LSTM. Defaults to 512.
            d_z (int, optional): Dimension of the latent representation. Defaults to 128.
        """        
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.d_z = d_z
        self.encoder_lstm = torch.nn.LSTM(input_size=5, hidden_size=lstm_hidden_size, num_layers=1, batch_first=False, bidirectional=True)
        # self.mu_fc = torch.nn.Linear(in_features=lstm_hidden_size * 4, out_features=d_z)
        self.mu_fc = torch.nn.Linear(in_features=lstm_hidden_size * 2, out_features=d_z)
        # self.sigma_fc = torch.nn.Linear(in_features=lstm_hidden_size * 4, out_features=d_z)
        self.sigma_fc = torch.nn.Linear(in_features=lstm_hidden_size * 2, out_features=d_z)
        
    def forward(self, input_strokes: torch.Tensor, masks: torch.Tensor):
        """Pass the ground-truth sketches through the encoder to learn latent representation for them.

        Args:
            input_strokes (Tensor): shape (longest_len, batch_size, 5) (longest_len already includes the SOS token)
            masks (_type_): the mask for actual strokes - shape(longest_len - 1, batch_size)
            
        Returns:
            mu_z (batch_size, d_z) mu for the distribution of the z-embedding
            sigma_z (batch_size, d_z) sigma for the distribution of the z-embedding
            z_embedding: (batch_size, d_z) embeddings for all stroke sequences in a batch 
        """
        
        ## check input shapes
        
        longest_len, batch_size, dim3 = input_strokes.shape
        assert dim3 == 5
        assert masks.shape == (longest_len - 1, batch_size)
        current_device = next(self.parameters()).device
        
        ## pass through LSTM layer
        
        temp, (concatenated_forward_backward_hidden_state, concatenated_forward_backward_cell_state) = self.encoder_lstm(input_strokes)
        assert concatenated_forward_backward_hidden_state.shape == (2, batch_size, self.lstm_hidden_size)
        assert concatenated_forward_backward_cell_state.shape == (2, batch_size, self.lstm_hidden_size)
        
        hidden_state_shaped_into_vector = concatenated_forward_backward_hidden_state.transpose(0, 1).reshape(batch_size, -1)
        cell_state_shaped_into_vector = concatenated_forward_backward_cell_state.transpose(0, 1).reshape(batch_size, -1)
        
        # combined_hidden_cell = torch.cat((hidden_state_shaped_into_vector, cell_state_shaped_into_vector), dim=1)
        # assert combined_hidden_cell.shape == (batch_size, self.lstm_hidden_size * 4)
        
        combined_hidden_cell = torch.cat((hidden_state_shaped_into_vector, ), dim=1)
        assert combined_hidden_cell.shape == (batch_size, self.lstm_hidden_size * 2)
        
        ## get the mu_z and sigma_z, the parameters for the distribution for the latent space
        
        # mu_z: (batch_size, d_z) mu for the distribution of the z-embedding
        mu_z = self.mu_fc(combined_hidden_cell)
        assert mu_z.shape == (batch_size, self.d_z)

        # sigma_z = exp(sigma_hat/2)
        # sigma_z (batch_size, d_z) sigma for the distribution of the z-embedding
        sigma_hat = self.sigma_fc(combined_hidden_cell)
        assert sigma_hat.shape == (batch_size, self.d_z)
        sigma_z = torch.exp(sigma_hat / 2)
        assert sigma_z.shape == (batch_size, self.d_z)

        ## sample from N(0, I), then get z
        # z_embedding: (batch_size, d_z) * embeddings for all stroke sequences in a batch 
        z_embedding = mu_z + sigma_z * torch.normal(mean=mu_z.new_zeros(mu_z.shape).to(current_device), std=sigma_z.new_ones(sigma_z.shape).to(current_device))
        assert z_embedding.shape == (batch_size, self.d_z)
        
        return z_embedding, mu_z, sigma_z
        
                
class SketchRNNDecoder(torch.nn.Module):
    """Decoder module for Sketch RNN. Consists of an LSTM layer and linear projections. Passed into the LSTM layer are the concatenated input strokes and z_embedding (latent representation generated by the encoder) of the input strokes. The h0, c0 for the LSTM are generated from the z_embedding latent representation as well. The output of the LSTM is then projected via linear layers to get the parameters for the Gaussian mixture and for the q-logits that make up the distribution for the p1-p2-p3 values."""
    
    def __init__(self, lstm_hidden_size: int = 1024, d_z: int = 128, num_Gaussians: int = 20):
        """_summary_

        Args:
            lstm_hidden_size (int, optional): Hidden dimension of the LSTM layer. Defaults to 1024.
            d_z (int, optional): Dimension of the latent representation passed into the decoder. Defaults to 128.
            num_Gaussians (int, optional): Number of Gaussians in the mixture. Defaults to 20.
        """
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.d_z = d_z
        self.num_Gaussians = num_Gaussians
        self.init_h0 = torch.nn.Linear(in_features=d_z, out_features=self.lstm_hidden_size)
        self.init_c0 = torch.nn.Linear(in_features=d_z, out_features=self.lstm_hidden_size)
        self.decoder_lstm = torch.nn.LSTM(input_size=d_z + 5, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=False, bidirectional=False)
        self.fc_mixture = torch.nn.Linear(in_features=self.lstm_hidden_size, out_features=6 * num_Gaussians)
        self.fc_qlogits = torch.nn.Linear(in_features=self.lstm_hidden_size, out_features=3)
        
    def forward(self, z_embedding: torch.Tensor, input_strokes: torch.Tensor, input_state: tuple[torch.Tensor, torch.Tensor] = None, temperature: float = 1.0):
        """Pass the z_embedding (generated by the encoder) through the decoder to get the Gaussian mixture and the distribution for p1-p2-p3 values. Grouth-truth data are also necessary.

        Args:
            z_embedding (torch.Tensor): the latent representation - shape (batch_size, d_z)
            input_strokes (torch.Tensor): shape (seq_len, batch_size, 5)
            input_state: input state for the decoder LSTM. A tuple of hidden state and cell state. Each with shape (1, batch_size, self.lstm_hidden_size). If None, then h0, c0 will be generated from the z_embedding.
            temperature: to adjust the "randomness" of the output distributions. During training, temperature should be 1.0. During sampling, temperature is set between 0 and 1. Closer to 0 means more deterministic.
            
        Returns:
            # MultivariateNormal distributions with loc (seq_len, batch_size, num_Gaussians, 2) std dev (seq_len, batch_size, num_Gaussians, 2, 2): a "tensor" of 2-D Gaussian distributions (to sample delta_x, delta_y on generation)
            Gaussian_distributions: torch.distributions.MultivariateNormal
            
            # Categorical distributions with logits shape (seq_len, batch_size, num_Gaussians): a "tensor" of Categorical distributions for the "mixing coefficients" or "pi" values
            mixing_coefficients: torch.distributions.Categorical
            
            # Categorical distributions with logits shape (seq_len, batch_size, 3): a "tensor" of Categorical distributions for "q logits" value (to sample p1, p2, p3 on generation)
            p_distribution: torch.distributions.Categorical
            
            # state: last state generated by decoder LSTM. A tuple of hidden state and cell state. Each with shape (1, batch_size, self.lstm_hidden_size)
        """        
        ## Check the inputs
        seq_len, batch_size, dim3 = input_strokes.shape
        assert dim3 == 5
        assert z_embedding.shape == (batch_size, self.d_z)
        if self.training:
            assert temperature == 1.0
        
        ## Prepare the data to feed into the decoder
        current_device = next(self.parameters()).device
        
        z_embedding_repeated = z_embedding.unsqueeze(-1).repeat(1, 1, seq_len).permute(2, 0, 1)
        assert z_embedding_repeated.shape == (seq_len, batch_size, self.d_z)
        # concat_input_and_z (seq_len, batch_size, d_z + 5): concat "z" into each stroke of the input data
        concat_input_and_z = torch.concat((input_strokes, z_embedding_repeated), dim=-1)
        assert concat_input_and_z.shape == (seq_len, batch_size, self.d_z + 5)
        
        
        if input_state is None:
            # h0 (batch_size, hidden_size), then reshaped to (1, batch_size, hidden_size)
            h0 = torch.tanh(self.init_h0(z_embedding))
            h0 = h0.unsqueeze(0)
            assert h0.shape == (1, batch_size, self.lstm_hidden_size)
            
            # c0 (batch_size, hidden_size), then reshaped to (1, batch_size, hidden_size)
            c0 = torch.tanh(self.init_c0(z_embedding))
            c0 = c0.unsqueeze(0)
            assert c0.shape == (1, batch_size, self.lstm_hidden_size)
            state = (h0, c0)
        else:
            state = input_state
            
        
        ## Pass through decoder LSTM
        
        # decoder_lstm_outputs (seq_len, batch_size, hidden_size)
        decoder_lstm_outputs, state = self.decoder_lstm(concat_input_and_z, state)
        assert decoder_lstm_outputs.shape == (seq_len, batch_size, self.lstm_hidden_size)
        assert state[0].shape == state[1].shape == (1, batch_size, self.lstm_hidden_size)
        
        ## Pass through projection layers to get the parameters for distributions
        
        # Gaussian_mixture_params (seq_len, batch_size, 6*num_Gaussians)
        Gaussian_mixture_params = self.fc_mixture(decoder_lstm_outputs)
        assert Gaussian_mixture_params.shape == (seq_len, batch_size, 6*self.num_Gaussians)
        
        # Gaussian_mixture_params_reshaped (seq_len, batch_size, num_Gaussians, 6): represents the 6 parameters for each Gaussian for each timestep, in each sequence in the batch
        Gaussian_mixture_params_reshaped = Gaussian_mixture_params.reshape(seq_len, batch_size, self.num_Gaussians, 6)
        assert Gaussian_mixture_params_reshaped.shape == (seq_len, batch_size, self.num_Gaussians, 6)
        
        # q_logits_params (seq_len, batch_size, 3)
        q_logits_params = self.fc_qlogits(decoder_lstm_outputs)
        assert q_logits_params.shape == (seq_len, batch_size, 3)
        
        ## Create the distributions to return
        
        # MultivariateNormal distributions with loc (seq_len, batch_size, num_Gaussians, 2) std dev (seq_len, batch_size, num_Gaussians, 2, 2): a "tensor" of 2-D Gaussian distributions (to sample delta_x, delta_y on generation)
        Gaussian_distributions: torch.distributions.MultivariateNormal
        # Categorical distributions with logits shape (seq_len, batch_size, num_Gaussians): a "tensor" of Categorical distributions for the "mixing coefficients" or "pi" values
        mixing_coefficients: torch.distributions.Categorical
        # Categorical distributions with logits shape (seq_len, batch_size, 3): a "tensor" of Categorical distributions for "q logits" value (to sample p1, p2, p3 on generation)
        p_distribution: torch.distributions.Categorical
        
        # In the last dimension of the Gaussian_mixture_params tensor are 6 values for each Gaussian
        # first value: pi_hat, used to compute mixing coefficient for that Gaussian
        # second and third values: mu_x and mu_y for that Gaussian
        # fourth and fifth values: sigma_x_hat and sigma_y_hat for that Gaussian, used to compute sigma_x and sigma_y
        # sixth value: rho_xy_hat, used to compute rho_xy, the rho-coefficient, used to construct the covariance matrix for that Gaussian
        pi_hat = Gaussian_mixture_params_reshaped[:, :, :, 0] / temperature
        # pi = torch.nn.functional.softmax(pi_hat, dim=-1)
        assert pi_hat.shape == (seq_len, batch_size, self.num_Gaussians)
        # mixing_coefficients = torch.distributions.Categorical(probs=pi)
        mixing_coefficients = torch.distributions.Categorical(logits=pi_hat)
        
        mu_for_all_Gaussians = Gaussian_mixture_params_reshaped[:, :, :, 1:3]
        
        sigma_x_hat = Gaussian_mixture_params_reshaped[:, :, :, 3] 
        sigma_x = torch.exp(sigma_x_hat) * math.sqrt(temperature)
        sigma_y_hat = Gaussian_mixture_params_reshaped[:, :, :, 4] 
        sigma_y = torch.exp(sigma_y_hat) * math.sqrt(temperature)
        rho_xy_hat = Gaussian_mixture_params_reshaped[:, :, :, 5]
        rho_xy = torch.nn.functional.tanh(rho_xy_hat)
        
        # clamp the values to prevent the covariance matrix from failing PyTorch's PositiveDefinite check:
        # following this technique from here: https://nn.labml.ai/sketch_rnn/index.html
        sigma_x = torch.clamp_min(sigma_x, 1e-5)
        sigma_y = torch.clamp_min(sigma_y, 1e-5)
        rho_xy = torch.clamp(rho_xy, -1 + 1e-5, 1 - 1e-5)
        
        covar_matrix_for_all_Gaussians = torch.zeros(size=(seq_len, batch_size, self.num_Gaussians, 2, 2))
        covar_matrix_for_all_Gaussians = covar_matrix_for_all_Gaussians.to(current_device)
        covar_matrix_for_all_Gaussians[:, :, :, 0, 0] = sigma_x**2
        covar_matrix_for_all_Gaussians[:, :, :, 1, 1] = sigma_y**2
        
        covar = rho_xy * (sigma_x) * (sigma_y)
        covar_matrix_for_all_Gaussians[:, :, :, 0, 1] = covar
        covar_matrix_for_all_Gaussians[:, :, :, 1, 0] = covar
        Gaussian_distributions = torch.distributions.MultivariateNormal(loc=mu_for_all_Gaussians, covariance_matrix=covar_matrix_for_all_Gaussians)
        
        # p_dist_probs = torch.nn.functional.softmax(q_logits_params / temperature, dim=-1)
        # p_distribution = torch.distributions.Categorical(probs=p_dist_probs)

        p_distribution = torch.distributions.Categorical(logits=q_logits_params / temperature)
        
        return Gaussian_distributions, mixing_coefficients, p_distribution, state
    
    def sample(self, z_embedding: torch.Tensor, temperature: float = 0.4, max_len: int = 200) -> torch.Tensor:
        """Generate a new sketch from the z_embedding.

        Args:
            z_embdding (torch.Tensor): Latent representation. Shape (1, d_z)
            temperature (float, optional): Temperature to control randomness. Defaults to 0.4.
            max_len (int, optional): Maximum generated length to prevent forever generation. Defaults to 200.

        Returns:
            torch.Tensor: Generated stroke sequence. Shape (generated_length, 5)
        """        
        self.eval()
        # check the input
        assert z_embedding.shape == (1, self.d_z)
        
        current_device = next(self.parameters()).device
        
        generated_stroke_sequence: torch.Tensor
        
        # first, should include the SOS "stroke"
        generated_stroke_sequence = torch.Tensor([0, 0, 1, 0, 0]).unsqueeze(0).to(current_device)
        
        decoder_input_stroke: torch.Tensor = generated_stroke_sequence[0].unsqueeze(0).unsqueeze(0)
        state = None

        for i in range(max_len):
            
            ## feed the curent token through the decoder for one time step
            assert decoder_input_stroke.shape == (1, 1, 5)
            Gaussian_distributions, mixing_coefficients, p_distribution, state = self.forward(z_embedding=z_embedding, input_strokes=decoder_input_stroke, input_state=state, temperature=temperature)
            
            ## sample
            
            # create the next token
            next_stroke: torch.Tensor
            
            # sample from p_distribution
            p_idx_sampled: int = p_distribution.sample()[0, 0].item()
            
            # sample from the mixing_coefficients
            random_Gaussian_idx: int = mixing_coefficients.sample()[0, 0].item()
            xx, yy = Gaussian_distributions.sample()[0, 0, random_Gaussian_idx]
            delta_x: float = xx.item()
            delta_y: float = yy.item()
            
            next_stroke = torch.Tensor([delta_x, delta_y, 0, 0, 0]).to(current_device)
            next_stroke[p_idx_sampled + 2] = 1
            assert next_stroke.shape == (5, )
            generated_stroke_sequence = torch.cat((generated_stroke_sequence, next_stroke.unsqueeze(0)), dim=0)
            
            # if found the EOS flag
            if next_stroke[4] == 1:
                break
            else:
                decoder_input_stroke = next_stroke.unsqueeze(0).unsqueeze(0)
                
            
        generated_length, _ = generated_stroke_sequence.shape
        assert generated_stroke_sequence.shape == (generated_length, 5)
        
        return generated_stroke_sequence
        
        


class SketchRNN(torch.nn.Module):
    """SketchRNN model, encapsulating an encoder and a decoder, both of which are recurrent neural networks, as per this paper: A Neural Representation of Sketch Drawings. https://arxiv.org/abs/1704.03477."""
    
    def __init__(self, encoder: "SketchRNNEncoder", decoder: "SketchRNNDecoder", weight_KL):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.weight_KL = weight_KL

    def forward(self, input_strokes, masks):
        """Pass the strokes for ground-truth sketches and the masks (generated after the data preprocessing steps) through the encoder and the decoder to arrive at the final loss value.

        Args:
            input_strokes (torch.Tensor): shape (longest_len, batch_size, 5) (longest_len already includes the SOS token)
            masks (torch.Tensor): the mask for actual strokes - shape(longest_len - 1, batch_size)
            
        Returns:
            final_loss (torch.Tensor): weighted among the reconstruction loss the KL loss, as per the paper.
        """        
        
        ## Check the inputs
        longest_len, batch_size, dim3 = input_strokes.shape
        assert dim3 == 5
        assert masks.shape == (longest_len - 1, batch_size)
        
        ### pass through encoder
        
        # z_embedding (batch_size, d_z) - embeddings for all stroke sequences in a batch 
        z_embedding: torch.Tensor
        # mu_z (batch_size, d_z) - mu for the distribution of the z-embedding
        mu_z = torch.Tensor
        # sigma_z (batch_size, d_z) - sigma for the distribution of the z-embedding
        sigma_z = torch.Tensor
        z_embedding, mu_z, sigma_z = self.encoder(input_strokes, masks)
        # assert z_embedding.shape == (batch_size, d_z)
        # assert mu_z.shape == (batch_size, d_z)
        # assert sigma_z.shape == (batch_size, d_z)
        assert z_embedding.shape == mu_z.shape == sigma_z.shape == (batch_size, self.encoder.d_z)

        ### pass through decoder
    
        # output
        # MultivariateNormal distributions with loc (longest_len-1, batch_size, num_Gaussians, 2) std dev (longest_len-1, batch_size, num_Gaussians, 2, 2): a "tensor" of 2-D Gaussian distributions (to sample delta_x, delta_y on generation)
        Gaussian_distributions: torch.distributions.MultivariateNormal
        # Categorical distributions with logits shape (longest_len-1, batch_size, num_Gaussians): a "tensor" of Categorical distributions for the "mixing coefficients" or "pi" values
        mixing_coefficients: torch.distributions.Categorical
        # Categorical distributions with logits shape (longest_len-1, batch_size, 3): a "tensor" of Categorical distributions for "q logits" value (to sample p1, p2, p3 on generation)
        p_distribution: torch.distributions.Categorical
        
        Gaussian_distributions, mixing_coefficients, p_distribution, _ = self.decoder(z_embedding, input_strokes[:-1])
        
        assert isinstance(Gaussian_distributions, torch.distributions.MultivariateNormal)
        assert Gaussian_distributions.loc.shape == (longest_len - 1, batch_size, self.decoder.num_Gaussians, 2)
        assert Gaussian_distributions.covariance_matrix.shape == (longest_len - 1, batch_size, self.decoder.num_Gaussians, 2, 2)
        assert isinstance(mixing_coefficients, torch.distributions.Categorical)
        assert mixing_coefficients.logits.shape == (longest_len - 1, batch_size, self.decoder.num_Gaussians)
        assert isinstance(p_distribution, torch.distributions.Categorical)
        assert p_distribution.logits.shape == (longest_len - 1, batch_size, 3)

        ### pass through loss calculators
    
        # loss_p, loss_strokes, loss_KL (batch_size, )
        loss_p: torch.Tensor
        loss_strokes: torch.Tensor
        loss_KL: torch.Tensor
        
        loss_strokes = self._calculate_loss_strokes(input_strokes, masks, Gaussian_distributions, mixing_coefficients)
        loss_p = self._calculate_loss_p(input_strokes, masks, p_distribution)
        loss_KL = self._calculate_loss_kl(mu_z, sigma_z)

        assert loss_p.shape == loss_strokes.shape == loss_KL.shape == (batch_size, )
        
        return loss_strokes, loss_p, loss_KL
        
    
    def sample(self, input_stroke_sample: torch.Tensor, mask_sample: torch.Tensor) -> torch.Tensor:
        """Pass one sample through the model to generate a new sketch. The sample stroke sequence is fed for the encoder to learn latent representation. The decoder then generates a sketch that should semantically be similar to the input stroke sequence.

        Args:
            input_stroke_sample (torch.Tensor): One sample of the ground-truth stroke sequence (to be passed first into the encoder). Shape (seq_len, 5)
            mask_sample (torch.Tensor): One sample of the mask associated with that stroke sequence. Shape (seq_len - 1, )

        Returns:
            torch.Tensor: generated stroke sequence. Shape (generated_length, 5).
        """        
                
        with torch.no_grad():
            
            self.eval()
            # pass through the encoder
            # need to reshape input_stroke for the encoder
            # output: need the z_embedding, shaped (1, d_z)
            z_embedding: torch.Tensor
            input_stroke_sample_reshaped = input_stroke_sample.unsqueeze(1)
            mask_sample_reshaped = mask_sample.unsqueeze(1)
            seq_len, _, _ = input_stroke_sample_reshaped.shape

            _, _, z_embedding = self.encoder.forward(input_stroke_sample_reshaped, mask_sample_reshaped)
            assert z_embedding.shape == (1, self.encoder.d_z)
            
            # pass the z_embedding + SOS "stroke" through the decoder, step by step
            # accumulate the stroke sequence one step at a time, until encountering p3=1
            # output: stroke sequence (generated_length, 5)
            generated_stroke_sequence: torch.Tensor
            generated_stroke_sequence = self.decoder.sample(z_embedding, max_len=round(seq_len*1.2), temperature=0.4)
            generated_length, _ = generated_stroke_sequence.shape
            assert generated_stroke_sequence.shape == (generated_length, 5)
            
            return generated_stroke_sequence
        
    @staticmethod    
    def plot(seq: torch.Tensor, savefig_file_path: str):
        import matplotlib.pyplot as plt
        # Take the cumulative sums of $(\Delta x, \Delta y)$ to get $(x, y)$
        seq = seq.detach()
        seq[:, 0:2] = torch.cumsum(seq[:, 0:2], dim=0)
        # Create a new numpy array of the form $(x, y, q_2)$
        seq[:, 2] = seq[:, 3]
        seq = seq[:, 0:3].detach().cpu().numpy()

        # Split the array at points where $q_2$ is $1$.
        # i.e. split the array of strokes at the points where the pen is lifted from the paper.
        # This gives a list of sequence of strokes.
        strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
        plt.clf()
        plt.axis('equal')
        # Plot each sequence of strokes
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        # Don't show axes
        plt.axis('on')
        # Show the plot
        # plt.show()
        # from datetime import datetime

        # Get current timestamp in a formatted string
        # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Adjust format as needed (YYYY-MM-DD_HH-MM-SS)
        
        # Check if the parent directory exists (one level up from the file)
        if not os.path.exists(os.path.dirname(savefig_file_path)):
            # Create the parent directory if it doesn't exist
            os.makedirs(os.path.dirname(savefig_file_path))

        # Save the plot with the timestamped filename
        plt.savefig(savefig_file_path)
            
    
    def _calculate_loss_strokes(self, input_strokes, masks, Gaussian_distributions: torch.distributions.MultivariateNormal, mixing_coefficients: torch.distributions.Categorical):
        """Calculate the "strokes" loss - the log-likelihood of the Gaussian mixture explaining the original ground-truth data"""
        
        # Check the inputs
        longest_len, batch_size, dim3 = input_strokes.shape
        assert dim3 == 5
        assert masks.shape == (longest_len - 1, batch_size)
        
        # input_data # extracted the delta_x, delta_y part: (longest_len - 1, batch_size, 2)
        strokes_deltax_deltay = input_strokes[1:, :, 0:2]
        assert strokes_deltax_deltay.shape == (longest_len -1, batch_size, 2)
        
        # repeat the delta_x, delta_y data (in each time step of each sequence in the batch) to feed the same copy into all Guassians (of each time step in each sequence)
        num_Gaussians = Gaussian_distributions.loc.shape[-2]
        strokes_deltax_deltay_repeated = strokes_deltax_deltay.unsqueeze(-2).expand(-1, -1, num_Gaussians, -1)
        assert strokes_deltax_deltay_repeated.shape == (longest_len - 1, batch_size, num_Gaussians, 2)
        
        # compute the log-probabilities (each stroke) based on Gaussian mixture (taking into account the mask) - for each sequence in batch
        # log_probs (longest_len - 1, batch_size)
        log_probs: torch.Tensor
        
        # weighted probability of explaining the delta_x and delta_y for each time step for each sequence in the batch
        weighted_probs = torch.sum(mixing_coefficients.probs * torch.exp(Gaussian_distributions.log_prob(strokes_deltax_deltay_repeated)), dim=-1)
        assert weighted_probs.shape == (longest_len - 1, batch_size)
        
        log_probs = (masks * torch.log(weighted_probs + 1e-6))
        
        assert log_probs.shape == (longest_len -1, batch_size)
        
        # sum over all timesteps, then normalized: dividing by longest sequence (# of actual strokes), which is longest_len - 1
        # loss_strokes (batch_size, )
        loss_strokes: torch.Tensor
        loss_strokes = -torch.mean(log_probs, dim=0)
        assert loss_strokes.shape == (batch_size, )
        return loss_strokes
        
    def _calculate_loss_p(self, input_strokes, masks, p_distribution: torch.distributions.Categorical):
        """Calculate the "p-value" loss - the log-likelihood of the Gaussian mixture explaining the original ground-truth p-values part of the data"""
        
        # Check the inputs
        longest_len, batch_size, dim3 = input_strokes.shape
        assert dim3 == 5
        assert masks.shape == (longest_len - 1, batch_size)
        
        # input_data - extracted the p1, p2, p3 part. (longest_len - 1, batch_size, 3)
        p_values = input_strokes[1:, :, 2:]
        assert p_values.shape == (longest_len -1, batch_size, 3)
        
        # Cross-entropy vs. the q_logits (from decoder)
        # log_probs (longest_len - 1, batch_size): log probability for each p-value for each timestep (stroke), for each sequence in batch
        log_probs = torch.sum(p_values * torch.log_softmax(p_distribution.logits, dim=-1), dim=-1)
        assert log_probs.shape == (longest_len - 1, batch_size)
        assert torch.isnan(log_probs).any() == False
        
        # sum over all time steps, then normalized: dividing by longest sequence (# of actual strokes), which is longest_len - 1
        # loss_p (batch_size, )
        loss_p = -torch.mean(log_probs, dim=0)
        assert loss_p.shape == (batch_size, )
        return loss_p
    
    def _calculate_loss_kl(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Calculate the KL loss between the latent space distribution learned by the encoder vs. N(0, I) because we want the latent space to be close to N(0, I)."""
        # check the input
        batch_size, d_z = mu.shape
        assert mu.shape == sigma.shape

        sigma_hat = torch.log(sigma) * 2 
        assert sigma_hat.shape == (batch_size, d_z) 
        
        loss_kl = (-0.5) * torch.mean( (1 + sigma_hat - mu**2 - sigma**2), dim=-1) 
        assert loss_kl.shape == (batch_size, )
        
        return loss_kl