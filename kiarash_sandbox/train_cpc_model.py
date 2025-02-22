# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for training and evaluating a CPC model.

"""

import numpy as np
import time
import sys
import os
import pickle
from importlib.machinery import SourceFileLoader
from copy import deepcopy
from torch import cuda, no_grad, save, load
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm 

from utils import convert_py_conf_file_to_text
# from utils import visualize_tsne
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('Usage: \n1) python train_cpc_model.py \nOR \n2) python train_cpc_model.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        # import conf_train_cpc_model_orig_implementation as conf
        # conf_file_name = 'conf_train_cpc_model_orig_implementation.py'
        from confs import conf_train_ldmcpc_model as conf
        conf_file_name = 'kiarash_sandbox/confs/conf_train_ldmcpc_model.py'
    except ModuleNotFoundError:
        sys.exit('''Usage: \n1) python train_cpc_model.py \nOR \n2) python train_cpc_model.py <configuration_file>\n\n
        By using the first option, you need to have a configuration file named "conf_train_cpc_model.py" in the same directory 
        as "train_cpc_model.py"''')


# Import our models
from torch.utils.data import DataLoader, Subset
from encoding_models.cpc_model import CPC_encoder_mlp, CPC_autoregressive_model, CPC_postnet, LogisticRegression
from cpc_data_loader import CPCDataset
from pytorch_utils.loss.contrastive_loss import CPCLoss

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)

# Import our learning rate scheduler
if conf.use_lr_scheduler:
    scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler]), conf.lr_scheduler)

if __name__ == '__main__':
    # Open the file for writing
    file = open(conf.name_of_log_textfile, 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
        
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')

    # Initialize our models
    Encoder = CPC_encoder_mlp(**conf.encoder_params)
    AR_model = CPC_autoregressive_model(**conf.ar_model_params)
    W = CPC_postnet(**conf.w_params)
    
    # Pass the models to the available device
    Encoder = Encoder.to(device)
    AR_model = AR_model.to(device)
    W = W.to(device)
    
    enconder_num_params = sum(p.numel() for p in Encoder.parameters() if p.requires_grad)
    ar_num_params = sum(p.numel() for p in AR_model.parameters() if p.requires_grad)
    w_num_params = sum(p.numel() for p in W.parameters() if p.requires_grad)
    total_num_params = enconder_num_params + ar_num_params + w_num_params
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write(f"Number of parameters:\n")
        f.write(f"Encoder #params:  {enconder_num_params:7d} (%{100*enconder_num_params/total_num_params:3.1f})\n")
        f.write(f"AR #params:       {ar_num_params:7d} (%{100*ar_num_params/total_num_params:3.1f})\n")
        f.write(f"W #params:        {w_num_params:7d} (%{100*w_num_params/total_num_params:3.1f})\n")
        f.write(f"Total #params: {total_num_params:7d}\n\n")

    # Give the parameters of our models to an optimizer
    model_parameters = list(Encoder.parameters()) + list(AR_model.parameters()) + list(W.parameters())
    optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
    
    # Get our learning rate for later use
    learning_rate = optimizer.param_groups[0]['lr']
    
    # Give the optimizer to the learning rate scheduler
    if conf.use_lr_scheduler:
        lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params)

    # Instantiate our loss function as a class
    loss_function = CPCLoss(**conf.loss_params)

    # Variables for early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience_counter = 0
    
    if conf.load_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Loading model from file...\n')
            f.write(f'Loading model {conf.encoder_best_model_name}\n')
            f.write(f'Loading model {conf.ar_best_model_name}\n')
            f.write(f'Loading model {conf.w_best_model_name}\n')
        Encoder.load_state_dict(load(conf.encoder_best_model_name, map_location=device, weights_only=True))
        AR_model.load_state_dict(load(conf.ar_best_model_name, map_location=device, weights_only=True))
        W.load_state_dict(load(conf.w_best_model_name, map_location=device, weights_only=True))
        best_model_encoder = deepcopy(Encoder.state_dict())
        best_model_ar = deepcopy(AR_model.state_dict())
        best_model_w = deepcopy(W.state_dict())
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n\n')
    else:
        best_model_encoder = None
        best_model_ar = None
        best_model_w = None
    
    
    # Initialize the data loaders
    from utils import prepare_data
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data()
    training_set = CPCDataset(x=X_train, y=y_train)
    train_data_loader = DataLoader(training_set, **conf.params_train)
    
    validation_set = CPCDataset(x=X_validation, y=y_validation)
    validation_data_loader = DataLoader(validation_set, **conf.params_test)
    
    test_set = CPCDataset(x=X_test, y=y_test)
    test_data_loader = DataLoader(test_set, **conf.params_test)
    
    
    # Determine our timesteps t. In the original CPC paper: "The output of the GRU at every timestep is used as the context c from which we
    # predict 12 timesteps in the future using the contrastive loss". We first determine whether our future_predicted_timesteps is a number
    # or a list of numbers.
    if isinstance(conf.future_predicted_timesteps, int):
        # future_predicted_timesteps is a number, so we have n timesteps where t in [0, 1, ..., n - 1] and our
        # n = num_frames_encoding - future_predicted_timesteps
        timesteps = np.arange(conf.num_frames_encoding - conf.future_predicted_timesteps)
        
        # Define the maximum future predicted timestep
        max_future_timestep = conf.future_predicted_timesteps
        
    elif isinstance(conf.future_predicted_timesteps, list):
        # future_predicted_timesteps is a list of numbers, so we define timesteps based on the largest number in the list
        max_future_timestep = max(conf.future_predicted_timesteps)
        
        # Then determine timesteps
        timesteps = np.arange(conf.num_frames_encoding - max_future_timestep)
        
    else:
        sys.exit('Configuration setting "future_predicted_timesteps" must be either an integer or a list of integers!')
    
    # Flag for indicating if max epochs are reached
    max_epochs_reached = 1
    
    # Record the metrics over epochs
    metrics = {'epoch_loss_training':[],
               'epoch_loss_validation':[],
               'epoch_acc_training':[],
               'epoch_acc_validation':[]}
    
    # Start training our model
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Starting training...\n')
        
        for epoch in range(1, conf.max_epochs + 1):
            start_time = time.time()
    
            # Lists containing the losses of each epoch
            epoch_loss_training = []
            epoch_loss_validation = []
    
            # Indicate that we are in training mode, so e.g. dropout will function
            Encoder.train()
            AR_model.train()
            W.train()
            
            if conf.rnn_models_used_in_ar_model:
                # Initialize the RNN's hidden vector
                hidden = None
            
            # Open the log file for writing
            log_file = open(f"kiarash_sandbox/logs/epochlog_{conf.ar_model_params['type']}_ldmfcst{conf.w_use_ldm_params}_k{max_future_timestep}_{conf.loss_flag}.txt", "w")

            # Store the features/embeddings
            Z_feats_training = []
            C_feats_training = []
            train_labels = []

            # Loop through every batch of our training data
            for train_data in tqdm(train_data_loader, desc=f"Epoch {epoch}/{conf.max_epochs}, training: ", unit="batch", file=log_file):
                    # The loss of the batch
                    loss_batch = 0.0
                    
                    # Get the batches
                    X_input, batch_labels = train_data
                    X_input = X_input.to(device)
                    
                    # Zero the gradient of the optimizer
                    optimizer.zero_grad()
                    
                    # Pass our data through the encoder
                    Z = Encoder(X_input)
                    
                    # Create the output of the AR model. Note that the default AR_model flips the dimensions of Z from the
                    # form [batch_size, num_features, num_frames_encoding] into [batch_size, num_frames_encoding, num_features])
                    if conf.rnn_models_used_in_ar_model:
                        C, hidden = AR_model(Z, hidden)
                    else:
                        C = AR_model(Z)
                    
                    # We go through each timestep one at a time
                    for t in timesteps:
                        
                        # The encodings of the future timesteps, i.e. z_{t+k} where k in [1, 2, ..., max_future_timestep]
                        Z_future_timesteps = Z[:,(t + 1):(t + max_future_timestep + 1),:].permute(1, 0, 2)
                        
                        # c_t is the context latent representation that summarizes all encoder embeddings z_k where k <= t
                        c_t = C[:, t, :]
                        
                        # Each of the predicted future embeddings z_{t+k} where k in [1, 2, ..., max_future_timestep] (or k in
                        # future_predicted_timesteps if future_predicted_timesteps is a list) are computed using the post-net
                        if conf.w_use_ldm_params:
                            # Get the k-step ahead weights calculated based on LDM parameters; W_k = C.A^K
                            weight_matrices = AR_model.get_postnet_weight_matrices(conf.future_predicted_timesteps)
                            # W.set_weights(weight_matrices)
                            predicted_future_Z = W(c_t, weight_matrices)
                        else:
                            predicted_future_Z = W(c_t)
                        # predicted_future_Z = W(c_t)
                        
                        # Compute the loss of our model
                        # if batch_labels != []:
                        #     loss = loss_function(Z_future_timesteps, predicted_future_Z, batch_labels)
                        # else:
                        if conf.w_params['detach']:
                            loss = loss_function(Z_future_timesteps.detach(), predicted_future_Z)
                        else:
                            loss = loss_function(Z_future_timesteps, predicted_future_Z)
                        
                        # Add the loss to the total loss of the batch
                        loss_batch += loss
                        
                    # Perform the backward pass
                    loss_batch.backward()
                    
                    # Update the weights
                    optimizer.step()

                    # Add the loss to the total loss of the batch
                    epoch_loss_training.append(loss_batch.item())


                    # Save the features
                    Z_feats_training.append(Z.mean(axis=1))  #[batch_size, num_frames_encoding, num_features]
                    C_feats_training.append(C.mean(axis=1))  #[batch_size, num_frames_encoding, num_features]
                    # C_feats_training.append(C[:,-1, :])  #[batch_size, num_frames_encoding, num_features]
                    train_labels.append(batch_labels)    
                
            Z_feats_training = torch.cat(Z_feats_training).detach().cpu().numpy()
            C_feats_training = torch.cat(C_feats_training).detach().cpu().numpy()
            train_labels = torch.cat(train_labels).detach().cpu().numpy()

            # Indicate that we are in evaluation mode, so e.g. dropout will not function
            Encoder.eval()
            AR_model.eval()
            W.eval()

            # Store the features/embeddings
            Z_feats_val = []
            C_feats_val = []
            val_labels = []

            # Make PyTorch not calculate the gradients, so everything will be much faster.
            with no_grad():
                
                # Loop through every batch of our validation data and perform a similar process as for the training data
                for validation_data in tqdm(validation_data_loader, desc=f"Epoch {epoch}/{conf.max_epochs}, validation: ", unit="batch", file=log_file):
                    loss_batch = 0.0
                    X_input, batch_labels = validation_data
                    X_input = X_input.to(device)
                    Z = Encoder(X_input)
                    if conf.rnn_models_used_in_ar_model:
                        C, hidden = AR_model(Z, hidden)
                    else:
                        C = AR_model(Z)
                        
                    for t in timesteps:
                        Z_future_timesteps = Z[:,(t + 1):(t + max_future_timestep + 1),:].permute(1, 0, 2)
                        c_t = C[:, t, :]
                        if conf.w_use_ldm_params:
                            # Get the k-step ahead weights calculated based on LDM parameters; W_k = C.A^K
                            weight_matrices = AR_model.get_postnet_weight_matrices(conf.future_predicted_timesteps)
                            # W.set_weights(weight_matrices)
                            predicted_future_Z = W(c_t, weight_matrices)
                        else:
                            predicted_future_Z = W(c_t)
                        # predicted_future_Z = W(c_t)
                        loss = loss_function(Z_future_timesteps, predicted_future_Z)
                        loss_batch += loss
                    epoch_loss_validation.append(loss_batch.item())
                    
                    # Save the features
                    Z_feats_val.append(Z.mean(axis=1))  #[batch_size, num_frames_encoding, num_features]
                    C_feats_val.append(C.mean(axis=1))  #[batch_size, num_frames_encoding, num_features]
                    # C_feats_val.append(C[:, -1])  #[batch_size, num_frames_encoding, num_features]
                    val_labels.append(batch_labels)    
            
            log_file.close()
                
            Z_feats_val = torch.cat(Z_feats_val).cpu().numpy()
            C_feats_val = torch.cat(C_feats_val).cpu().numpy()
            val_labels = torch.cat(val_labels).cpu().numpy()

            # Calculate mean losses
            epoch_loss_training = np.array(epoch_loss_training).mean()
            epoch_loss_validation = np.array(epoch_loss_validation).mean()
            metrics['epoch_loss_training'].append(epoch_loss_training)
            metrics['epoch_loss_validation'].append(epoch_loss_validation)

            # Check early stopping conditions
            if epoch_loss_validation < lowest_validation_loss:
                lowest_validation_loss = epoch_loss_validation
                patience_counter = 0
                best_model_encoder = deepcopy(Encoder.state_dict())
                best_model_ar = deepcopy(AR_model.state_dict())
                best_model_w = deepcopy(W.state_dict())
                best_validation_epoch = epoch
                if conf.save_best_model:
                    save(best_model_encoder, conf.encoder_best_model_name)
                    save(best_model_ar, conf.ar_best_model_name)
                    save(best_model_w, conf.w_best_model_name)
            else:
                patience_counter += 1
            
            # Monitor classification accuracy on validation set
            # Classify speakers
            clf = LogisticRegression(penalty='l2')
            # clf = SVC(C=0.1)
            clf.fit(C_feats_training, train_labels)
            train_labels_predicted = clf.predict(C_feats_training)
            val_labels_predicted = clf.predict(C_feats_val)
            train_acc = accuracy_score(train_labels, train_labels_predicted)
            val_acc = accuracy_score(val_labels, val_labels_predicted)
            metrics['epoch_acc_training'].append(train_acc)
            metrics['epoch_acc_validation'].append(val_acc)

            end_time = time.time()
            epoch_time = end_time - start_time
            
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'Epoch: {epoch:04d} | '
                  f'Mean training loss: {epoch_loss_training:7.4f} | '
                  f'Mean validation loss: {epoch_loss_validation:7.4f} (lowest: {lowest_validation_loss:7.4f}) | '
                  f'training acc: {100*train_acc:3.4f}% | validation acc : {100*val_acc:3.4f}% | '
                  f'Duration: {epoch_time:7.4f} seconds\n')
                
            # We check that do we need to update the learning rate based on the validation loss
            if conf.use_lr_scheduler:
                if conf.lr_scheduler == 'ReduceLROnPlateau':
                    lr_scheduler.step(epoch_loss_validation)
                else:
                    lr_scheduler.step()
                current_learning_rate = optimizer.param_groups[0]['lr']
                if current_learning_rate != learning_rate:
                    learning_rate = current_learning_rate
                    with open(conf.name_of_log_textfile, 'a') as f:
                        f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
            
            # If patience counter is fulfilled, stop the training
            if patience_counter >= conf.patience:
                max_epochs_reached = 0
                break
            
        if max_epochs_reached:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nMax number of epochs reached, stopping training\n\n')
        else:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nExiting due to early stopping\n\n')
        
        if best_model_encoder is None:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nNo best model. The criteria for the lowest acceptable validation loss not satisfied!\n\n')
            sys.exit('No best model, exiting...')
        else:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'\nBest epoch {best_validation_epoch} with validation loss {lowest_validation_loss}\n\n')
        
    # Plot the metrics
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(metrics['epoch_loss_training'], label='Training Loss', color='blue')
    plt.plot(metrics['epoch_loss_validation'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss vs Epochs'); plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(metrics['epoch_acc_training'], label='Training Accuracy', color='green')
    plt.plot(metrics['epoch_acc_validation'], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Accuracy vs Epochs'); plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"epochs_{conf.ar_model_params['type']}_ldmfcst{conf.w_use_ldm_params}.png")

    # Test the model
    if conf.test_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('\n\nStarting testing... \n')
            
        # Load the best version of the model
        if not conf.rand_init:    
            try:
                Encoder.load_state_dict(load(conf.encoder_best_model_name, map_location=device, weights_only=True))
                AR_model.load_state_dict(load(conf.ar_best_model_name, map_location=device, weights_only=True))
                W.load_state_dict(load(conf.w_best_model_name, map_location=device, weights_only=True))
            except (FileNotFoundError, RuntimeError):
                Encoder.load_state_dict(best_model_encoder)
                AR_model.load_state_dict(best_model_ar)
                W.load_state_dict(best_model_w)
        # Randomly initialize the model components
        else:
            Encoder = CPC_encoder_mlp(**conf.encoder_params).to(device)
            AR_model = CPC_autoregressive_model(**conf.ar_model_params).to(device)
            W = CPC_postnet(**conf.w_params).to(device)
        
        # Pass the models to the available device
        Encoder = Encoder.to(device)
        AR_model = AR_model.to(device)
        W = W.to(device)

        # TODO: fix dimensions
        classifier = LogisticRegression(input_dim=128, output_dim=51)
        classifier = classifier.to(device)
        
        # TODO: fix finetuning
        if True:
            Encoder.eval()
            AR_model.eval()
            W.eval()
            # Disable gradient computation for Encoder and AR_model
            for param in Encoder.parameters():
                param.requires_grad = False
            for param in AR_model.parameters():
                param.requires_grad = False

        # Collect parameters (though now they won’t be trainable)
        ftx_params = list(Encoder.parameters()) + list(AR_model.parameters())

        clf_params = list(classifier.parameters())
        ftx_optimizer = torch.optim.SGD(params=ftx_params, lr=1e-4, momentum=0.9, weight_decay=0.01)
        clf_optimizer = torch.optim.SGD(params=clf_params, lr=2e-3, momentum=0.9, weight_decay=0.1)

        loss_fn = torch.nn.CrossEntropyLoss()

        # TODO: fix the number of max_epochs
        for epoch in tqdm(range(100), desc='Training the classifier'):
            train_correct = 0
            train_total = 0
            
            # Encoder.train()
            # AR_model.train()
            # classifier.train()
            for train_data in train_data_loader:
                # The loss of the batch
                loss_batch = 0.0

                # Get the batches
                X_input, batch_labels = train_data
                X_input = X_input.to(device)
                batch_labels = batch_labels.to(device)

                # Zero the gradient of the optimizer
                ftx_optimizer.zero_grad()
                clf_optimizer.zero_grad()

                # Pass our data through the encoder
                Z = Encoder(X_input)
                if conf.rnn_models_used_in_ar_model:
                    C, hidden = AR_model(Z, hidden)
                else:
                    C = AR_model(Z)
                logits = classifier(C[:, -1])

                # Compute loss and backpropagate
                loss_batch = loss_fn(logits, batch_labels)
                loss_batch.backward()
                # ftx_optimizer.step()
                clf_optimizer.step()

                # Compute training accuracy
                predicted_labels = torch.argmax(logits, dim=1)
                train_correct += (predicted_labels == batch_labels).sum().item()
                train_total += batch_labels.size(0)

            # Compute and log training accuracy
            train_accuracy = 100 * train_correct / train_total

            # Compute test accuracy
            test_correct = 0
            test_total = 0
            Encoder.eval()
            AR_model.eval()
            classifier.eval()
            with torch.no_grad():
                for test_data in test_data_loader:
                    # Get batches
                    X_input, batch_labels = test_data
                    X_input = X_input.to(device)
                    batch_labels = batch_labels.to(device)

                    # Forward pass
                    Z = Encoder(X_input)
                    if conf.rnn_models_used_in_ar_model:
                        C, hidden = AR_model(Z, hidden)
                    else:
                        C = AR_model(Z)
                    logits = classifier(C[:, -1])

                    # Compute predictions
                    predicted_labels = torch.argmax(logits, dim=1)
                    test_correct += (predicted_labels == batch_labels).sum().item()
                    test_total += batch_labels.size(0)

            # Compute test accuracy
            test_accuracy = 100 * test_correct / test_total

            # Log accuracies
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f"Epoch {epoch+1}: Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%\n")


    # Define the file path for saving the metrics
    metrics_file = os.path.join('kiarash_sandbox/metrics', f"metrics_{conf.ar_model_params['type']}_ldmfcst{conf.w_use_ldm_params}_{conf.future_predicted_timesteps}_{conf.loss_flag}")

    # Dump the results into the metrics folder
    with open(metrics_file, 'wb') as pickle_file:
        pickle.dump(metrics, pickle_file)

