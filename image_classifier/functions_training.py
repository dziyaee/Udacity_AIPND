# Imports here

# General imports
import matplotlib.pyplot as plt
import numpy as np
import time

# Pytorch imports
import torch


# Forward Pass + Loss/Acc calculations function
def fpass(images, labels, model, criterion):
    ''' Function to calculate loss and accuracy of predictions of images and labels evaluated against a given network

        Args:
            images (tensor): batch of images of shape (B x 3 x 224 x 224), where B = Batch Size
            labels (tensor): batch of labels of shape (B)
            model (pytorch model): pretrained net with custom classifier. Accepts inputs of shape (B x 3 x 224 x 224), and produces outputs of shape (B x C), where C = number of classes
            criterion (pytorch loss function): Negative Log-Likelihood Loss. Accepts log probabilities of shape (B x C), and labels of shape (B), and outputs a single loss value

        Returns:
            loss (tensor): scalar representing the computed batch loss
            acc (tensor): scalar representing the computed batch accuracy
    '''

    # Calculate Log Probabilities via Forward pass
    log_ps = model.forward(images)

    # Calculate loss via Criterion
    loss = criterion(log_ps, labels)

    # Calculate Probabilities from Log Probabilities
    ps = torch.exp(log_ps)

    # Calculate top-k classes via .topk()
    top_p, top_class = ps.topk(1, dim=1)

    # Calculate success of top-1 predictions by checking for equality with labels; reshape labels to same shape as top_class
    success = torch.eq(top_class, labels.view(top_class.shape))

    # Calculate accuracy by taking the mean of the success vector; force success to type torch.cuda.FloatTensor as required by torch.mean
    acc = torch.mean(success.type(torch.FloatTensor))

    return loss, acc


# Save checkpoint function
def save_checkpoint(net, hp, track, data, i, loss, acc, checkpoint_path):
    ''' Function to save a Checkpoint Dict containing the Net Dict, HP Dict, Track Dict, State_Dict of Model, Update Index, Min Validation Loss, Corresponding Validation Accuracy, and Class to Index dict of current dataset

        Args:
            net (dict): Net Dict
            hp (dict): Hyper Params Dict
            track (dict): Tracker Dict
            data (dict): Data Dict
            i (int): Update index at which the minimum validation loss occurred which prompted the checkpoint save
            loss (float): Validation Loss
            acc (float): Validation Accuracy
            checkpoint_path (string): path used to save the checkpoint

        Returns:
            None
    '''

    checkpoint = {"net": net,
                  "hp": hp,
                  "track": track,
                  "state_dict": net['model'].state_dict(),
                  "update_index": i,
                  "min_valid_loss": loss,
                  "valid_acc": acc,
                  "class_to_idx": data['train']['dataset'].class_to_idx
                 }

    torch.save(checkpoint, checkpoint_path)

    return None


# Training & Validation Main Function
def train_net(data, hp, track, net, dev, checkpoint_path, batch_prints, valid_prints, epoch_prints, plots):
    ''' Function to train and validate a classifier used within a pretrained net with frozen params

        Args:
            data (dict): Data Dict used to get the Training and Validation DataLoaders and the number of batches per dataset
            hp (dict): Hyper Params Dict used to get the Number of Epochs to train for
            track (dict): Tracking Dict used to store and track loss/acc values per data set
            net (dict): Net Dict used to get the model, criterion, and optimizer to use for training and validation
            dev (string): chosen device to use for training ("cuda" or "cpu")
            checkpoint_path (string): desired path for checkpoint saving
            batch_prints, valid_prints, epoch_prints (booleans): toggle print statements on or off for corresponding iterations
            plots (boolean): toggle loss/acc plots at the end of training on or off

        Returns:
            track (dict): Tracking Dict
    '''

    start_time = time.time()

    # Assign chosen device
    device = torch.device(dev)

    # Network: model, criterion, and optimizer. Assign model to current device
    model = net['model']
    criterion = net['criterion']
    optimizer = net['optimizer']
    model.to(device)

    # Hyper Params: Number of Epochs
    n_epochs = hp['n_epochs']

    # Data: Training DataLoader, Validation DataLoader
    train_loader = data['train']['dataloader']
    valid_loader = data['valid']['dataloader']

    # Number of batches / total batches per train dataset
    n_train_batches = data['train']['n_batches']
    n_total_train_batches = n_epochs * n_train_batches

    # Number of batches / number of validations / total batches per valid dataset
    n_valid_batches = data['valid']['n_batches']
    n_valids = int(np.floor(n_total_train_batches / n_valid_batches))
    n_total_valid_batches = n_valids * n_valid_batches

    # i = Total Training Batch Index
    i = 0

    # j = Total Validation Batch Index
    j = 0

    # v = Validation Index
    v = 0

    # Number of Validation Batches also assigned to b to be used as index
    b = n_valid_batches

    # e = Epoch Index
    for e in range(n_epochs):

        epoch_time = time.time()

        print(f"\nEpoch {e + 1}/{n_epochs}\n")

        # Reset training running loss/acc per epoch
        train_running_loss = 0
        train_running_acc = 0

        print("Training...")

        # Iterate through training DataLoader
        for images, labels in train_loader:

            batch_time = time.time()

            # Set model to training mode
            model.train()

            # Set gradients to zero
            optimizer.zero_grad()

            # Assign image and label batch tensors to chosen device
            images = images.to(device)
            labels = labels.to(device)

            # Compute batch loss and acc using custom fpass function. Loss is computed via Negative Log-Likelihood Loss Criterion
            loss, acc = fpass(images, labels, model, criterion)

            # Compute gradients via Back Prop using calculated loss
            loss.backward()

            # Update weights using computed gradients
            optimizer.step()

            # Update training running loss / acc
            train_running_loss += loss.item()
            train_running_acc += acc.item()

            # Store training batch loss / acc
            track['train']['batch_loss'][i] = loss.item()
            track['train']['batch_acc'][i] = acc.item()

            if batch_prints:

                print(f"Training Batch {i + 1}/{n_total_train_batches}: Batch Loss = {loss.item():.4f}... Batch Acc = {(acc.item() * 100):.1f}%... Batch Time = {(time.time() - batch_time):.1f}s... Elapsed Time = {(time.time() - start_time):.1f}s")

            # Check if validation should occur. Next index i (i+1) is checked due to pythonic indexing starting at 0. Example, if we want to validate every 10 updates, since update 1 occured at i = 0, update 10 occurs at i = 9. Therefore, we want to validate when i = 9 such that (i + 1) & 10 = (9 + 1) % 10 = 0
            if (i + 1) % b == 0:

                print("\nValidating...")

                # Reset validation running loss/acc per validation
                valid_running_loss = 0
                valid_running_acc = 0

                # Proceed with no gradient calculations for the entire validation set
                with torch.no_grad():

                    # Iterate through validation DataLoader
                    for images, labels in valid_loader:

                        batch_time = time.time()

                        # Set model to evaluation mode
                        model.eval()

                        # Assign image and label batch tensors to chosen device
                        images = images.to(device)
                        labels = labels.to(device)

                        # Compute batch loss and acc using custom fpass function. Loss is computed via Negative Log-Likelihood Loss Criterion
                        loss, acc = fpass(images, labels, model, criterion)

                        # Update validation running loss / acc
                        valid_running_loss += loss.item()
                        valid_running_acc += acc.item()

                        # Store validation batch loss / acc
                        track['valid']['batch_loss'][j] = loss.item()
                        track['valid']['batch_acc'][j] = acc.item()

                        if batch_prints:

                            print(f"Validation Batch {j + 1}/{n_total_valid_batches}: Batch Loss = {loss.item():.4f}... Batch Acc = {(acc.item() * 100):.1f}%... Batch Time = {(time.time() - batch_time):.1f}s... Elapsed Time = {(time.time() - start_time):.1f}s")

                        # End Validation Batch, iterate total validation batch index j
                        j += 1

                # Calculate average validation batch loss / acc per Validation
                track['valid']['av_loss'][v] = valid_running_loss / b
                track['valid']['av_acc'][v] = valid_running_acc / b

                # Calculcate average training batch loss / acc per Validation. Example, if i = 9, b = 10, we want the calculate the average training loss/acc over the last 10 training batches. This would mean indexing from [0: 10] = [(9 + 1) - 10: 10] = [(i + 1) - b: (i + 1_]
                track['train']['av_loss'][v] = np.mean(track['train']['batch_loss'][(i + 1) - b: (i + 1)])
                track['train']['av_acc'][v] = np.mean(track['train']['batch_acc'][(i + 1) - b: (i + 1)])

                # On first validation, store validation loss and save checkpoint
                if v == 0:

                    min_valid_loss = track['valid']['av_loss'][v]
                    min_valid_loss_i = i
                    save_checkpoint(net, hp, track, data, i, track['valid']['av_loss'][v], track['valid']['av_acc'][v], checkpoint_path)
                    print(f"\nNew Minimum Validation Loss achieved; Checkpoint saved to: {checkpoint_path}")

                # On subsequent validations, update min validation loss and save checkpoint only if current min loss lower than previous min loss
                elif track['valid']['av_loss'][v] < min_valid_loss:

                    min_valid_loss = track['valid']['av_loss'][v]
                    min_valid_loss_i = i

                    # Save Checkpoint
                    save_checkpoint(net, hp, track, data, i, track['valid']['av_loss'][v], track['valid']['av_acc'][v], checkpoint_path)
                    print(f"\nNew Minimum Validation Loss achieved! Checkpoint saved to: {checkpoint_path}")

                else:

                    print("\nValidation Loss higher than minimum; skipping Checkpoint...")

                if valid_prints:

                    print(f"\nValidation {v + 1}/{n_valids} Complete... Elapsed Time = {(time.time() - start_time):.1f}s")
                    print("Validation: Av Loss = {:.4f}... Av Acc = {:.1f}%...".format(track['valid']['av_loss'][v], track['valid']['av_acc'][v] * 100))
                    print("Training:   Av Loss = {:.4f}... Av Acc = {:.1f}%...\n".format(track['train']['av_loss'][v], track['train']['av_acc'][v] * 100))
                    print("Training...")

                # End Validation, iterate validation index v
                v += 1

            # End Training Batch, iterate total training batch index i
            i += 1

        # Calculate Training Loss / Acc per Epoch
        track['train']['epoch_loss'][e] = train_running_loss / n_train_batches
        track['train']['epoch_acc'][e] = train_running_acc / n_train_batches

        if epoch_prints:

            print("\nEpoch {}/{} Complete... Training Epoch Loss = {:.4f}... Training Epoch Acc = {:.1f}%... Epoch Time = {:.1f}s... Elapsed Time = {:.1f}s".format(e + 1, n_epochs, track['train']['epoch_loss'][e], track['train']['epoch_acc'][e] * 100, time.time() - epoch_time, time.time() - start_time))

        # End Epoch, (epoch index e is iterated implicitly)

    print(f"\nTraining Complete. Best Validation Average Loss = {min_valid_loss:.4f} at update {min_valid_loss_i + 1}")

    # Plot training batch, training average, and validation average loss/acc if chosen
    if plots:

        plot_stats(track, n_train_batches, n_total_train_batches, n_valid_batches)

    return track


# Training & Validation loss/acc plotting function
def plot_stats(track, n_train_batches, n_total_train_batches, n_valid_batches):
    ''' Function to plot stats from a training / validation run using a track dict

        Args:
            track (dict): Track Dict containing Training Batch Loss/Acc, Training & Validation Average Loss/Acc, Validation Update Range
            n_train_batches (int): Number of Training Batches
            n_total_train_batches (int): Number of Total Training Batches over all epochs
            n_valid_batches (int): Number of validation batches per validation

        Returns:
            None
    '''

    fig, ax = plt.subplots(2, figsize = (20, 10), sharex=True)

    # Dict containing loss and accuracy multipliers (Due to plotting accuracy * 100)
    X = {'loss': 1, 'acc': 100}

    # Validation update indices range
    valid_range = np.arange(n_valid_batches, n_total_train_batches + 1, n_valid_batches)

    # Min validation loss and corresponding update index. [0][0] indexing is used just to get the int from the array
    min_loss = np.min(track['valid']['av_loss'])
    min_loss_i = ((np.where(track['valid']['av_loss'] == min_loss)[0][0]) + 1) * n_valid_batches

    # Max validation acc and corresponding update index. [0][0] indexing is used just to get the int from the array
    max_acc = np.max(track['valid']['av_acc'])
    max_acc_i = ((np.where(track['valid']['av_acc'] == max_acc)[0][0]) + 1) * n_valid_batches

    # Tuple of min_loss and max_acc, and tuple of their indices
    best_stat = (min_loss, max_acc)
    best_i = (min_loss_i, max_acc_i)

    # Labels and legends
    legend = ['Training Batch Loss', 'Training Average Loss', 'Validation Average Loss']
    ylab = ['Loss', 'Accuracy']
    xlab = ['', 'Update Index']

    # Loss through enumerated X dict. i = 0, 1. d = X
    for i, d in enumerate(X.items()):

        # x = 'loss', 'acc'. m = 1, 100
        x, m = d

        # Plot training batch, training av, validation av loss/acc. loss is mulitplied by 1, acc is multiplied by 100
        ax[i].plot(track['train']['batch_' + x] * m, lw=0.7)
        ax[i].plot(valid_range, track['train']['av_'+ x] * m, lw=3)
        ax[i].plot(valid_range, track['valid']['av_'+ x] * m, lw=3)

        # Set legends and labels
        ax[i].legend(legend)
        ax[i].set_ylabel(ylab[i], fontsize=20)
        ax[i].set_xlabel(xlab[i], fontsize=20)

        # Plot red dashed lines to make it easier to see the best loss/acc locations on each graph
        ax[i].axvline(best_i[i], c='r', ls='--')
        ax[i].axhline(best_stat[i] * m, c='r', ls='--')

        # Set xticks to represent the epoch update indices to make it easier to see where it epoch occured and turn on grid
        epoch_indices = np.arange(0, n_total_train_batches + 1, n_train_batches)
        ax[i].set_xticks(epoch_indices)
        ax[i].grid()

    # Titles
    ax[0].set_title(f"Min Average Validation Loss = {min_loss:.4f} || Update = {min_loss_i}", fontsize=15)
    ax[1].set_title(f"Max Average Validation Acc = {(max_acc * 100):.1f}% || Update = {max_acc_i}", fontsize=15)

    # ylims and yticks for loss graph
    ax[0].set_ylim(0, 5)
    ax[0].set_yticks(np.arange(0, 5 + 0.5, 0.5))

    # ylims, yticks, and yticklabels for acc graph
    ax[1].set_ylim(0, 100)
    ax[1].set_yticks(np.arange(0, 100 + 10, 10))
    y_tick_labs = [str(int(y)) + '%' for y in ax[1].get_yticks()]
    ax[1].set_yticklabels(y_tick_labs)

    fig.tight_layout()

    plt.show(block=True)

    return None


