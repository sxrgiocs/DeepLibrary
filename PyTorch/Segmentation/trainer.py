import os
import time
import numpy as np
import torch
import torch.nn as nn

from src.utils.misc_utils import convert_seconds, get_class_name
from Metrics.DiceCoefficient import DiceCoefficient


class SegmentationTrainer(nn.Module):
    def __init__(self, model, classes, loss=None, metric=None, optimizer=None, learning_rate=1e-4):
        super(SegmentationTrainer, self).__init__()

        # Select the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the model and classes
        self.model = model.to(self.device)
        self.classes = classes

        # Default arguments
        default_loss = nn.BCELoss() if classes == 2 else nn.CrossEntropyLoss()
        default_metric = DiceCoefficient(num_classes=self.classes)
        default_optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Define training arguments
        self.criterion = loss if loss is not None else default_loss
        self.metric = metric if metric is not None else default_metric
        self.optimizer = optimizer if optimizer is not None else default_optimizer
        self.learning_rate = learning_rate
        self.scheduler = None

        # Create a dictionary to store the training parameters
        self.params = {'model': self.model.__class__.__name__,
                       'optimizer': self.optimizer.__class__.__name__,
                       'scheduler': self.scheduler.__class__.__name__ if self.scheduler else None,
                       'loss': get_class_name(self.criterion),
                       'learning_rates': self.learning_rate,
                       'epochs': 0.,
                       'training_loss': 0.,
                       'validation_loss': 0.,
                       'training_time': 0,
                       }

        # Create a dictionary to store the state dicts
        self.state_dicts = {'model': None,
                            'optimizer': None,
                            'scheduler': None}

    def fit(self, trainloader, validloader, epochs):
        # Define empty arrays to store the losses during training
        tr_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)

        # Define empty arrays to store the metrics during training
        tr_metric = np.zeros(epochs)
        val_metric = np.zeros(epochs)

        # Define array to store learning rates if a scheduler is used
        learning_rates = np.zeros(epochs) if self.scheduler else self.learning_rate

        # Define start training time
        start_time = time.time()

        for e in range(int(epochs)):
            # Define empty train loss
            train_loss = 0.
            train_metric = 0.

            # Define start epoch time
            epoch_time = time.time()

            # Set model to training mode
            self.train()

            # Iterate per batch
            for data, target, _ in trainloader:
                # Move the data and target to the device
                data, target = data.to(self.device).float(), target.to(self.device).float()
                # Compute the output of the model
                output = self.model.forward(data)

                # Compute the loss
                loss = self.criterion(output, target)
                train_loss += loss.item() * data.size(0)  # batch loss

                # Reset the gradients
                self.optimizer.zero_grad()
                # Backpropagate gradients
                loss.backward()
                # Update the model parameters
                self.optimizer.step()

                # Compute the metric
                metric = self.metric(output, target)
                train_metric += metric.item() * data.size(0)

            # Append the training loss per epoch
            tr_loss[e] = train_loss / len(trainloader.dataset)
            tr_metric[e] = train_metric / len(trainloader.dataset)

            # Disable gradients for validation
            with torch.no_grad():
                # Create empty validation loss
                valid_loss = 0.
                valid_metric = 0.

                # Set model to evaluation mode
                self.model.eval()

                # Iterate per batch
                for data, target, _ in validloader:
                    # Send the data to the device as with training
                    data, target = data.to(self.device), target.to(self.device)
                    # Compute the output
                    output = self.model.forward(data)
                    # Compute the loss
                    loss = self.criterion(output, target)
                    valid_loss += loss.item() * data.size(0)  # batch loss
                    # Compute the metric
                    metric = self.metric(output, target)
                    valid_metric += metric.item() * data.size(0)

            # Append the validation loss per epoch
            val_loss[e] = valid_loss / len(validloader.dataset)
            val_metric[e] = valid_metric / len(validloader.dataset)

            # Get the learning rate
            if self.scheduler is not None:
                learning_rates[e] = self.scheduler.get_last_lr()
                # Update learning rate if scheduler is not None
                self.scheduler.step()

            # Print information after each epoch
            if (e % 1 == 0):
                epoch_time_str = convert_seconds(time.time() - epoch_time)

                print(f'Epoch [{e + 1}/{epochs}] - {epoch_time_str}\t '
                      f'Loss: {tr_loss[e]:.4f} - Val loss: {val_loss[e]:.4f} | '
                      f'Metric: {tr_metric[e]:.4f} - Val metric: {val_metric[e]:.4f}'
                      )

        # Get the time it has taken to train in HHh MMm SSs format
        train_time = convert_seconds(time.time() - start_time)
        time_per_epoch = convert_seconds((time.time() - start_time) // (e + 1))

        print(f'\nTotal training time: {train_time}')
        print(f'Average time per epoch: {time_per_epoch}')

        # Update the params dictionary
        self.params['training_loss'] = tr_loss
        self.params['validation_loss'] = val_loss
        self.params['training_time'] = train_time
        self.params['epochs'] = e + 1,
        self.params['learning_rates'] = learning_rates

        # Update the state dictionary
        self.state_dicts['model'] = self.state_dict()
        self.state_dicts['optimizer'] = self.optimizer.state_dict()
        scheduler_state_dict = self.scheduler.state_dict() if self.scheduler else None
        self.state_dicts['scheduler'] = scheduler_state_dict

        return tr_loss, val_loss

    def evaluate(self, testloader, classwise=True, save_segmentations=False, save_path=None):
        # Define empty arrays to store the loss during evaluation
        eval_loss = 0.

        # Define empty arrays to store the metric during evaluation
        if classwise is True:
            self.metric.classwise = True
            eval_metric = torch.zeros(self.classes)
        else:
            eval_metric = 0.

        # Set model to evaluation mode
        self.model.eval()

        # Disable gradients
        with torch.no_grad():
            # Iterate per batch
            for data, target, name in testloader:
                # Move the data and target to the device
                data, target = data.to(self.device), target.to(self.device)
                # Compute the output
                output = self.model.forward(data)

                if save_segmentations:
                    assert save_path is not None, 'Empty save path. Cannot save images'

                    # Create the directory if it does not exist
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    # Iterate over batch samples
                    for i in range(data.shape[0]):
                        save_name = name[i].rstrip('.nii.gz')

                        # Convert tensor to NumPy array
                        output_np = output[i].cpu().detach().numpy()

                        # Save it
                        filepath = os.path.join(save_path, save_name + '.npz')
                        np.savez(filepath, output_np)

                # Compute the loss
                loss = self.criterion(output, target)
                eval_loss += loss.item() * data.size(0)

                # Compute the metric coefficient
                metric = self.metric(output, target).cpu()
                eval_metric += metric * data.size(0)

        # Compute the total loss
        loss = eval_loss / len(testloader.dataset)

        # Compute the total metric
        metric = eval_metric / len(testloader.dataset)

        return loss, metric

    def save(self, path, best=False):
        torch.save(self.state_dicts, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        return self
