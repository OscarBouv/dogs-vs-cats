from utils import AverageMeter
import torch
import numpy as np
from tqdm import tqdm

class TrainingSession:

    """
        Class for running training session.
    """

    def __init__(self, model, train_loader, val_loader,
                 optimizer, epochs, device,
                 model_path):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.epochs = epochs

        self.device = device

        self.model_path = model_path

    def train_one_epoch(self, criterion):

        """
            Training procedure for one epoch.
        """

        losses = AverageMeter()
        acc = AverageMeter()

        self.model.train()

        with tqdm(self.train_loader, unit="batch") as tepoch:

            for (x, y) in tepoch:

                # SUPERVISED TRAIN EPOCH
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                loss = criterion(output, y)

                # Compute accuracy
                y_pred = output.argmax(dim=1, keepdim=True).squeeze()
                accuracy = (y_pred == y).sum().item() / x.size(0)

                # measure record loss and accuracies
                losses.update(loss.item(), x.size(0))
                acc.update(accuracy, x.size(0))

                # Update progress bar with average train loss and accuracy
                tepoch.set_postfix(train_loss=losses.avg, train_acc=acc.avg)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            return losses.avg

    def validate(self, criterion, val_loss_min):

        """
            Compute loss and accuracy on validation set.
        """

        losses = AverageMeter()
        accuracies = AverageMeter()

        self.model.eval()

        with torch.no_grad():

            with tqdm(self.val_loader, unit="batch") as tepoch:

                for (x_val, y_val) in tepoch:

                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    # compute output
                    output = self.model(x_val)
                    loss = criterion(output, y_val)

                    output = output.float()
                    loss = loss.float()

                    # measure accuracy and record loss
                    y_pred = output.argmax(dim=1, keepdim=True).squeeze()
                    accuracy = (y_pred == y_val).sum().item() / x_val.size(0)

                    losses.update(loss.item(), x_val.size(0))
                    accuracies.update(accuracy, x_val.size(0))

                val_loss = losses.avg

                if val_loss <= val_loss_min:

                    # Save model
                    print(f'Val loss decreased {val_loss_min} --> {val_loss}.')
                    print(f'Saving model at {self.model_path}.')
                    torch.save(self.model.state_dict(), self.model_path)

                    # Update min validation loss
                    val_loss_min = val_loss

        return accuracies.avg, val_loss, val_loss_min

    def run(self):

        """
            Run training session.
        """

        # Load self.model
        self.model.to(self.device)

        # Criterion
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Initialize min validation loss
        val_loss_min = np.inf

        # Supervised train loop
        for epoch in range(self.epochs):

            self.train_one_epoch(criterion)

            # Evaluate on validation set
            val_acc, val_loss, val_loss_min = self.validate(criterion, val_loss_min)

            print(f'Acc/valid at epoch {epoch} : {val_acc}')
            print(f'Loss/valid at epoch {epoch} : {val_loss}')

        print("End of training.")


if __name__ == "__main__":
    pass
