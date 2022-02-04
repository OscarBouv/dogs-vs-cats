from utils import AverageMeter
from sklearn.metrics import accuracy_score
import torch

class TrainingSession:

    """
        Documentation #TODO
    """

    def __init__(self, model, train_loader, val_loader,
                 optimizer, epochs, device,
                 writer, display_ratio):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.epochs = epochs

        self.device = device
        self.writer = writer

        self.display_ratio = display_ratio

    def train_one_epoch(self, criterion, epoch):

        """Training procedure for one epoch.

        Parameters
        ----------
        criterion : loss criterion for optimization procedure.

        epoch : index of single epoch training procedure.

        Returns
        -------
        self : average losses of training epoch.

        """

        losses = AverageMeter()
        self.model.train()

        for i, (x, y) in enumerate(self.train_loader):

            # SUPERVISED
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)

            loss = criterion(y_pred, y.view(-1, 1).to(torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure record loss
            losses.update(loss.item(), x.size(0))

            if i % self.display_ratio == 0:
                print(f"Epoch: [{epoch}][{i}/{len(self.train_loader)}]")

            self.writer.add_scalar("Loss Train [AVG]", losses.avg, epoch)
            self.writer.flush()

        return losses.avg

    def validate(self, criterion, epoch):

        losses = AverageMeter()
        accuracies = AverageMeter()

        self.model.eval()

        with torch.no_grad():

            for i, (x_val, y_val) in enumerate(self.val_loader):

                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)

                # compute output
                output = self.model(x_val)
                loss = criterion(output, y_val.view(-1, 1).to(torch.float32))

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                y_pred = (output > 0)
                accuracy = accuracy_score(y_pred.cpu().numpy(), y_val.cpu().numpy())

                losses.update(loss.item(), x_val.size(0))
                accuracies.update(accuracy.item(), x_val.size(0))

                if i % self.display_ratio == 0:
                    print(f'Val: [{i}/{len(self.val_loader)}]')

        self.writer.add_scalar("Loss Val [AVG]", losses.avg, epoch)
        self.writer.add_scalar("Accuracy Val [AVG]", accuracies.avg, epoch)

        self.writer.flush()

        return accuracies.avg, losses.avg

    def run_train(self):

        # load self.model
        self.model.to(self.device)

        # # criterion
        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        # Supervised train loop
        for epoch in range(self.epochs):

            trainloss = self.train_one_epoch(criterion, epoch)

            print('Loss/train', trainloss, epoch)

            # evaluate on validation set
            val_acc, val_loss = self.validate(criterion, epoch)

            print(f'Acc/valid at epoch {epoch} : {val_acc}')
            print(f'Loss/valid at epoch {epoch} : {val_loss}')

        self.writer.close()


if __name__ == "__main__":
    pass
