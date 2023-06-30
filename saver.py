import os
import torch


class Saver:
    def __init__(self, model: torch.nn.Module, path: str = None):
        '''
        Saver class to save the model and the loss values

        Parameters
        ----------
        model : torch.nn.Module
            The model to save
        path : str, optional
            The path where to save the model and the loss values, if None the current working directory is used, by default None
        '''
        self.path = path
        if self.path is None:
            self.path = os.getcwd()
        self.model_count = 0
        self.model = model

    def load(self):
        models_dir = os.path.join(self.path, 'models')

        if os.path.exists(models_dir):
            models = [model for model in os.listdir(
                models_dir) if model.endswith('.pth')]
            numbers = [int(model.split('_')[1].split('.')[0]) for model in models]
            numbers.sort()
            if len(numbers) > 0:
                last_model = numbers[-1]
                model_path = os.path.join(models_dir, f'model_{last_model}.pth')
                self.model_count = last_model
                self.model.load_state_dict(torch.load(model_path))
                print(f'Loaded model {model_path}')

        print(f'Current model {self.model_count}')
        self.model_count += 1

    def save_model(self):
        models_dir = os.path.join(self.path, 'models')
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        model_path = os.path.join(
            models_dir, f'model_{self.model_count}.pth')
        print(f'Saving model {model_path}')
        torch.save(self.model.state_dict(), model_path)
        self.model_count += 1

    def save_loss(self, loss: float, epoch: int):
        loss_path = os.path.join(self.path, 'loss.csv')
        if not os.path.exists(loss_path):
            with open(loss_path, 'w') as f:
                f.write('model,epoch,loss\n')

        with open(loss_path, 'a') as f:
            f.write(f'{self.model_count},{epoch},{loss}\n')

    def save(self, loss: float, epoch: int):
        '''
        Save the model and the loss values

        Parameters
        ----------
        loss : float
            The loss value
        epoch : int
            The epoch number
        '''
        self.save_model()
        self.save_loss(loss, epoch)

    def read_loss(self):
        loss_path = os.path.join(self.path, 'loss.csv')
        if not os.path.exists(loss_path):
            return None

        results = []
        with open(loss_path, 'r') as f:
            for line in f.readlines()[1:]:
                model, epoch, loss = line.split(',')
                results.append((int(model), int(epoch), float(loss)))

        return results
