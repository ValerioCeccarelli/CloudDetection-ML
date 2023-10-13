import os
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from typing import Union

# TODO: add a verbose parameter
# TODO: change nn.Module to the real model?
# TODO: add the docstrings

# TODO: add a save for test_loss and validation_loss
# TODO: add a save structure to contain iper-parameter (also the one that not change)


class MySaver:
    def __init__(self, file_name: str, create_if_not_exist: bool = True) -> None:
        if not file_name.endswith(".pth"):
            raise ValueError(f"The file '{file_name}' hould be a .pth file.")

        if not os.path.exists(file_name):
            if create_if_not_exist:
                torch.save([], file_name)
            else:
                raise FileNotFoundError(
                    f"The file '{file_name}' does not exist.")

        content = torch.load(file_name)
        if not isinstance(content, list):
            raise FileNotFoundError(
                f"The file '{file_name}' contains some invalid data.")

        self.__file_name = file_name

        self.__MODEL_STATE_DICT = 'model_state_dict'
        self.__OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
        self.__SCHEDULER_STATE_DICT = 'scheduler_state_dict'

        self.__EPOCH_NUMBER = 'epoch_number'

    def load_model(self, model: nn.Module) -> None:
        content: list = torch.load(self.__file_name)
        if len(content) > 0:
            last_save = content[-1]
            model.load_state_dict(last_save[self.__MODEL_STATE_DICT])

    def load_optimizer(self, optimizer: Adam) -> None:
        content: list = torch.load(self.__file_name)
        if len(content) > 0:
            last_save = content[-1]
            optimizer.load_state_dict(last_save[self.__OPTIMIZER_STATE_DICT])

    def load_scheduler(self, scheduler: ExponentialLR) -> None:
        content: list = torch.load(self.__file_name)
        if len(content) > 0:
            last_save = content[-1]
            scheduler.load_state_dict(last_save[self.__SCHEDULER_STATE_DICT])

    def load_last_state_if_present(self, model: nn.Module, optimizer: Adam, scheduler: ExponentialLR) -> Union[int, None]:
        content: list = torch.load(self.__file_name)
        if len(content) > 0:
            last_save = content[-1]

            model.load_state_dict(last_save[self.__MODEL_STATE_DICT])
            optimizer.load_state_dict(last_save[self.__OPTIMIZER_STATE_DICT])
            scheduler.load_state_dict(last_save[self.__SCHEDULER_STATE_DICT])

            return last_save[self.__EPOCH_NUMBER]
        else:
            return None

    def save_state(self, model: nn.Module, optimizer: Adam, scheduler: ExponentialLR, epoch_number: int) -> None:
        content: list[dict[str, any]] = torch.load(self.__file_name)

        if any(saved[self.__EPOCH_NUMBER] == epoch_number for saved in content):
            raise ValueError(
                f'The epoch number {epoch_number} is aready in the save file.')

        content.append({
            self.__MODEL_STATE_DICT: model.state_dict(),
            self.__OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            self.__SCHEDULER_STATE_DICT: scheduler.state_dict(),
            self.__EPOCH_NUMBER: epoch_number
        })
        torch.save(content, self.__file_name)

    def load_model_at_epoch(self, epoch: int, model: nn.Module):
        content: list = torch.load(self.__file_name)
        if epoch < len(content):
            pass
        else:
            raise ValueError(
                f"The file {self.__file_name} has only {len(content)} but you ask for index {epoch}")
