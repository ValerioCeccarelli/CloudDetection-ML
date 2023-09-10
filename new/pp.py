from new_model import CDFM3SF
import torch
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from prettytable import PrettyTable


model = CDFM3SF([4, 6, 3], gf_dim=4)  # 1.107.439

# input1 = torch.randn(1, 4, 384, 384)
# input2 = torch.randn(1, 6, 192, 192)
# input3 = torch.randn(1, 3, 64, 64)

# output = model(input1, input2, input3)

# print(output[0].shape)

# print the number of parameters
# print('Number of parameters: {}'.format(sum(p.numel()
#       for p in model.parameters())))

# resnet18 = models.resnet152(pretrained=False) # 60.192.808
# resnet = models.resnet18(pretrained=False)  # 11.689.512

# print('Number of parameters: {}'.format(sum(p.numel()
#                                             for p in resnet18.parameters() if p.requires_grad)))

# optimizer = Adam(model.parameters(), lr=0.00025, betas=(0.5, 0.9))
# optimizer.param_groups[0]['initial_lr'] = 0.00025
# scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=0)


# U-Net model has 28 million

# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         name = name.replace(".weight", "")
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params


# count_parameters(model)

last_epoch = 0
optimizer = Adam(model.parameters(), lr=0.00025, betas=(0.5, 0.9))
optimizer.param_groups[0]['initial_lr'] = 0.00025
scheduler = ExponentialLR(optimizer, gamma=0.90, last_epoch=last_epoch)
print("Scheduler loaded")
print(scheduler.get_last_lr())
scheduler.step()
print(scheduler.get_last_lr())
scheduler.step()
print(scheduler.get_last_lr())
scheduler.state_dict()
