from torch.optim import Adam, SGD

def get_optim(params, optim, lr, weight_decay=0, momentum=0.9):
    if optim == 'Adam':
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    if optim == 'SGD':
        return SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)