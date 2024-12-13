from torch.optim import Adam, SGD

def get_optim(params, args):
    if args.optim == 'Adam':
        return Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == 'SGD':
        return SGD(params=params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)