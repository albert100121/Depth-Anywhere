from torch import optim

from utils import utils


def get_optim(args, net, state_dict=None):
    if args.optim == 'Adam':
        optimizer = optim.Adam(
            utils.group_weight(net),
            lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay,
        )
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(
            utils.group_weight(net),
            lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay,
        )
    elif args.optim == 'SGD':
        optimizer = optim.SGD(
            utils.group_weight(net),
            lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError()

    if state_dict:
        optimizer.load_state_dict(state_dict)

    return optimizer