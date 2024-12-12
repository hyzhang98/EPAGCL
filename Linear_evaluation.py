import torch

from tqdm import tqdm
from torch.utils.data import random_split

from Model import LinearClassifier
from Utils import get_optim

def linear_evaluation(z, y, args, epochs, split=None):
    classes_num = y.max().item() + 1
    y = y.view(-1)
    feat_dim = z.size(1)
    
    if split == None:
        train_mask, test_mask, val_mask = dataset_split(z.size(0))
    else:
        train_mask, test_mask, val_mask = split['train'], split['test'], split['valid']
        
    train_set, train_label = z[train_mask], y[train_mask]
    test_set, test_label = z[test_mask], y[test_mask]
    val_set, val_label = z[val_mask], y[val_mask]

    classifier = LinearClassifier(feat_dim, classes_num).to(args.device)
    optimizer = get_optim(classifier.parameters(), args.eval_optim, args.eval_lr, args.eval_weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0

    eval_bar = tqdm(range(args.eval_epoch), desc='Evaluating')
    for epoch in eval_bar:
        classifier.train()
        optimizer.zero_grad()
        
        output = classifier(train_set)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        classifier.eval()
        val_acc = ac(classifier, val_set, val_label)
        test_acc = ac(classifier, test_set, test_label)
        if val_acc > best_val_acc or (val_acc == best_val_acc and test_acc > best_test_acc):
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            path = "./best_eval.pth"
            torch.save(classifier.state_dict(), path)
        
        eval_bar.set_description("Eval epoch {}, best test acc: {:.4f}".format(epochs, best_test_acc))
        eval_bar.set_postfix(loss=loss.item())
    eval_bar.close()
    classifier.load_state_dict(torch.load("./best_eval.pth"))
    classifier.eval()
    return "best epoch: {}, best test acc: {:.4f}".format(best_epoch, best_test_acc), best_test_acc
    
def ac(model, dataset, label):
    output = model(dataset)
    _, pred = torch.max(output, 1)
    total = label.size(0)
    correct = (pred == label).sum().item()
    return 100 * correct / total

def dataset_split(num, train_ratio = 0.1, val_ratio = 0.1):
    train_len = int(num * train_ratio)
    val_len = int(num * val_ratio)
    test_len = num - train_len - val_len
    
    train, test, val = random_split(torch.arange(0, num), (train_len, test_len, val_len))
    train_idx, test_idx, val_idx = train.indices, test.indices, val.indices
    return get_mask(num, train_idx), get_mask(num, test_idx), get_mask(num, val_idx)

def get_mask(len, idx):
    mask = torch.zeros(len).to(torch.bool)
    mask[idx] = True
    return mask
