import torch
import os
import numpy as np

from tqdm import tqdm
from time import time
from thop import profile

from Model import MyModel
from Utils import get_optim
from Augmentation import add_edges, add_edge_weights, drop_edges, drop_edge_weights, drop_feature_random, add_edge_random, drop_feature_weights, drop_feature
from Linear_evaluation import linear_evaluation
from Load_data import get_dataset
from Arguments import get_args

def train(data, edge_weights, adding_edge, dropping_edge_weights, args):
    model = MyModel(data.num_features, args.feat_dim, args.proj_hidden_dim, args.temperature).to(args.device)
    optimizer = get_optim(model.parameters(), args)

    best_ac = 0.
    # flops, _ = profile(model, inputs=(data.x, data.edge_index))
    # print('Flops = {}'.format(flops))
    train_bar = tqdm(range(args.epoch), desc='Training')
    for epoch in train_bar:
        # t1 = time()
        model.train()
        optimizer.zero_grad()

        if args.drop_edge:
            e1 = drop_edges(data.edge_index, dropping_edge_weights, args.edge_drop_rate_1)
            e2 = drop_edges(data.edge_index, dropping_edge_weights, args.edge_drop_rate_2)
        else:
            e1 = data.edge_index
            e2 = data.edge_index

        if args.add_edge_random:
            edge_index_1 = add_edge_random(e1)
            edge_index_2 = add_edge_random(e2)
        elif args.add_edge:
            edge_index_1 = add_edges(edge_weights, adding_edge, e1)
            edge_index_2 = add_edges(edge_weights, adding_edge, e2)
        else:
            edge_index_1 = e1
            edge_index_2 = e2
        if args.add_single:
            edge_index_2 = e2

        x_1 = drop_feature_random(data.x, args.feat_drop_rate_1)
        x_2 = drop_feature_random(data.x, args.feat_drop_rate_2)

        _, z1 = model(x_1, edge_index_1)
        _, z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2, args.batch_compute)
        loss.backward()
        optimizer.step()
        # t2 = time()
        # print(t2-t1)
        train_bar.set_description("Train epoch {}, loss: {:.4f}".format(epoch + 1, loss.item()))

        if (epoch + 1) % args.loss_log == 0:
            path = args.output_path + "/loss.txt"
            f = open(path, "a")
            print("Train epoch {}, loss: {:.4f}".format(epoch + 1, loss.item()), file=f)
            f.close()
        if (epoch + 1) % args.eval == 0: # early stop
            path = args.output_path + "/eval.txt"
            model.eval()
            z = model(data.x, data.edge_index)[0].detach()
            y = data.y
            output, ac= linear_evaluation(z, y, args, epoch + 1)
            if ac > best_ac:
                best_ac = ac
                torch.save(model.state_dict(), args.output_path + '/best_ac_ckpt.pth')
            f = open(path, "a")
            print(output, file=f)
            f.close()
        if (epoch + 1) % args.save_model == 0:
            path = args.output_path + "/ckpt_epoch{}.pth".format(epoch + 1)
            torch.save(model.state_dict(), path)
    train_bar.close()
    print("best ac: {:.4f}".format(best_ac))
    return best_ac

def setup(data, args):
    if args.not_add_edge:
        adding_edge_weights, adding_edge = None, None
    else:
        adding_edge_weights, adding_edge = add_edge_weights(data.edge_index, args.edge_add_rate)
    if args.not_drop_edge:
        dropping_edge_weights = None
    else:
        dropping_edge_weights = drop_edge_weights(data.edge_index)
    return adding_edge_weights, adding_edge, dropping_edge_weights

if __name__ == '__main__':
    args = get_args()
    data = get_dataset(args.dataset_path, args.dataset).to(args.device)
    t0 = time()
    adding_edge_weights, adding_edge, dropping_edge_weights = setup(data, args)
    t1 = time()
    print(f"Data Processing Time: {t1-t0:.2f}s")
    output_path = args.output_path
    
    print(f"dataset:{args.dataset}  device:{args.device}  feature dim:{args.feat_dim}")
    acc = []
    for i in range(args.repeat):
        args.output_path = output_path + '/' + str(i+1)
        ac, z = train(data, adding_edge_weights, adding_edge, dropping_edge_weights, args)
        acc.append(ac)

    acc = np.array(acc)
    print("max:{:.2f}, min:{:.2f}, mean:{:.2f}, std:{:.2f}".format(acc.max(), acc.min(), acc.mean(), acc.std()))
    f = open(output_path+'/result.txt',"a")
    print(acc, file=f)
    print("mean:{:.2f}, std:{:.2f}".format(acc.mean(), acc.std()), file=f)
    dir = output_path.replace('in-progress', 'completed')
    os.rename(output_path, dir)
    print("result saved to" + dir)