
import os
import argparse
import os.path
import torch
import pdb
#cifar10  dataset 3*32*32

parser = argparse.ArgumentParser()

parser.add_argument('--i', default='miniimage_84.pt', help='input directory')
parser.add_argument('--o', default='cifar10_au1_5.pt', help='output file')
parser.add_argument('--n_tasks', default=20, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

x_tr,  y_tr, x_te, y_te = torch.load(os.path.join(args.i))
# x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
# x_te = x_te.float().view(x_te.size(0), -1) / 255.0

cpt = int(100/ args.n_tasks)#每一类中的类别个数
u_y = y_tr.unique()
idx = torch.randperm(u_y.size(0))
u_y = u_y[idx]
mask = dict(zip(torch.arange(u_y.size(0)).int().tolist(), u_y.int().tolist()))
masked_ytr = [mask[y.item()] for y in y_tr]
masked_yte = [mask[y.item()] for y in y_te]
y_tr = torch.tensor(masked_ytr).long()
y_te = torch.tensor(masked_yte).long()

tasks_tr = []
tasks_te = []
for t in range(0,20):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)

    tt_tr = torch.full_like(y_tr[i_tr], t)
    tt_te = torch.full_like(y_te[i_te], t)

    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone(), tt_tr])#需要减3
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone(), tt_te])
    
torch.save([tasks_tr, tasks_te], 'miniimage_84.pt')