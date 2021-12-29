#svhn  dataset 3*32*32
import torch
import os
#torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"]="0" #对GPU进行选择
tasks_tr = []
tasks_te = []

x_tr, y_tr = torch.load(os.path.join('svhn_train.pt'))
x_te, y_te = torch.load(os.path.join('svhn_test.pt'))

x_tr = torch.from_numpy(x_tr)
y_tr = torch.from_numpy(y_tr)
x_te = torch.from_numpy(x_te)
y_te = torch.from_numpy(y_te)

x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0

cpt = 2#每一类中的类别个数
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
for t in range(0,5):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)

    tt_tr = torch.full_like(y_tr[i_tr], t)
    tt_te = torch.full_like(y_te[i_te], t)

    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone(), tt_tr])#需要减3
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone(), tt_te])
torch.save([tasks_tr, tasks_te], 'svhn-5.pt')