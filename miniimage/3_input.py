import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from wideresnet import WideResNet

BATCH_SIZE = 50000
transform_train = transforms.Compose([
    transforms.Resize((84,84)),
    # transforms.RandomResizedCrop(84),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
 # 需要更多数据预处理，自己查
])
transform_test = transforms.Compose([
    transforms.Resize((84,84)),
    # transforms.Resize(256),
    # transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
 # 需要更多数据预处理，自己查
])

# transform_train = transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
#  # 需要更多数据预处理，自己查
# ])
# transform_test = transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
#  # 需要更多数据预处理，自己查
# ])


#读取数据
dataset_train = datasets.ImageFolder('./train', transform_train)
dataset_test = datasets.ImageFolder('./test', transform_test)
#dataset_val = datasets.ImageFolder('data/val', transform)

# 上面这一段是加载测试集的
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=50000, shuffle=True) # 训练集 50000
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1000, shuffle=False) # 测试集 10000
#val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True) # 验证集
# 对应文件夹的label
# print(dataset_train.class_to_idx)   # 这是一个字典，可以查看每个标签对应的文件夹，也就是你的类别。
#                                     # 训练好模型后输入一张图片测试，比如输出是99，就可以用字典查询找到你的类别名称
# print(dataset_test.class_to_idx)
for batch_idx, (inputs1,  targets) in enumerate(train_loader):
    x_tr, y_tr= inputs1,  targets
    # torch.save((x_tr, y_tr), 'mi.pt')

for batch_idx, (inputs1, targets) in enumerate(test_loader):
    x_te, y_te = inputs1, targets

torch.save((x_tr, y_tr, x_te,  y_te), 'miniimage_84.pt')