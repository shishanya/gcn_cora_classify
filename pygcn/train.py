# 在 Python2 中导入未来的支持的语言特征中division (精确除法)，
# 即from __future__ import division ，当我们在程序中没有导入该特征时，
# "/“操作符执行的只能是整除，也就是取整数，只有当我们导入division(精确算法)以后，
# ”/"执行的才是精确算法。
from __future__ import division
# 在开头加上from __future__ import print_function这句之后，即使在python2.X，
# 使用print就得像python3.X那样加括号使用。python2.X中print不需要括号，而在python3.X中则需要。
from __future__ import print_function
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy
from models import GCN

'''
定义一个显示超参数的函数，将代码中所有的超参数打印
'''
def show_Hyperparameter(args):
    argsDict = args.__dict__
    print(argsDict)
    print('the settings are as following')
    for key in argsDict:
        print(key,':',argsDict[key])

'''
训练设置
'''
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode',action='store_true', default=False,
                    help='Validate during traing pass')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')
# 权重衰减
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability)')

# 如果程序不禁止使用GPU且当前主机的GPU可用，arg.cuda就为True
args = parser.parse_args()
show_Hyperparameter(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

#生成随机数种子
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

# 载入数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# print(labels)
# print(labels.max().item())
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
print("***")
for item in model.parameters():
    print(item)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# 写入cuda加速
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_train = idx_train.cuda()


def train(epoch):
    t = time.time()
    # 转为训练模式，梯度清零
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    # 先是通过model.eval()转为测试模式，之后计算输出，并单独对测试集计算损失函数和准确率。
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        model.eval()
        output = model(features, adj)

    # 验证集的损失函数
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}'.format(time.time() - t))


# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model  逐个epoch进行train，最后test
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test()

torch.cuda.empty_cache()




























