#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-02-24 22:05:46
# @Author  : Tuo Feng (fengt@stu.xidian.edu.cn)
# @Link    : https://blog.csdn.net/taifengzikai/
# @Version : $Id$

import torch
import time

from distance import EMDLoss, ChamferLoss


'''
Test Wasserstein(EMD) Loss!
'''

dist01 = EMDLoss()

print('compute EMDLoss ALG1')
p1 = torch.rand(32,1024,3).cuda()#.double()
p2 = torch.rand(32,1024,3).cuda()#.double()

p1.requires_grad = True
p2.requires_grad = True

s = time.time()

cost  = dist01(p1, p2)
print('Wasserstein (EMD) cost from ALG1:')
print(cost)

loss1 = torch.sum(cost)
print('Wasserstein (EMD) loss from ALG1: %.05f'%loss1)
loss1.backward()

emd_time = time.time() - s
print('Time: ', emd_time)

'''
Test Chamfer Loss!
'''

print('compute ChamferLoss')
dist02 = ChamferLoss()

s2 = time.time()

cost1, cost2 = dist02(p1, p2)

print('chamfer cost1')
print(cost1)
print('chamfer cost2')
print(cost2) 

if True:
    reducer = torch.mean
else:
    reducer = torch.sum

loss = (reducer(cost1)) + (reducer(cost2))
print('chamfer loss: %.05f' %loss.data.cpu().numpy())

cd_time = time.time() - s2
print('Time: ', cd_time)
