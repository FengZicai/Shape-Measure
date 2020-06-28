import torch
import torch.nn as nn

import shape_measure as metric

# Wasserstein distance function
class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, xyz1, xyz2):
        if not xyz1.is_cuda:
            cost, match = metric.emd_distance_forward(xyz1, xyz2)
        else:
            cost, match = metric.emd_distance_forward_cuda(xyz1, xyz2)
        self.save_for_backward(xyz1, xyz2, match)
        return cost
        
    @staticmethod
    def backward(self, grad_output):
        xyz1, xyz2, match = self.saved_tensors
        if not xyz1.is_cuda:
            grad_xyz1, grad_xyz2 = metric.emd_distance_backward(xyz1, xyz2, match)
        else:
            grad_xyz1, grad_xyz2 = metric.emd_distance_backward_cuda(xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2

# Chamfer distance function
class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            metric.cd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            metric.cd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            metric.cd_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            metric.cd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss,self).__init__()
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)

class EMDLoss(nn.Module):
	'''
	Computes the (approximate) Earth Mover's Distance between two point sets (from optas's github). 
	'''
	def __init__(self):
		super(EMDLoss, self).__init__()

	def forward(self, xyz1, xyz2):
		return EarthMoverDistanceFunction.apply(xyz1, xyz2)