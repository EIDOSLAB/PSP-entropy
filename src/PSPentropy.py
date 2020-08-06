import numpy as np
import torch

class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module[1].register_forward_hook(self.hook_fn)
			self.name = module[0]
		else:
			self.hook = module[1].register_backward_hook(self.hook_fn)
			self.name = module[0]
	def hook_fn(self, module, input, output):
		self.input = input
		self.output = output
	def close(self):
		self.hook.remove()

def compute_entropy_1st(modelx, train_loader, hookF, classes, device):
	H = [None] * len(hookF)
	H_classwise = [None] * len(hookF)
	P = [None] * len(hookF)
	N = [None] * len(hookF)
	M = torch.zeros(classes, device=device)
	for idx in range(len(hookF)):
		H_classwise[idx] = torch.zeros((np.prod(hookF[idx].output.shape[1:]), classes), device=device)
		P[idx] = torch.zeros((np.prod(hookF[idx].output.shape[1:]), classes), device=device)
		N[idx] = torch.zeros((np.prod(hookF[idx].output.shape[1:]), classes), device=device)
	with torch.no_grad():
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			for this_idx_1 in range(classes):
				M[this_idx_1] += torch.sum(yb == this_idx_1).item()
			modelx(xb)
			for idx in range(len(hookF)):
				for this_idx_1 in range(classes):
					this_yb_1 = ((yb == this_idx_1).type(torch.float)).unsqueeze(dim = 1)
					P[idx][:, this_idx_1] += torch.sum((hookF[idx].output.view(this_yb_1.shape[0], -1)>0) * this_yb_1, dim=0)
					N[idx][:, this_idx_1] += torch.sum((hookF[idx].output.view(this_yb_1.shape[0], -1)<=0) * this_yb_1, dim=0)
	for idx in range(len(hookF)):
		P[idx] = torch.clamp(P[idx] / M.unsqueeze(dim=0), 0.0001, 0.9999)
		N[idx] = torch.clamp(N[idx] / M.unsqueeze(dim=0), 0.0001, 0.9999)
		for this_idx_1 in range(classes):
			H_classwise[idx][:,this_idx_1] -= (P[idx][:,this_idx_1] * torch.log2(P[idx][:,this_idx_1] ) + (N[idx][:,this_idx_1]  * torch.log2(N[idx][:,this_idx_1])))
		H[idx] = torch.sum(H_classwise[idx], dim=1)
	return H, H_classwise

def compute_PSPentropy(model, dataset, device, order = 1, classes = 10):
	hookF = [Hook(layer,backward=False) for layer in list(model._modules.items())]
	for batch_idx, (inputs, targets) in enumerate(dataset):
		inputs, targets = inputs.to(device), targets.to(device)
		model(inputs)
		break;

	if order == 1:
		for idx in range(len(hookF)):
			H, H_classwise= compute_entropy_1st(model, dataset, hookF, classes, device)
		for h in hookF:
			h.close()
		return H, H_classwise
	else:
		#not yet implemented
		error()
