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


def compute_z(modelx, train_loader, hookF, device):
	total = 0.0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			modelx(inputs)
			for this_z in hookF:
				total += torch.mean(this_z.output**2)
	return total.item()/len(hookF)

def compute_entropy_1st(modelx, train_loader, hookF, entropy_collector, device):
	for idx in range(len(hookF)):
		entropy_collector[idx]*=0.0
	total_norm_samples = torch.zeros(10, device = device)
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			print(batch_idx)
			inputs, targets = inputs.to(device), targets.to(device)
			modelx(inputs)
			for tar in range(10):
				for idx in range(len(hookF)-1):
					intermed = ((hookF[idx].output > 0.0).reshape(len(targets), -1) * ((targets == tar).unsqueeze(dim=1))).type(torch.int)
					entropy_collector[idx][:,tar] += torch.sum(intermed, dim=0).reshape(-1)
				total_norm_samples[tar] += torch.sum((targets == tar).type(torch.int))
	total = 0.0
	elements = 0
	for tar in range(10):
		for idx in range(len(hookF)-1):
			freq = 1.0 * entropy_collector[idx][:,tar] / total_norm_samples[tar]
			H = freq*torch.log2(freq + 1e-15) + (1.0-freq)*torch.log2(1.0-freq + 1e-15)
			if tar == 0:
				elements += H.numel()
			total -= torch.sum(H)
	return total.item()/elements

def compute_entropy_2nd(modelx, train_loader, hookF, entropy_collector, device):
	for idx in range(len(hookF)):
		entropy_collector[idx]*=0.0
	total_norm_samples = torch.zeros(10, device = device)
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			print(batch_idx)
			inputs, targets = inputs.to(device), targets.to(device)
			modelx(inputs)
			for tar in range(10):
				for idx in range(len(hookF)-1):
					intermed = (((hookF[idx].output > 0.0).reshape(len(targets), -1) * ((targets == tar).unsqueeze(dim=1))).type(torch.float))
					p_11 = torch.mm(torch.transpose(intermed,0,1), intermed)
					entropy_collector[idx][:,:,tar] += p_11
					del intermed
					del p_11
					intermed = (((hookF[idx].output <= 0.0).reshape(len(targets), -1) * ((targets == tar).unsqueeze(dim=1))).type(torch.float))
					p_00 = torch.mm(torch.transpose(intermed,0,1), intermed)
					entropy_collector[idx][:,:,tar] += p_00
					del p_00
					del intermed
				total_norm_samples[tar] += torch.sum((targets == tar).type(torch.int))
	total = 0.0
	elements = 0
	for tar in range(10):
		for idx in range(len(hookF)-1):
			freq = 1.0 * torch.sum(torch.tril(entropy_collector[idx][:,:,tar], diagonal=0)) / (total_norm_samples[tar] * entropy_collector[idx][:,:,tar].numel()/2)
			print(freq)
			H = (1.0-freq)*torch.log2(1.0-freq + 1e-15)
			if tar == 0:
				elements += H.numel()
			total -= torch.sum(H)
	return total.item()/elements

def compute_PSPentropy(model, dataset, device, order = 1):
	hookF = [Hook(layer,backward=False) for layer in list(model._modules.items())]
	for batch_idx, (inputs, targets) in enumerate(dataset):
		inputs, targets = inputs.to(device), targets.to(device)
		model(inputs)
		break;

	entropy_collector = [None] * len(hookF)
	if order == 1:
		for idx in range(len(hookF)):
			entropy_collector[idx] = torch.zeros(hookF[idx].output.shape[1:], device=device)
		H = compute_entropy_1st(modelx, train_loader, hookF, entropy_collector, device)
	elif order == 2:
		for idx in range(len(hookF))
			entropy_collector[idx] = torch.zeros((np.prod(hookF[idx].output.shape[1:]),np.prod(hookF[idx].output.shape[1:]), 10), device=device)
		H = compute_entropy_2nd(modelx, train_loader, hookF, entropy_collector, device)

	else:
		#not yet implemented
		error()
	del entropy_collector
	for h in hookF:
		h.close()
	return H
