import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, ConcatDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score

from model import Net, NetBW, AttackNet_MembershipInference

def model_inversion(net, device, label_index, alpha, beta, gamma, lam, img_size):

	def cost_fuction(x):
		y = F.softmax(net(x))
		return 1 - y[0][label_index]

	x = torch.zeros(img_size).to(device)
	x.unsqueeze_(0)
	x.requires_grad = True
	net.eval()
	prev_losses = torch.empty((beta)).fill_(beta)

	min_x = None
	min_loss = 1e6
	
	for i in range(alpha):
	
		loss = cost_fuction(x)
		net.zero_grad()
		loss.backward()
		x.data = x - lam * x.grad.data
		x.grad.zero_()
		loss = loss.item()
		if min_loss > loss:
			min_loss = loss
			min_x = x.data

		if i % 500 == 0:
			print('Loss:', loss)
		if loss > torch.max(prev_losses):
			print('No more update')
			break
		if loss < gamma:
			print('Reach the min value; Loss is: ', loss)
			break

		# Assign prev loss
		pi = i % beta
		prev_losses[pi] = loss
  
	return min_x, min_loss

def model_stealing(net, device, img_size, data, verbose=False):
	assert data in ['mnist', 'fashion', 'cifar10'], \
		"Data Error: Only mnist, fashion and cifar10 are valida datasets"
	
	if data == 'mnist':
		stolen_model = NetBW(10).to(device)
		NUM_RANDOM_POINTS = 80_000
		BATCH_SIZE = 4
	
	elif data == 'fashion':
		stolen_model = NetBW(10).to(device)
		NUM_RANDOM_POINTS = 60_000
		BATCH_SIZE = 8

	elif data == 'cifar10':
		stolen_model = Net(10).to(device)
		NUM_RANDOM_POINTS = 60_000
		BATCH_SIZE = 8


	X, Y = [], []
	for _ in range(int(NUM_RANDOM_POINTS/BATCH_SIZE)):
		X.append(torch.rand((BATCH_SIZE,*img_size))*2-1)
		Y.append(net(X[-1].to(device)).data)

	criterion = nn.MSELoss()
	optimizer = optim.SGD(stolen_model.parameters(), lr=0.001, momentum=0.9)

	print('BEGIN: Stealing model')
	for epoch in range(20): 
		print((epoch+1)*5, '%')
		running_loss = 0.0
		for i, data in enumerate(zip(X,Y), 0):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = stolen_model(inputs)
			loss = criterion(outputs, labels.data)
			loss.backward()
			optimizer.step()

			# progress statistics
			running_loss += loss.item()
			if (i % 4000 == 3999) and verbose:    # print every 4000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('\nFinished stealing model!')
	return stolen_model

def membership_inference(attack_data_loader, input_size, hidden_size, lr, epochs, batch_size):
    
    attack_model = AttackNet_MembershipInference(input_size, hidden_size)
    
    
    criterion = nn.MSELoss()
    optimizer_attack = torch.optim.ASGD(attack_model.parameters(), lr)
  
    
    for epoch in range(epochs):
         running_loss = 0.0
      
         for i, data in enumerate(attack_data_loader, 0):
              
              inputs, labels = data
              
              optimizer_attack.zero_grad()
              # Forward pass
              outputs = attack_model(inputs)
              # Compute Loss
              loss = criterion(outputs, labels)
             
              
              # Backward pass
              loss.backward()
           
            
              optimizer_attack.step()
              
              # print statistics
              running_loss += loss.item()
              if i % 1000== 999:    # print every 1500 mini-batches
                  print('[Epoch %d, Batch %1d] loss: %.6f' %
                        (epoch + 1, i + 1, running_loss / 999))
                  running_loss = 0.0
                      
    print('Finished Training Attack Model')
  
    correct = 0
    total = 0
    with torch.no_grad():
        for data in attack_data_loader:
            X, y = data
            outputs = attack_model(X).round()
            total += batch_size
            correct += (outputs == y).sum().item()
        
    
    print('Accuracy of the attack on the All Shadow(Dshadow) Data: %d %%' % (
        100 * correct / total))
    return attack_model