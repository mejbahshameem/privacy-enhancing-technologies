import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score
from model import Net, NetBW
from attack import model_inversion, model_stealing, membership_inference
import preprocessDataset
import shadowModel

def train(net, epoch, valid_step, 
		  optimizer, scheduler, criterion, 
		  trainloader, testloader, device, mode):

	best_model = {}
	best_model['net'] = None
	best_model['valid_loss'] = 1e10
	best_model['precision'] = .0
	best_model['accuracy'] = .0
	best_model['recall'] = .0
	
	for e in range(epoch):
	
		training_loss = .0
		net.train()
	
		for i, data in enumerate(trainloader):

			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			scheduler.step()
			training_loss += loss.item()

			if i % valid_step == 0:
				acc, r, p, valid_loss = test(net, criterion, testloader, device)
				print('Epoch %d Iteration %d: train loss: %f, valid loss: %f, accuracy: %f, recall: %f, precision: %f' % (
				e+1, i, training_loss / (i+1), valid_loss, acc, r, p))
		
				if valid_loss < best_model['valid_loss']:
					best_model['net'] = copy.deepcopy(net)
					best_model['valid_loss'] = valid_loss
					best_model['precision'] = p
					best_model['accuracy'] = acc
					best_model['recall'] = r

		if mode == 'DEBUG':
			break

	return best_model

	  
def test(net, criterion, testloader, device):

	net.eval()
	valid_loss = .0
	pred_lbls = []
	true_lbls = []

	with torch.no_grad():
		for data in testloader:
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			valid_loss += loss.item()
			_, pred = torch.max(outputs, 1)
			pred_lbls.extend(pred.cpu().tolist())
			true_lbls.extend(labels.cpu().tolist())
	
	acc = accuracy_score(true_lbls, pred_lbls)
	r = recall_score(true_lbls, pred_lbls, average='macro') 
	p = precision_score(true_lbls, pred_lbls, average='macro')
	valid_loss /= len(testloader)

	return acc, r, p, valid_loss


def main():

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data', type=str, required=True)
	parser.add_argument('--attack', type=str, required=True)
	parser.add_argument('--mode', type=str, default='debug')
	parser.add_argument('--epoch', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--valid_step', type=int, default=200)
	parser.add_argument('--lr', type=float, default=.001)
	parser.add_argument('--loss_fn', type=str, default='cross_entropy')
		parser.add_argument('--opter', type=str, default='SGD')

	# Arguments for model inversion
	parser.add_argument('--alpha', type=int, default=5000)
	parser.add_argument('--beta',  type=int, default=1000)
	parser.add_argument('--gamma', type=float, default=.0001)
	parser.add_argument('--lambd', type=float, default=.1)

	# TODO: Arguments for model stealing (By David)


	# TODO: Arguments for membership inference attack (By Misbah)


	args = parser.parse_args()

	data = args.data
	batch_size = args.batch_size
	epoch = args.epoch
	lr = args.lr
	mode = args.mode
	opter = args.opter
	valid_step = args.valid_step
	attack = args.attack

	if mode == 'debug':
		epoch = 1
		valid_step = 500
	else:
		pass

	if attack != 'membership_inference' and attack != 'model_inversion' and attack != 'model_stealing':
		raise Exception('The attack', attack, 'is not implemented')

	if torch.cuda.is_available():
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')

	if data == 'mnist':

		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5), (0.5))])
		img_size = (1, 28, 28)

		trainset = torchvision.datasets.MNIST(
			'./data', train=True, download=True, transform=transform)
		
		testset = torchvision.datasets.MNIST(
			'./data', train=False, download=True, transform=transform)
		
		if (len(trainset) + len(testset)/4)% batch_size != 0:
			batch_size = int(batch_size - (len(trainset) + len(testset)/4)% batch_size)  
		
		trainloader = torch.utils.data.DataLoader(
			trainset, batch_size=batch_size, shuffle=True, num_workers=0)
		testloader = torch.utils.data.DataLoader(
			testset, batch_size=batch_size, shuffle=True, num_workers=0)
		classes = [ i for i in range(10) ]

		net = NetBW(len(classes)).to(device)
		shadow_net = NetBW(len(classes)).to(device)

	elif data == 'fashion':

		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5), (0.5))])
		img_size = (1, 28, 28)

		trainset = torchvision.datasets.FashionMNIST(
			'./data', train=True, download=True, transform=transform)
		testset = torchvision.datasets.FashionMNIST(
			'./data', train=False, download=True, transform=transform)
	
		if (len(trainset) + len(testset)/4)% batch_size != 0:
			batch_size = int(batch_size - (len(trainset) + len(testset)/4)% batch_size) 
			
		trainloader = torch.utils.data.DataLoader(
			trainset, batch_size=batch_size, shuffle=True, num_workers=0)
		testloader = torch.utils.data.DataLoader(
			testset, batch_size=batch_size, shuffle=True, num_workers=0)
		classes = (
			"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
			"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

		net = NetBW(len(classes)).to(device)
		shadow_net = NetBW(len(classes)).to(device)

	elif data == 'cifar10':

		transform = transforms.Compose(
			[transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		img_size = (3, 32, 32)

		trainset = torchvision.datasets.CIFAR10(
			'./data', train=True, download=True, transform=transform)
		testset = torchvision.datasets.CIFAR10(
			'./data', train=False, download=True, transform=transform)
		
		if (len(trainset) + len(testset)/4)% batch_size != 0:
			batch_size = int(batch_size - (len(trainset) + len(testset)/4)% batch_size) 
			
		trainloader = torch.utils.data.DataLoader(
			trainset, batch_size=batch_size, shuffle=True, num_workers=0)
		testloader = torch.utils.data.DataLoader(
			testset, batch_size=batch_size, shuffle=True, num_workers=0)
		classes = (
			'plane', 'car', 'bird', 'cat','deer', 
			'dog', 'frog', 'horse', 'ship', 'truck')

		net = Net(len(classes)).to(device)
		shadow_net = Net(len(classes)).to(device)

	else:
		raise Exception('No dataset ', data)

	if args.loss_fn == 'cross_entropy':
		criterion = nn.CrossEntropyLoss()
	elif args.loss_fn == 'mse':
		criterion = nn.MSELoss()

	if opter == 'SGD':
		opter = optim.SGD(net.parameters(), lr=lr, momentum=.9)
	elif opter == 'Adam':
		opter = optim.Adam(net.parameters(), lr=lr)
	else:
		raise Exception('The optimizer is not implemented', opter)

	scheduler = optim.lr_scheduler.StepLR(opter, 2000)
	
	if attack == 'membership_inference':
		Dtrainloader, Dnonmemberloader, Dtrainshadowloader, Doutshadowloader = preprocessDataset.process_image_dataset(trainset,testset,batch_size)
		print('Training Target Model...')
		best_state = train(
			net, epoch, valid_step, 
			opter, scheduler, criterion, 
			Dtrainloader, Dnonmemberloader, device, mode)
	
		print('Best result:')
		print('Valid Loss:', best_state['valid_loss'])
		print('Valid Accuracy:', best_state['accuracy'])
		print('Precision:', best_state['precision'])
		print('Recall:', best_state['recall'])
		print('Finished Training Target Model...')
		victim_model = best_state['net']
		victim_model.cpu()
		
		print('Training Shadow Model...')
		lr = 0.001
		momentum = 0.9
		epochs = 10
		shadow_model = shadowModel.shadow_model_membership_inference(shadow_net, Dtrainshadowloader, Doutshadowloader, lr, momentum, epochs)
		
		#Process Posterior data for building ATTACK model
		#3 Highest Posteriors are used to Genearte feature vectors for Multilayer Perceptron (ATTACK model)
		#Posteriors are sorted from HIGH to LOW values as mentioned in the paper
		#mlp_X contains the feature tensors for tarining the Attack model
		#mlp_Y contains the corresponding labels (1-member/0-nonmember)
		posterior_batch_size = 4
		shuffle = True #This is mandatory to set to True
		attack_data_loader = preprocessDataset.process_posterior_dataset(shadow_model, Dtrainshadowloader, Doutshadowloader, posterior_batch_size, batch_size, shuffle)
		
		#Now feed these posteriors in attack_data_loader for training the ATTACK model which is built on MLP as per the paper
		input_size = 3
		hidden_size = 64
		epochs = 25
		lr = 1e-05
		attack_model = membership_inference(attack_data_loader, input_size, hidden_size, lr, epochs, posterior_batch_size)
		#As attack_model is already trained on the whole shadow data(Dtrainshadow and Doutshadow)
		#The next step would be to get the posteriors for all the samples in target dataset(Dtarget-both members and non-member) using target model
		#target_X contains the posteriors for all target data that will be then checked for membership
		#target_Y contains the corresponding labels (1-member/0-nonmember)
		posterior_batch_size = 4
		shuffle = False  #This is mandatory to set to False for comparison purpose
		target_posteriors_loader = preprocessDataset.process_posterior_dataset(victim_model, Dtrainloader, Dnonmemberloader, posterior_batch_size, batch_size, shuffle)
		
		#Now target dataset is ready with their posteriors to be fed in attack_model
		#Finally, let's get the Membership Inference for All the target Data
	
		correct = 0
		total = 0
		TP = 0
		FP = 0
		TN = 0
		FN = 0
		
		with torch.no_grad():
			for i, data in enumerate (target_posteriors_loader,0):
				X, y = data
				outputs = attack_model(X).round()
				#As target data is not shuffled later on we know that
				#the first half target_posteriors_loader is member and the later half is non-member
				#that is why i < len(Dtrainloader) is used for checking the prediction of class 1 and vice versa
				#class/label 1 means member and vice versa
				
				if i < (len(Dtrainloader)*batch_size)/(posterior_batch_size):
					TP += (outputs == 1).sum().item()
					FN += (outputs == 0).sum().item()
					
				else:
					FP += (outputs == 1).sum().item()
					TN += (outputs == 0).sum().item()
				
			
		
		print('Accuracy of the attack on the all Target Data: %d %%' % (
			100 * (TP+TN) / (len(target_posteriors_loader) * posterior_batch_size)))
		
		print('Precision of the attack on the all Target Data: %d %%' % (
			100 * (TP / (TP+FP))))
		
		print('Recall of the attack on the all Target Data: %d %%' % (
			100 * (TP / (TP+FN))))

	else:
		
		best_state = train(
		net, epoch, valid_step, 
		opter, scheduler, criterion, 
		trainloader, testloader, device, mode)

		print('Best result:')
		print('Valid Loss:', best_state['valid_loss'])
		print('Valid Accuracy:', best_state['accuracy'])
		print('Precision:', best_state['precision'])
		print('Recall:', best_state['recall'])

		victim_model = best_state['net']    

		if attack == 'model_inversion':
			alpha = args.alpha
			beta = args.beta
			gamma = args.gamma
			lam = args.lambd
			
			if mode == 'debug':
				alpha = 10
				beta = 10
				gamma = .01
				lam = 100
	
			if data == 'cifar10':
				label_index = classes.index('cat')
			elif data == 'fashion':
				label_index = classes.index("Sneaker")
			elif data == 'mnist':
				label_index = classes.index(0)

			img, min_loss = model_inversion(
				victim_model, device, label_index, alpha, beta, gamma, lam, img_size)
			print('Min Loss:', min_loss)

		elif attack == 'model_stealing':
			stolen_net = model_stealing(victim_model, device, img_size, data)
			print('Performance summary of the stolen model:')
			print('Accuracy: {}, Recall: {}, Precision: {}, Validation Los: {}'.format(*test(stolen_net, criterion, testloader, device)))
			
		

if __name__ == '__main__':
	main()