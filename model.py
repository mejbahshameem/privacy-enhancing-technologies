import torch
import torch.nn as nn
import torch.nn.functional as F

# class NetBW(nn.Module):
# 	def __init__(self):
# 		super(NetBW, self).__init__()
# 		self.conv1 = nn.Conv2d(1, 6, 5)
# 		self.conv2 = nn.Conv2d(6, 16, 5)
# 		self.fc1 = nn.Linear(16*4*4, 120)
# 		self.fc2 = nn.Linear(120, 84)
# 		self.fc3 = nn.Linear(84, 10)

# 	def forward(self, x):
# 		# Max pooling over a (2, 2) window
# 		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
# 		# If the size is a square you can only specify a single number
# 		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
# 		x = x.view(-1, self.num_flat_features(x))
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return x

# 	def num_flat_features(self, x):
# 		size = x.size()[1:]  # all dimensions except the batch dimension
# 		num_features = 1
# 		for s in size:
# 			num_features *= s
# 		return num_features


# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.conv1 = nn.Conv2d(3, 6, 5)
# 		self.conv2 = nn.Conv2d(6, 16, 5)
# 		self.fc1 = nn.Linear(16*5*5, 120)
# 		self.fc2 = nn.Linear(120, 84)
# 		self.fc3 = nn.Linear(84, 10)

# 	def forward(self, x):
# 		# Max pooling over a (2, 2) window
# 		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
# 		# If the size is a square you can only specify a single number
# 		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
# 		x = x.view(-1, self.num_flat_features(x))
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return x


# 	def num_flat_features(self, x):
# 		size = x.size()[1:]  # all dimensions except the batch dimension
# 		num_features = 1
# 		for s in size:
# 			num_features *= s
# 		return num_features

#Attack network
class AttackNet_MembershipInference(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttackNet_MembershipInference, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

class ResBlock(nn.Module):

	def __init__(self, channels, filters):

		super(ResBlock, self).__init__()

		self.res = nn.Sequential(
			nn.BatchNorm2d(channels),
			nn.ReLU(),
			nn.Conv2d(channels, filters, kernel_size=(3,3), stride=1, padding=1),
			nn.BatchNorm2d(filters),
			nn.ReLU(),
			nn.Conv2d(filters, filters, kernel_size=(3,3), stride=1, padding=1),
			nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=1)
		)

		self.conv = nn.Conv2d(channels, filters, kernel_size=(3,3), stride=1, padding=1)

		self.res.apply(self._init_weights)
		nn.init.xavier_normal_(self.conv.weight)

	def forward(self, x):
		return self.res(x) + self.conv(x)

	def _init_weights(self, m):
		if type(m) == nn.Conv2d:
			nn.init.xavier_normal_(m.weight)


class Net(nn.Module):
  
	def __init__(self, num_classes):
		super(Net, self).__init__()
		self.resBlock1 = ResBlock(3, 8)
		self.dropout1 = nn.Dropout2d()
		self.resBlock2 = ResBlock(8, 16)
		self.dropout2 = nn.Dropout2d()
		self.resBlock3 = ResBlock(16, 32)
		self.dropout3 = nn.Dropout2d()
		self.resBlock4 = ResBlock(32, 64)
		self.dropout4 = nn.Dropout2d()
		self.classifier = nn.Sequential(
			nn.ReLU(),
			nn.Flatten(),
			# torch.flatten(),
			nn.Linear(64*32*32, 16*8*8),
			nn.ReLU(),
			nn.Linear(16*8*8, 10)
		)
		self.classifier.apply(self._init_weights)

	def forward(self, x):
		x = self.resBlock1(x)
		x = self.dropout1(x)
		x = self.resBlock2(x)
		x = self.dropout2(x)
		x = self.resBlock3(x)
		x = self.dropout3(x)
		x = self.resBlock4(x)
		x = self.dropout4(x)
		# x = torch.flatten(nn.ReLU(x))
		x = self.classifier(x)
		return x

	def _init_weights(self, m):
		if type(m) == nn.Linear:
			nn.init.xavier_normal_(m.weight)


class NetBW(nn.Module):
  
	def __init__(self, num_classes):
		super(NetBW, self).__init__()
		self.resBlock1 = ResBlock(1, 8)
		self.dropout1 = nn.Dropout2d()
		self.resBlock2 = ResBlock(8, 16)
		self.dropout2 = nn.Dropout2d()
		self.resBlock3 = ResBlock(16, 32)
		self.dropout3 = nn.Dropout2d()
		self.resBlock4 = ResBlock(32, 64)
		self.dropout4 = nn.Dropout2d()
		self.classifier = nn.Sequential(
			nn.ReLU(),
			nn.Flatten(),
			# torch.flatten(),
			nn.Linear(64*28*28, 16*7*7),
			nn.ReLU(),
			nn.Linear(16*7*7, 10)
		)
		self.classifier.apply(self._init_weights)

	def forward(self, x):
		x = self.resBlock1(x)
		x = self.dropout1(x)
		x = self.resBlock2(x)
		x = self.dropout2(x)
		x = self.resBlock3(x)
		x = self.dropout3(x)
		x = self.resBlock4(x)
		x = self.dropout4(x)
		# x = torch.flatten(nn.ReLU(x))
		x = self.classifier(x)
		return x

	def _init_weights(self, m):
		if type(m) == nn.Linear:
			nn.init.xavier_normal_(m.weight)

