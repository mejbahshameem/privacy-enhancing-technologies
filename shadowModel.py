import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def shadow_model_membership_inference(net_shadow, Dtrainshadowloader, Doutshadowloader, lr, momentum, num_epochs):
    net_shadow.cpu()
    criterion = nn.CrossEntropyLoss()
    optimizer_shadow = optim.SGD(net_shadow.parameters(), lr, momentum)
        
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(Dtrainshadowloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer_shadow.zero_grad()
    
            # forward + backward + optimize
            outputs = net_shadow(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_shadow.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[Epoch %d, Batch %3d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
    print('Finished Training Shadow Model')
    
    #Check accuracy in the training data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in Dtrainshadowloader:
            images, labels = data
            outputs = net_shadow(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the prediction of image classification in training shadow(Dtrainshadow) Data): %d %%' % (
        100 * correct / total))
    
    #Check accuracy in the samples which were not seen during training
    correct = 0
    total = 0
    with torch.no_grad():
        for data in Doutshadowloader:
            images, labels = data
            outputs = net_shadow(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the prediction of image classification in out shadow(Doutshadow) Data : %d %%' % (
        100 * correct / total))
    
    return net_shadow #Returns the trained shadow model