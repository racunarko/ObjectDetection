import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Grayscale
from torch.optim.lr_scheduler import StepLR

from torchsummary import summary

import cv2
from dataset import MNISTLocalization
from network import Net

# ovo je ok
def train_step(model, device, train_loader, optimizer, epoch, image_size=128, scheduler=None, log_interval=100, verbose=True):
    model.train()
    train_losses = []
    train_counter = []
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['image']
        target_class = sample_batched['label']
        target_bbox = sample_batched['bbox']
        data, target_class, target_bbox = data.to(device), target_class.to(device), [x.float().to(device) for x in target_bbox]
        
        # normalize the bounding box coordinates
        x1 = target_bbox[0] / image_size
        y1 = target_bbox[1] / image_size
        x2 = target_bbox[2] / image_size
        y2 = target_bbox[3] / image_size
        
        optimizer.zero_grad()
        output_class, output_bbox = model(data)
        
        loss = F.nll_loss(output_class, target_class)
        loss += F.mse_loss(output_bbox[0], x1) + F.mse_loss(output_bbox[1], y1) + F.mse_loss(output_bbox[2], x2) + F.mse_loss(output_bbox[3], y2)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            if verbose:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        
        if scheduler:
            scheduler.step()

    return train_losses, train_counter

#ovo je ok
def train_network(model, device, train_loader, test_loader, optimizer, epoch, scheduler=None, log_interval=100):
    train_losses = []
    train_counter = []
    test_losses_clsf = []
    test_losses_bbox = []
    test_accuracies = []
    test_counter = [i*len(train_loader.dataset) for i in range(epoch + 1)]
    
    test_loss_clsf, test_accuracy, correct, test_loss_bbox = test(model, device, test_loader)
    test_losses_clsf.append(test_loss_clsf)
    test_losses_bbox.append(test_loss_bbox)
    test_accuracies.append(test_accuracy)
    
    for epoch in range(1, epoch + 1):
        new_train_losses, new_train_counter = train_step(model, device, train_loader, optimizer, epoch, scheduler=scheduler, log_interval=log_interval)
        train_losses.extend(new_train_losses)
        train_counter.extend(new_train_counter)
        
        test_loss_clsf, test_accuracy, correct, test_loss_bbox = test(model, device, test_loader)
        test_losses_clsf.append(test_loss_clsf)
        test_losses_bbox.append(test_loss_bbox)
        test_accuracies.append(test_accuracy)
        

# ovo je ok
def IoU(pred, target, iou_threshold = 0.7):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(pred[0], target[0])
	yA = max(pred[1], target[1])
	xB = min(pred[2], target[2])
	yB = min(pred[3], target[3])
	# compute the area of intersection rectangle
	intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	pred_area = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
	target_area = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(pred_area + target_area - intersection_area)
	# return the intersection over union value
	return iou > iou_threshold

# ovo je ok
def test(model, device, test_loader, image_size=128, verbose=True):
    model.eval()
    
    test_loss_clsf = 0
    test_loss_bbox = 0
    correct = 0
    
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            data = sample_batched['image']
            target_class = sample_batched['label']          
            target_bbox = sample_batched['bbox']
            data, target_class, target_bbox = data.to(device), target_class.to(device), [x.float().to(device) for x in target_bbox]
            
            output_class, output_bbox = model(data)
            
            test_loss_clsf += F.nll_loss(output_class, target_class, reduction='sum').item()
            
            test_loss_bbox += F.mse_loss(output_bbox[0], target_bbox[0] / image_size, reduction='sum').item()
            test_loss_bbox += F.mse_loss(output_bbox[1], target_bbox[1] / image_size, reduction='sum').item()
            test_loss_bbox += F.mse_loss(output_bbox[2], target_bbox[2] / image_size, reduction='sum').item()
            test_loss_bbox += F.mse_loss(output_bbox[3], target_bbox[3] / image_size, reduction='sum').item()

            pred = output_class.data.max(1, keepdim=True)[1]
            torch.tensor(output_bbox).detach().cpu().numpy()
            torch.tensor(target_bbox).detach().cpu().numpy()

            # count the number of prediction with an IoU above a certain threshold
            iou_list = IoU(torch.tensor(output_bbox).detach().cpu().numpy(), torch.tensor(target_bbox).detach().cpu().numpy(), iou_threshold = 0.7)
            correct += np.sum(pred.cpu().numpy().flatten() == target_class.cpu().numpy().flatten() & iou_list)
            
    test_loss_clsf /= len(test_loader.dataset)
    test_loss_bbox /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    test_loss /= (len(test_loader.dataset)/test_loader.batch_size)
    if verbose:
        print('\n[Test] Classification + Detection: Avg. loss: {:.4f}, Accuracy: {:5d}/{:5d} ({:2.2f}%) | Object detection: Avg. loss: {:.4f}\n'.format(
            test_loss_clsf,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            test_loss_bbox))

    return test_loss_clsf, test_accuracy, correct, test_loss_bbox

# ovo je ok
def show_bbox(image, label, bbox, pred_class=None, pred_bbox=None):
    """Show image with bbox"""
    image = image.squeeze()
    plt.imshow(image, cmap='gray')

    # plot the ground truth bounding box
    imgsize = image.shape
    bbox *= [imgsize[1],imgsize[0],imgsize[0],imgsize[1]]
    plt.plot([bbox[0],bbox[0]+bbox[3],bbox[0]+bbox[3],bbox[0],bbox[0]],
             [bbox[1],bbox[1],bbox[1]+bbox[2],bbox[1]+bbox[2],bbox[1]], c='g')
    plt.title("Label: {}".format(label))
    
    if pred_bbox is not None: 
        # plot the predicted bounding box (if provided)
        pred_bbox *= [imgsize[1],imgsize[0],imgsize[0],imgsize[1]]
        plt.plot([pred_bbox[0],pred_bbox[0]+pred_bbox[3],pred_bbox[0]+pred_bbox[3],pred_bbox[0],pred_bbox[0]],
                 [pred_bbox[1],pred_bbox[1],pred_bbox[1]+pred_bbox[2],pred_bbox[1]+pred_bbox[2],pred_bbox[1]], c='r')
        
    if pred_class is not None:
        plt.title("Predicted class: {}".format(pred_class))

# ovo je ok
def get_number_of_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    print(device)
    
    batch_size_train = 128
    batch_size_test = 128
    image_size = 128

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = MNISTLocalization(image_size=image_size, train=True, transform=transform)
    test_set = MNISTLocalization(image_size=image_size, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=False, num_workers=1, pin_memory=True)

    # check the sizes of the sets
    print("Size of training data: {}".format(len(train_set)))
    print("Size of test data: {}".format(len(test_set)))

    net = Net(image_size=image_size)
    # summary(net, (1, 128, 128), device="cpu")

    model = net.to(device)
    
    number_of_params = get_number_of_model_parameters(model)
    print("Number of the model parameters:", number_of_params)

    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 5
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))

    # do ovdje je ok

    train_losses, train_counter, test_losses_clsf, test_accuracies, test_losses_bbox,  test_counter = train_network(model, device, train_loader, test_loader, optimizer, epochs, scheduler=scheduler, log_interval=100)
    
    loss_history = train_losses
    
    test(model, device, test_loader, image_size=image_size, verbose=True)
    
    plt.plot(np.arange(len(loss_history)), loss_history)
    
    model.eval()

    fig = plt.figure()
    for i in range(5):
        # get a random index
        idx = np.random.randint(0,len(test_set))
        sample = test_set[idx]
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        
        # predict the bounding box for a given image
        data = sample['image']
        data = data.reshape((1,)+data.shape) # shape must be (b,c,h,w)
        data = data.to(device)
        with torch.no_grad():
            output_bbox, output_class = model(data)
            
        sample['pred_bbox'] = output_bbox[0].cpu().numpy()
        sample['pred_class'] = output_class[0].data.argmax().item()
        show_bbox(**sample)

        if i == 3:
            plt.show()
            break
        
    model.eval()

    fig = plt.figure()
    plt.tight_layout()
    ax.axis('off')
    ax = plt.subplot(1, 1, 1)
    for i in range(len(test_set)):
        sample = test_set[i]
        
        # predict the bounding box for a given image
        data = sample['image']
        data = data.reshape((1,)+data.shape) # shape must be (b,c,h,w)
        data = data.to(device)
        with torch.no_grad():
            output_bbox, _ = model(data)
        
        target = sample['bbox'].reshape((1,4))
        iou= IoU(output_bbox.cpu().numpy(), target, iou_threshold = 0.7)
        
        if not iou:
            print(output_bbox.cpu().numpy(), target,iou)
            sample['pred_bbox'] = output_bbox[0].cpu().numpy()
            sample['pred_class'] = output_class[0].data.argmax().item()
            show_bbox(**sample)
            break

    plt.show()
    
    torch.save(model.state_dict(), "mnist-model.pth")

    
if __name__ == '__main__':
    main()