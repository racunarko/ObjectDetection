import torch, torchvision
import torchvision.transforms as transforms
import net
import torch.optim as optim


def main():
    net = net.Net()
    
    transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    batch_size = 4
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.paratmeters(), lr=0.001, momentum=0.9)
    
    # ovo je za prvu ruku nadena funkcija iz dokumentacije PyTorch-a, budem mijenjao ovaj dio
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # do ovdje ce se mijenjati

if __name__ == "__main__":
    main()