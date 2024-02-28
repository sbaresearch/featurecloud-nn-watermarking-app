import torch
import torch.nn as nn

from torch.utils.data import DataLoader

def train(model, train_loader, optimizer, criterion, device, turn_off_bn):
    model.train()
    if turn_off_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for parameters in m.parameters():
                    parameters.requires_grad = False

    run_loss = 0.0
    run_correct = 0
    counter = 0

    for i, (images, labels) in enumerate(train_loader, 0):
        assert (i == counter)
        counter += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        run_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        run_correct += (predictions == labels).sum().item()

        loss.backward()
        optimizer.step()
    
    epoch_loss = run_loss / counter
    epoch_acc = 100. * (run_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc

def test(model, test_loader, criterion, device):
    model.eval()

    run_loss = 0.0
    run_correct = 0
    counter = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, 0):
            assert (i == counter)
            counter += 1
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            run_loss += loss.item()
            
            _, predictions = torch.max(outputs, 1)
            run_correct += (predictions == labels).sum().item()

    epoch_loss = run_loss / counter
    epoch_acc = 100. * (run_correct / len(test_loader.dataset))
    return epoch_loss, epoch_acc

def embed(model, trigger_set, wm_th, batch_size, optimizer_name, 
                            lr, momentum, max_epochs, turn_off_bn):
    
    train_loader = DataLoader(trigger_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(trigger_set, batch_size=batch_size, shuffle=False, num_workers=4)

    assert (optimizer_name in ['sgd', 'adam'])
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr,  momentum)
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    current_epoch = 0
    train_loss, train_acc = 0, 0
    wm_loss, wm_acc = 0, 0
    while current_epoch < max_epochs and wm_acc < wm_th:
        current_epoch += 1

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, turn_off_bn)
        wm_loss, wm_acc = test(model, test_loader, criterion, device)

        print(f'Epoch {current_epoch}')
        print(f'Training loss: {train_loss}; wm loss: {wm_loss}')
        print(f'Training accuracy: {train_acc}; wm accuracy: {wm_acc}')
    
    return model, current_epoch, wm_acc 