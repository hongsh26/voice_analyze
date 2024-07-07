import torch


def train(model, train_loader, val_loader, criterion, optimizer, epoch):
    n_epochs = epoch
    model.train()
    for epoch in range(n_epochs):
        val_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                total += y_val_batch.size(0)
                correct += (val_predicted == y_val_batch).sum().item()

        val_accuracy = correct / total
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), f"adam-batch16-epoch20.pth")
    print(f"Model saved. fakaAudio.pth")