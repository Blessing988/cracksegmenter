import csv




num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    with open(train_csv_path, mode='a', newline='') as train_file:
        train_writer = csv.writer(train_file)
        train_writer.writerow([epoch + 1, train_loss])

    val_loss, val_metrics = validate(model, val_loader, criterion, device)

    with open(val_csv_path, mode='a', newline='') as val_file:
        val_writer = csv.writer(val_file)
        val_writer.writerow([
            epoch + 1,
            val_loss,
            val_metrics['IoU'],
            val_metrics['Dice'],
            val_metrics['Precision'],
            val_metrics['Recall'],
            val_metrics['F1 Score']
        ])

    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    print(f'Val Metrics: {val_metrics}')

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best model saved.')
    else:
        epochs_no_improve += 1
        print(f'No improvement for {epochs_no_improve} epochs.')

    # Early stopping
    if epochs_no_improve >= early_stopping_patience:
        print('Early stopping!')
        break
