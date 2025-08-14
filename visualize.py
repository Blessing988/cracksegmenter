import matplotlib.pyplot as plt
import torch

def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    samples = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            
            if num_classes == 1:
                preds = torch.sigmoid(outputs) > 0.5
            else:
                preds = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                if samples >= num_samples:
                    return
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                image = (image * np.array([0.229, 0.224, 0.225]) + 
                         np.array([0.485, 0.456, 0.406])) * 255
                image = image.astype(np.uint8)
                
                mask = masks[i].cpu().numpy()
                pred = preds[i].cpu().numpy()
                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title('Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap='gray')
                plt.title('Prediction')
                plt.axis('off')
                
                plt.show()
                
                samples += 1

# Example usage after training epochs
visualize_predictions(model, val_loader, device, num_samples=3)



# Visualize attention map for the self-supervised segmentation work
# Set model to evaluation mode
model.eval()
with torch.no_grad():
    data = test_data.to(device)  # Single batch or sample
    output, att_score, attention_map_f, attention_map_s, attention_map_l, context_f, context_s, context_l = model(data)
    
    # Move attention maps to CPU and convert to numpy for visualization
    attention_map_f = attention_map_f[0].cpu().numpy()  # First sample in batch
    attention_map_s = attention_map_s[0].cpu().numpy()
    attention_map_l = attention_map_l[0].cpu().numpy()
    
    # Visualize fine-scale attention map
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(attention_map_f, cmap='hot', interpolation='nearest')
    plt.title('Fine-Scale Attention Map')
    plt.colorbar()
    
    # Visualize small-scale attention map
    plt.subplot(1, 3, 2)
    plt.imshow(attention_map_s, cmap='hot', interpolation='nearest')
    plt.title('Small-Scale Attention Map')
    plt.colorbar()
    
    # Visualize large-scale attention map
    plt.subplot(1, 3, 3)
    plt.imshow(attention_map_l, cmap='hot', interpolation='nearest')
    plt.title('Large-Scale Attention Map')
    plt.colorbar()
    
    plt.show()
