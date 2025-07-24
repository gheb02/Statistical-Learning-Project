import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import random
import shap
from collections import Counter
import torch.optim.lr_scheduler as lr_scheduler
from models import *
import tqdm
import os
import pandas as pd
from glob import glob

def create_image_df(base_path):
    data = []
    labels = os.listdir(base_path)
    for label in labels:
        class_dir = os.path.join(base_path, label)
        if os.path.isdir(class_dir):
            for img_path in glob(os.path.join(class_dir, "*")):
                data.append((img_path, label))
    return pd.DataFrame(data, columns=["img_path", "label"])

def get_targets_from_subset(subset):
    return [subset.dataset.targets[i] for i in subset.indices]


# Helper function to load a specified number of images and labels from a DataLoader
def get_images_from_loader(loader, num_images):
    images = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            images.append(x)
            labels.append(y)
            if len(torch.cat(images)) >= num_images:
                break
    images = torch.cat(images)[:num_images]
    labels = torch.cat(labels)[:num_images]
    return images.to(device), labels.to(device)


# Linear beta schedule used for noise scaling over time
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# Forward diffusion process in latent space
def q_sample(z_0, t, noise, alphas_cumprod):
    """
    Adds noise to latent vector z_0 at timestep t.
    """
    sqrt_alpha_cumprod = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod[t]) ** 0.5
    return sqrt_alpha_cumprod.unsqueeze(1) * z_0 + sqrt_one_minus_alpha_cumprod.unsqueeze(1) * noise

# Training loop for probabilistic denoising model
def train_denoiser(autoencoder, dataloader, latent_dim, device, epochs=10):
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False  # Freeze autoencoder during denoiser training

    denoiser = LatentDenoiserResidual(latent_dim).to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-4)

    print("Starting Probabilistic Denoiser Training...")

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            with torch.no_grad():
                z_0 = autoencoder.encoder(images)  # Extract latent representation

            t = torch.randint(0, T, (images.size(0),), device=device)  # Random timesteps
            noise = torch.randn_like(z_0)
            z_t = q_sample(z_0, t, noise, alphas_cumprod.to(device))  # Noisy latent

            # Predict mean and std of the noise
            mean_noise_pred, std_noise_pred = denoiser(z_t, t)

            # Compute NLL loss assuming Gaussian noise
            std_noise_pred_clipped = std_noise_pred.clamp(min=1e-6)  # Numerical stability

            loss = 0.5 * torch.log(std_noise_pred_clipped**2) + \
                   0.5 * ((noise - mean_noise_pred) / std_noise_pred_clipped)**2

            loss = torch.mean(loss)  # Mean over batch and latent dimensions

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Probabilistic Denoiser Loss = {avg_loss:.6f}")

    return denoiser


def train_ddm_classifier(autoencoder, train_loader, val_loader, latent_dim, num_classes, device, T, alphas_cumprod, q_sample, epochs=20): 
    # Freeze autoencoder parameters
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    # Initialize the noise-aware classifier
    classifier = LatentClassifierResidual(latent_dim, num_classes, hidden_dim=128, dropout=0.2).to(device)
    
    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

    # Binary classification loss
    classification_loss_fn = torch.nn.BCEWithLogitsLoss()
    alphas_cumprod = alphas_cumprod.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 10

    print(f"Starting DDM Classifier Training for {epochs} epochs (Only Classification Loss)...")
    print(f"Classifier architecture: {classifier}")
    print(f"Optimizer: {optimizer}")
    print(f"Loss Function: {classification_loss_fn}")
    print("-" * 80)

    for epoch in range(epochs):
        classifier.train()
        train_class_loss = 0
        train_all_preds_logits = []
        train_all_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Encode image into latent space using frozen encoder
            z_0 = autoencoder.encoder(images)

            # Sample a random timestep and noise
            t = torch.randint(0, T, (images.size(0),), device=device)
            noise = torch.randn_like(z_0)

            # Apply forward diffusion (add noise)
            z_t = q_sample(z_0, t, noise, alphas_cumprod)

            # Get classification and noise prediction outputs
            classification_logits, mu_pred, std_pred = classifier(z_t, t)

            # Compute binary classification loss
            class_loss = classification_loss_fn(classification_logits, labels.float())

            # Compute Gaussian NLL for noise prediction
            nll = 0.5 * (((noise - mu_pred) / std_pred) ** 2 + 
                         2 * torch.log(std_pred) + 
                         torch.log(torch.tensor(2 * torch.pi, device=device)))
            nll_loss = nll.mean()

            lambda_nll = 1.0  # Weight for noise loss
            loss = class_loss + lambda_nll * nll_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            train_class_loss += loss.item() * images.size(0)
            train_all_preds_logits.append(classification_logits.cpu())
            train_all_labels.append(labels.cpu())

        # Concatenate predictions and labels for training metrics
        train_all_preds_logits = torch.cat(train_all_preds_logits)
        train_all_labels = torch.cat(train_all_labels)

        train_probs = torch.sigmoid(train_all_preds_logits)
        train_preds = (train_probs > 0.5).long() # Convert probabilities to binary predictions with a neutral threshold of 0.5

        avg_train_class_loss = train_class_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_all_labels, train_preds)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            train_all_labels, train_preds, average='binary', zero_division=0
        )

        # --- Validation loop ---
        classifier.eval()
        val_class_loss = 0
        val_all_preds_logits = []
        val_all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                z_0 = autoencoder.encoder(images)

                t = torch.randint(0, T, (images.size(0),), device=device)
                noise = torch.randn_like(z_0)
                z_t = q_sample(z_0, t, noise, alphas_cumprod)

                classification_logits, mu_pred, std_pred = classifier(z_t, t)
                class_loss = classification_loss_fn(classification_logits, labels.float())

                # NLL for validation
                nll = 0.5 * (((noise - mu_pred) / std_pred) ** 2 + 
                             2 * torch.log(std_pred) + 
                             torch.log(torch.tensor(2 * torch.pi, device=device)))
                nll_loss = nll.mean()

                loss = class_loss + lambda_nll * nll_loss

                val_class_loss += loss.item() * images.size(0)
                val_all_preds_logits.append(classification_logits.cpu())
                val_all_labels.append(labels.cpu())

        # Compute validation metrics
        val_all_preds_logits = torch.cat(val_all_preds_logits)
        val_all_labels = torch.cat(val_all_labels)

        val_probs = torch.sigmoid(val_all_preds_logits)
        val_preds = (val_probs > 0.5).long() # Convert probabilities to binary predictions with a neutral threshold of 0.5

        avg_val_class_loss = val_class_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_all_labels, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            val_all_labels, val_preds, average='binary', zero_division=0
        )

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train: Loss={avg_train_class_loss:.4f} Acc={train_acc:.4f} Prec={train_prec:.4f} Rec={train_rec:.4f} F1={train_f1:.4f}")
        print(f"Val:   Loss={avg_val_class_loss:.4f} Acc={val_acc:.4f} Prec={val_prec:.4f} Rec={val_rec:.4f} F1={val_f1:.4f}")
        print("-" * 80)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_class_loss)

        # Save best model and early stopping
        if avg_val_class_loss < best_val_loss:
            best_val_loss = avg_val_class_loss
            patience_counter = 0
            # torch.save(classifier.state_dict(), "best_ddm_classifier.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    return classifier


def evaluate_classifier(classifier, autoencoder, test_loader, device, T, alphas_cumprod, q_sample, lambda_nll=1.0):
    # Set both models to evaluation mode
    classifier.eval()
    autoencoder.eval()

    # Binary classification loss
    classification_loss_fn = torch.nn.BCEWithLogitsLoss()
    alphas_cumprod = alphas_cumprod.to(device)

    total_loss = 0
    all_logits = []
    all_labels = []

    print("\n📊 EVALUATING TEST SET PERFORMANCE")
    print("-" * 80)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Encode images into latent space
            z_0 = autoencoder.encoder(images)

            # Sample random timesteps and noise
            t = torch.randint(0, T, (images.size(0),), device=device)
            noise = torch.randn_like(z_0)

            # Generate noisy latent representations
            z_t = q_sample(z_0, t, noise, alphas_cumprod)

            # Classify and predict noise parameters
            logits, mu_pred, std_pred = classifier(z_t, t)

            # Compute classification loss
            class_loss = classification_loss_fn(logits.view(-1), labels.float())

            # Compute Gaussian Negative Log-Likelihood loss
            nll = 0.5 * (
                ((noise - mu_pred) / std_pred) ** 2
                + 2 * torch.log(std_pred)
                + torch.log(torch.tensor(2 * torch.pi, device=device))
            )
            nll_loss = nll.mean()

            # Total loss is a weighted sum of classification and noise prediction loss
            loss = class_loss + lambda_nll * nll_loss

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Concatenate predictions and labels across all batches
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(all_logits)
    preds = (probs > 0.5).long() # Convert probabilities to binary predictions with a neutral threshold of 0.5

    # Compute evaluation metrics
    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(all_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='weighted', zero_division=0
    )

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")
    print("-" * 80)

    return acc, prec, rec, f1

@torch.no_grad()
def infer_classifier(autoencoder, classifier, dataloader, alphas_cumprod, device, timestep, threshold=0.47, t_stable=False):
    """
    Runs inference with the classifier on a given dataloader, at a specific diffusion timestep.
    
    If t_stable=True, the classifier always receives t=0, regardless of the actual timestep used for sampling.
    This is useful to test the classifier`s ability to detect noise without being explicitly told the timestep.
    """
    classifier.eval()
    autoencoder.eval()
    
    all_preds = []
    all_labels = []

    if t_stable:
        # Use a fixed timestep t=0 for all predictions (classifier input only)
        t_0 = torch.tensor([0], device=device, dtype=torch.long)

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            z_0 = autoencoder.encoder(images)  # Encode images to latent space

            # Apply noise corresponding to the actual desired timestep
            t = torch.full((images.size(0),), timestep, device=device, dtype=torch.long)
            noise = torch.randn_like(z_0)
            z_t = q_sample(z_0, t, noise, alphas_cumprod)

            # Predict using fixed t=0, simulating noise-unaware classification
            logits, mu_pred, std_pred = classifier(z_t, t_0)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    else:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            z_0 = autoencoder.encoder(images)

            # Apply noise corresponding to the actual desired timestep
            t = torch.full((images.size(0),), timestep, device=device, dtype=torch.long)
            noise = torch.randn_like(z_0)
            z_t = q_sample(z_0, t, noise, alphas_cumprod)

            # Classifier is explicitly told the true timestep t
            logits, mu_pred, std_pred = classifier(z_t, t)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate predictions and labels from all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return all_preds, all_labels


def evaluate_predictions(preds, labels):
    """
    Computes accuracy, precision, recall, and F1 score for the predictions.
    """
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


def test_across_timesteps(autoencoder, classifier, dataloader, alphas_cumprod, device, timesteps, t_stable=False):
    """
    Runs classifier evaluation across multiple timesteps.
    
    If t_stable=True, the classifier always sees t=0.
    If False, the classifier is given the correct t.
    """
    print(f"{'Timestep':>8} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | {'F1':>6}")
    print("-" * 50)

    for t in timesteps:
        preds, labels = infer_classifier(
            autoencoder, classifier, dataloader, alphas_cumprod, device, 
            timestep=t, t_stable=t_stable
        )
        accuracy, precision, recall, f1 = evaluate_predictions(preds, labels)
        print(f"{t:8d} | {accuracy:8.4f} | {precision:9.4f} | {recall:7.4f} | {f1:6.4f}")


def evaluate_model_under_noise(ddm_classifier, tumor_classifier, autoencoder, dataloader, device, latent_dim, alphas_cumprod, q_sample):
    # Set all models to evaluation mode
    ddm_classifier.eval()
    tumor_classifier.eval()
    autoencoder.eval()

    # Define the timesteps at which to evaluate robustness (from 0 to 900, step 100)
    timesteps_to_test = list(range(0, 1000, 100))
    results = []

    for timestep in timesteps_to_test:
        ddm_preds_raw = []
        tumor_preds_raw = []
        all_labels = []

        # Compute the standard deviation of noise at the current timestep
        alpha_bar = alphas_cumprod[timestep].item()
        noise_std = (1 - alpha_bar) ** 0.5

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            z_0 = autoencoder.encoder(images)

            # Prepare timestep tensors for DDM classifier
            t_tensor = torch.full((images.size(0),), timestep, device=device, dtype=torch.long)
            t_0 = torch.tensor([0], device=device)  # Used to simulate t=0 as classifier input

            # Add Gaussian noise in the latent space
            noise = torch.randn_like(z_0)
            z_t = q_sample(z_0, t_tensor, noise, alphas_cumprod)

            # Forward pass through DDM classifier (with fixed t=0 at inference)
            logits, _, _ = ddm_classifier(z_t, t_0)
            ddm_prob = logits.sigmoid()

            # --- TumorClassifier path ---
            if timestep == 0:
                # No noise added at timestep 0
                images_noisy = images
            else:
                # Add Gaussian noise in image space
                noise_img = torch.randn_like(images) * noise_std
                images_noisy = torch.clamp(images + noise_img, 0.0, 1.0)  # Clamp to valid pixel range

            # Forward pass through CNN-based TumorClassifier
            logits_cnn = tumor_classifier(images_noisy)
            probs_cnn = torch.sigmoid(logits_cnn).squeeze(1)  # Output shape: [B]

            # Threshold predictions
            ddm_binary_pred = (ddm_prob > 0.47).long()
            cnn_binary_pred = (probs_cnn > 0.33).long()

            # Collect predictions and labels
            ddm_preds_raw.append(ddm_binary_pred.cpu())
            tumor_preds_raw.append(cnn_binary_pred.cpu())
            all_labels.append(labels.cpu())

        # Concatenate predictions and labels across all batches
        all_labels = torch.cat(all_labels)
        ddm_preds = torch.cat(ddm_preds_raw)
        tumor_preds = torch.cat(tumor_preds_raw)

        # Helper function to compute accuracy, precision, recall, and F1 score
        def compute_metrics(y_true, y_pred):
            y_true_np = y_true.numpy()
            y_pred_np = y_pred.numpy()
            acc = accuracy_score(y_true_np, y_pred_np)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true_np, y_pred_np, average='weighted', zero_division=0)
            return acc, prec, rec, f1

        # Compute metrics for both models
        ddm_metrics = compute_metrics(all_labels, ddm_preds)
        cnn_metrics = compute_metrics(all_labels, tumor_preds)

        # Store results for current timestep
        results.append({
            'timestep': timestep,
            'ddm': ddm_metrics,
            'cnn': cnn_metrics
        })

    return results


def print_comparison(results):
    print(f"{'Timestep':>8} | {'Model':>8} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6}")
    print("-" * 56)
    for res in results:
        t = res['timestep']
        for model_name, metrics in [('DDM', res['ddm']), ('CNN', res['cnn'])]:
            acc, prec, rec, f1 = [f"{m:.4f}" for m in metrics]
            print(f"{t:>8} | {model_name:>8} | {acc:>6} | {prec:>6} | {rec:>6} | {f1:>6}")


def plot_comparison(results):
    timesteps = [r['timestep'] for r in results]

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_idx = {'accuracy': 0, 'precision': 1, 'recall': 2, 'f1': 3}

    ddm_metrics = {metric: [] for metric in metrics}
    cnn_metrics = {metric: [] for metric in metrics}

    for res in results:
        for metric in metrics:
            ddm_metrics[metric].append(res['ddm'][metric_idx[metric]])
            cnn_metrics[metric].append(res['cnn'][metric_idx[metric]])

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(timesteps, ddm_metrics[metric], label='DDM Classifier', marker='o')
        plt.plot(timesteps, cnn_metrics[metric], label='CNN Classifier', marker='s')
        plt.title(f'{metric.capitalize()} over Timesteps')
        plt.xlabel('Timestep')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def train_ddm_resnet_classifier(
    ddm_classifier_model,
    optimizer_ddm,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    class_weights: torch.Tensor = None  
):
    ddm_classifier_model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    patience = 5
    min_delta = 0.0001

    # Calculate pos_weight for BCEWithLogitsLoss if class weights provided
    pos_weight = None
    if class_weights is not None and class_weights.numel() == 2:
        # Count positive and negative samples in training set to balance loss
        n_pos = sum(label == 1 for batch in train_loader for label in batch[1])
        n_neg = sum(label == 0 for batch in train_loader for label in batch[1])
        pos_weight_val = n_neg / (n_pos + 1e-8)
        pos_weight = torch.tensor([pos_weight_val], device=device)
        print(f"Using pos_weight: {pos_weight.item():.2f}")

    # Define binary cross-entropy loss with optional positive class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Starting training without uncertainty modeling...")

    for epoch in range(epochs):
        ddm_classifier_model.train()
        train_loss = 0.0
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer_ddm.zero_grad()

            logits = ddm_classifier_model(images)
            loss = criterion(logits, labels)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(ddm_classifier_model.parameters(), max_norm=1.0)
            optimizer_ddm.step()

            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader.dataset)

        # Validation phase
        ddm_classifier_model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            val_pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                logits = ddm_classifier_model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze(1)

                all_val_labels.append(labels.cpu())
                all_val_preds.append(preds.cpu())
                val_pbar.set_postfix({'loss': loss.item()})

        val_loss /= len(val_loader.dataset)
        all_val_labels_np = torch.cat(all_val_labels).numpy().squeeze()
        all_val_preds_np = torch.cat(all_val_preds).numpy()

        val_acc = accuracy_score(all_val_labels_np, all_val_preds_np)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            all_val_labels_np, all_val_preds_np, average='binary', zero_division=0
        )

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")

        # Save best model based on validation loss with early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = ddm_classifier_model.state_dict()
            epochs_no_improve = 0
            print(f"--> Saved best model with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Epochs with no improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        ddm_classifier_model.load_state_dict(best_model_state)

    print("\nTraining complete.")
    return ddm_classifier_model




def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        # Unsqueeze labels to match the model output shape
        labels = labels.to(device).float().unsqueeze(1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        # Squeeze back to initial size for comparison
        predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze(1) 

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels.squeeze(1)).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels.squeeze(1)).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            labels_loss = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels_loss)

            running_loss += loss.item() * inputs.size(0)

            predicted_probs = torch.sigmoid(outputs)
            predicted_classes = (predicted_probs > 0.5).long().squeeze(1)

            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted_classes.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = accuracy_score(all_labels, all_predicted)
    test_precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)


    return test_loss, test_accuracy, test_precision, test_recall, test_f1



def conformal_prediction(model, alpha, calibration_loader, test_loader, device, T=None, autoencoder=None):
  model.to(device)
  model.eval()

  nonconformity_scores = []

  if T and autoencoder is not None:
        with torch.no_grad():
            for images, labels in calibration_loader:
                images, labels = images.to(device), labels.to(device)
                z_0 = autoencoder.encoder(images)

                # Usa un nome diverso per il tensore timestep
                t_tensor = torch.randint(0, T, (images.size(0),), device=device)


                noise = torch.randn_like(z_0)
                z_t = q_sample(z_0, t_tensor, noise, alphas_cumprod)

                logits, _, _ = model(z_t, t_tensor)
                ddm_prob = logits.sigmoid()

                # Calculate nonconformity scores: 1 - P(true_class)
                # If true_label is 0, score = P(Class=1)
                # If true_label is 1, score = 1 - P(Class=1)

                # Combine using torch.where for efficiency
                # scores = torch.where(condition, value_if_true, value_if_false)
                scores_batch = torch.where(
                    labels == 0,         # condition: if true label is 0
                    ddm_prob,        # score is P(Class=1)
                    1 - ddm_prob     # else (true label is 1), score is 1 - P(Class=1)
                )
                nonconformity_scores.extend(scores_batch.cpu().numpy())

  else:
    with torch.no_grad():
        for images, labels in calibration_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1) # Output is [batch_size, 1], squeeze to [batch_size]
            probs_class1 = torch.sigmoid(logits) # P(Class=1)

            # Calculate nonconformity scores: 1 - P(true_class)
            # If true_label is 0, score = P(Class=1)
            # If true_label is 1, score = 1 - P(Class=1)

            # Combine using torch.where for efficiency
            # scores = torch.where(condition, value_if_true, value_if_false)
            scores_batch = torch.where(
                labels == 0,         # condition: if true label is 0
                probs_class1,        # score is P(Class=1)
                1 - probs_class1     # else (true label is 1), score is 1 - P(Class=1)
            )
            nonconformity_scores.extend(scores_batch.cpu().numpy())

  # Convert to numpy array and sort
  nonconformity_scores = np.array(nonconformity_scores)
  nonconformity_scores.sort()

  q_hat_index = int(np.ceil((len(nonconformity_scores) + 1) * (1 - alpha))) - 1
  q_hat = nonconformity_scores[q_hat_index]

  print(f"\nSignificance Level (alpha): {alpha}")
  print(f"Desired Coverage: {1 - alpha}")
  print(f"Number of calibration scores: {len(nonconformity_scores)}")
  print(f"Quantile index: {q_hat_index}")
  print(f"Calculated q_hat (threshold): {q_hat:.4f}")



  # Prediction Phase: Generate Prediction Sets for New Test Data
  correct_coverage_count = 0
  total_samples = 0
  total_set_size = 0

  all_set_sizes = []
  max_probs = []


  with torch.no_grad():
    if T and autoencoder is not None:
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            z_0 = autoencoder.encoder(images)

            # Usa un nome diverso per il tensore timestep
            t_tensor = torch.randint(0, T, (images.size(0),), device=device)

            noise = torch.randn_like(z_0)
            z_t = q_sample(z_0, t_tensor, noise, alphas_cumprod)

            logits, _, _ = model(z_t, t_tensor)
            ddm_prob = logits.sigmoid()

            for j in range(images.size(0)):
                true_label = labels[j].item()
                prob_class1 = ddm_prob[j].item()

                score_if_0 = prob_class1
                score_if_1 = 1 - prob_class1

                prediction_set = []
                if score_if_0 <= q_hat:
                    prediction_set.append(0)
                if score_if_1 <= q_hat:
                    prediction_set.append(1)

                # Check for coverage
                is_covered = (true_label in prediction_set)
                if is_covered:
                    correct_coverage_count += 1

                total_samples += 1
                total_set_size += len(prediction_set)

                all_set_sizes.append(len(prediction_set))

                max_probs.append(max(prob_class1, 1 - prob_class1))
    else:
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1)
            probs_class1 = torch.sigmoid(logits) # P(Class=1)

            for j in range(images.size(0)):
                true_label = labels[j].item()
                prob_class1 = probs_class1[j].item()


                score_if_0 = prob_class1
                score_if_1 = 1 - prob_class1

                prediction_set = []
                if score_if_0 <= q_hat:
                    prediction_set.append(0)
                if score_if_1 <= q_hat:
                    prediction_set.append(1)

                # Check for coverage
                is_covered = (true_label in prediction_set)
                if is_covered:
                    correct_coverage_count += 1

                total_samples += 1
                total_set_size += len(prediction_set)

                all_set_sizes.append(len(prediction_set))

                max_probs.append(max(prob_class1, 1 - prob_class1))



  empirical_coverage = correct_coverage_count / total_samples
  average_set_size = total_set_size / total_samples

  print(f"\nConformal Prediction Results:")
  print(f"Total samples in final test set: {total_samples}")
  print(f"Empirical Coverage: {empirical_coverage:.4f} (Expected: >= {1 - alpha})")
  print(f"Average Prediction Set Size: {average_set_size:.4f}")

  # Example of what average set size means:
  if average_set_size == 1.0:
      print("\nMost predictions are single-class sets (high confidence).")
  elif average_set_size > 1.0 and average_set_size < 2.0:
      print("\nMany predictions are single-class, but some are two-class sets (uncertain).")
  elif average_set_size == 2.0:
      print("\nAll predictions are two-class sets (model is highly uncertain or q_hat is too high).")


  plt.figure(figsize=(8, 6))
  plt.hist(all_set_sizes, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, align='mid', edgecolor='black')
  plt.xticks([0, 1, 2])
  plt.title('Distribution of Prediction Set Sizes')
  plt.xlabel('Prediction Set Size')
  plt.ylabel('Frequency')
  plt.grid(axis='y', alpha=0.75)
  plt.show()

    # --- Calibration Curve ---
  alphas = np.linspace(0.01, 0.5, 20)  # 20 alpha values from 0.01 to 0.5
  empirical_coverages = []
  desired_coverages = []

  for alpha in alphas:
      q_hat_index = int(np.ceil((len(nonconformity_scores) + 1) * (1 - alpha))) - 1
      q_hat_loop = nonconformity_scores[min(max(q_hat_index, 0), len(nonconformity_scores) - 1)]

      current_correct_coverage_count = 0
      current_total_samples = 0

      with torch.no_grad():
          if T and autoencoder is not None:
              for images, labels in test_loader:
                  images, labels = images.to(device), labels.to(device)

                  z_0 = autoencoder.encoder(images)
                  t_tensor = torch.randint(0, T, (images.size(0),), device=device)
                  noise = torch.randn_like(z_0)
                  z_t = q_sample(z_0, t_tensor, noise, alphas_cumprod)

                  logits, _, _ = model(z_t, t_tensor)
                  ddm_prob = logits.sigmoid()

                  for j in range(images.size(0)):
                      true_label = labels[j].item()
                      prob_class1 = ddm_prob[j].item()

                      score_if_0 = prob_class1
                      score_if_1 = 1 - prob_class1

                      prediction_set = []
                      if score_if_0 <= q_hat_loop:
                          prediction_set.append(0)
                      if score_if_1 <= q_hat_loop:
                          prediction_set.append(1)

                      is_covered = (true_label in prediction_set)
                      if is_covered:
                          current_correct_coverage_count += 1
                      current_total_samples += 1

          else:
              for images, labels in test_loader:
                  images = images.to(device)
                  labels = labels.to(device)

                  logits = model(images).squeeze(1)
                  probs_class1 = torch.sigmoid(logits)

                  for j in range(images.size(0)):
                      true_label = labels[j].item()
                      prob_class1 = probs_class1[j].item()

                      score_if_0 = prob_class1
                      score_if_1 = 1 - prob_class1

                      prediction_set = []
                      if score_if_0 <= q_hat_loop:
                          prediction_set.append(0)
                      if score_if_1 <= q_hat_loop:
                          prediction_set.append(1)

                      is_covered = (true_label in prediction_set)
                      if is_covered:
                          current_correct_coverage_count += 1
                      current_total_samples += 1

      empirical_coverage = current_correct_coverage_count / current_total_samples
      empirical_coverages.append(empirical_coverage)
      desired_coverages.append(1 - alpha)

  # --- Plot calibration curve ---
  plt.figure(figsize=(8, 6))
  plt.plot(desired_coverages, empirical_coverages, marker='o', linestyle='-', label='Empirical Coverage')
  plt.plot(desired_coverages, desired_coverages, color='red', linestyle='--', label='Ideal Coverage (1 - alpha)')
  plt.title('Empirical Coverage vs. Desired Coverage')
  plt.xlabel('Desired Coverage (1 - alpha)')
  plt.ylabel('Empirical Coverage')
  plt.grid(True)
  plt.legend()
  plt.ylim(0, 1.05)
  plt.xlim(0, 1.05)
  plt.show()

