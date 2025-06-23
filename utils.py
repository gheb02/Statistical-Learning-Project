import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
import shap
import lime

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        # Unsqueeze labels to match the model output shape [batch_size, 1]
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
        # Note: torch.max is typically used with multi-class output.
        # For binary classification with BCEWithLogitsLoss and output shape [batch_size, 1],
        # the output is a logit for class 1. You should threshold the sigmoid output
        # to get the predicted class (0 or 1).
        # Convert logits to probabilities and then to binary predictions
        predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze(1) # Squeeze back to [batch_size] for comparison

        total_samples += labels.size(0) # Use original labels.size(0) before unsqueeze
        correct_predictions += (predicted == labels.squeeze(1)).sum().item() # Compare with squeezed labels

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

# --- Validation Loop ---
def validate_epoch(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            # Unsqueeze labels to match the model output shape [batch_size, 1]
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            # Convert logits to probabilities and then to binary predictions
            predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze(1) # Squeeze back to [batch_size] for comparison

            total_samples += labels.size(0) # Use original labels.size(0) before unsqueeze
            correct_predictions += (predicted == labels.squeeze(1)).sum().item() # Compare with squeezed labels

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


# --- Test Loop ---
# The test_model function provided in the original code already calculates precision and recall.
# However, it uses torch.max which is incorrect for single-logit binary output.
# It also has the same label shape issue.
# Modify it similarly to the train/val loops for prediction logic and label shape.

def test_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_labels = []
    all_predicted = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Unsqueeze labels for loss calculation if needed, but store original for metrics
            labels_loss = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels_loss)

            running_loss += loss.item() * inputs.size(0)

            # Convert logits to probabilities and then to binary predictions
            predicted_probs = torch.sigmoid(outputs)
            predicted_classes = (predicted_probs > 0.5).long().squeeze(1) # Get 0 or 1 prediction

            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted_classes.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    # Use sklearn metrics with the collected lists
    test_accuracy = accuracy_score(all_labels, all_predicted)
    # average='weighted' is appropriate for potentially imbalanced binary classes
    test_precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_predicted, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_predicted, average='weighted', zero_division=0)


    return test_loss, test_accuracy, test_precision, test_recall, test_f1



def conformal_prediction(model, alpha, calibration_loader, device):
  model.to(device)
  model.eval()

  nonconformity_scores = []

  with torch.no_grad():
      for images, labels in calibration_loader:
          images = images.to(device)
          labels = labels.to(device)

          logits = classifier_model(images).squeeze(1) # Output is [batch_size, 1], squeeze to [batch_size]
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
    for i, (images, labels) in enumerate(calibration_loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = classifier_model(images).squeeze(1)
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


  alphas = np.linspace(0.01, 0.5, 20) # Test a range of alpha values from 1% to 50%
  empirical_coverages = []
  desired_coverages = []

  for alpha in alphas:
      q_hat_index = int(np.ceil((len(nonconformity_scores) + 1) * (1 - alpha))) - 1
      if q_hat_index < 0:
          q_hat_loop = nonconformity_scores[0]
      elif q_hat_index >= len(nonconformity_scores):
          q_hat_loop = nonconformity_scores[-1]
      else:
          q_hat_loop = nonconformity_scores[q_hat_index]

      current_correct_coverage_count = 0
      current_total_samples = 0

      with torch.no_grad():
          for images, labels in calibration_loader:
              images = images.to(device)
              labels = labels.to(device)

              logits = classifier_model(images).squeeze(1)
              probs_class1 = torch.sigmoid(logits)

              for j in range(images.size(0)):
                  true_label = labels[j].item()
                  prob_class1 = probs_class1[j].item()

                  score_if_0 = prob_class1
                  score_if_1 = 1 - prob_class1

                  prediction_set = []
                  if score_if_0 <= q_hat_loop: # Use q_hat_loop for current alpha
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

  plt.figure(figsize=(8, 6))
  plt.plot(desired_coverages, empirical_coverages, marker='o', linestyle='-', label='Empirical Coverage')
  plt.plot(desired_coverages, desired_coverages, color='red', linestyle='--', label='Desired Coverage (1-alpha)')
  plt.title('Empirical Coverage vs. Desired Coverage')
  plt.xlabel('Desired Coverage (1 - alpha)')
  plt.ylabel('Empirical Coverage')
  plt.grid(True)
  plt.legend()
  plt.ylim(0, 1.05)
  plt.xlim(0, 1.05)
  plt.show()
