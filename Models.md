## Modelli Untouchable:

- Autoencoder

## Modelli da ri-testare:

- Tumor Classifier (Autoencoder + CNN Head):

    -  Threshold: 33%
    - Accuracy: 93.6%
    - Precision: 90.71 %
    - Recall: 98.39 %
    - F1-Score: 94.39%
    - 4 False Negatives on 453 test images
    - Epochs: 20
    - Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-4)
    - Loss: Binary Cross-Entropy (BCEWithLogitsLoss)
    - Layer: 3 (Neurons 128, 64)

- Tumor Classifier Res (Autoencoder + CNN con Residual Blocks):

    - Epochs: (50 total)


- Denoiser Classifier:

    - Threshold: 49%
    - Accuracy: 90.73%
    - Precision: 86.27%
    - Recall: 98.79%
    - F1-Score: 92.11%
    - Epochs: 54 With Early Stopping (100 total)
    - Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
    - Loss: Binary Cross-Entropy (BCEWithLogitsLoss)
    - Scheduler: Mode: "min", factor = 0.3, patience = 5