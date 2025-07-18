## Modelli Untouchable:

- Autoencoder

## Modelli Finali

1. CNN Bruco

    - 

    1.1 Conformal Prediction:
            
            - 

2. CNN Attention

    - 
    
    2.1 Conformal Prediction:
            
            - 

3. Tumor Classifier (Autoencoder + CNN Head):

    - Threshold: 33%
    - Accuracy: 93.6%
    - Precision: 90.71 %
    - Recall: 98.39 %
    - F1-Score: 94.39%
    - 4 False Negatives on 453 test images
    - Epochs: 20
    - Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-4)
    - Loss: Binary Cross-Entropy (BCEWithLogitsLoss)
    - Layer: 3 (Neurons 128, 64)

        3.1 Conformal Prediction:
            
            - q_hat: 0.4438
            - empirical coverage: 0.9135
            - avg_set_size = 0.9808

4. Tumor Classifier Res (Autoencoder + CNN con Residual Blocks):

    - Epochs: (50 total)

        4.1 Conformal Prediction:
            
            - 


5. Denoiser Classifier:

    - Threshold: 47%
    - Accuracy: 93.27%
    - Precision: 91.49%
    - Recall: 96.41%
    - F1-Score: 93.89%
    - Epochs: 32 With Early Stopping (100 total)
    - Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
    - Loss: Binary Cross-Entropy (BCEWithLogitsLoss)
    - Scheduler: Mode: "min", factor = 0.3, patience = 5

    5.1 Conformal Prediction:
            
            - q_hat: 0.5042
            - empirical coverage: 0.9351
            - avg_set_size: 1.0144


6. Autoencoder + Denoiser + ResNet

    - Threshold: 26%
    - Accuracy: 93.27%
    - Precision: 91.49%
    - Recall: 96.41%
    - F1-Score: 93.89%
    - Epochs: 22 with Early Stopping (50 total)
    - Optimizer: AdamW (lr=1e-4, weight_decay=1e-3)
    - Loss: Binary Cross-Entropy (BCEWithLogitsLoss)

     6.1 Conformal Prediction:

        - q_hat = 0.0866
        - empirical coverage: 0.8918
        - avg_set_size = 0.9159