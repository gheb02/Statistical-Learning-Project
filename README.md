# Robust Brain MRI Classification on Latent Representations Leveraging Denoising Diffusion Probabilistic Models

## Abstract

This project arises from the need to explore new techniques to improve image classification in the medical field, with particular focus on the reliability and robustness of predictions. The main motivation stems from the frequent presence of noise in medical images, which can undermine the accuracy of traditional classification models. Inspired by the recent research work ”Robust Classification via a Single Diffusion Model Classification”, we decided to implement a denoising diffusion model aimed at enhancing the model’s ability to handle noisy data, as well as leveraging this approach for data augmentation to effectively expand the training dataset. During development, we realized that using an autoencoder to work in a latent space could simplify the process and help improve model robustness Indeed, the latent dimension compresses information into a more compact, potentially more stable, and noise-resilient representation. The project structure involved building an autoencoder, followed by training a MLP classifieron the latent dimension output by the encoder. Subsequently, we integrated a diffusion model capable of providing noise-related information to the system, thus training the model to classify accurately even under uncertainty and image degradation. Finally, the complete model combines the encoder, a denoising module that “corrupts” and then “cleans” the latent representation, and a ResNet for final classification.
This approach aims to enhance the safety and accuracy of classification in the critical medical context, demonstrating the potential of diffusion techniques and latent representations in handling noisy images.

---

## Repository Structure

### Jupyter Notebooks

These notebooks contain the core experimental workflows, including model training, evaluation, and interpretability analyses.

- `main.ipynb`: This comprehensive notebook covers the training and evaluation of the **Autoencoder**, **Encoder + MLP**, **DDPM** (Denoising Diffusion Probabilistic Model), and the **Noise-Aware Classifier**. It also includes the implementation of **SHAP** for model interpretability and **Conformal Prediction** for uncertainty quantification. Finally, it details the complete **Encoder + DDPM + ResNet hybrid pipeline**.


- `CNN_on_images.ipynb`: Simple **Convolutional Neural Network (CNN)** trained and tested directly on raw image data.

- `CNN_Attention.ipynb`: **CNN architecture incorporating an attention mechanism** designed to enhance post-hoc model interpretability when trained on raw images.

- `Resnet50.ipynb`: Focuses on the **fine-tuning of a ResNet50 model** for the project's classification task.

- `ResNet Classifier Head.ipynb`: **ResNet-based classifier head** trained on latent representations, designed to work in conjunction with the encoder.

- `multi_class.ipynb`: Adapts both the **CNN and Encoder + MLP architectures for a multi-class classification** task, specifically classifying healthy brains and three different types of cancer.

--- 
### Python Scripts

These files contain reusable code for model definitions and utility functions.

- `models.py`: Contains the **definitions of all the architectures** utilized throughout the project.

- `utils.py`: Provides a collection of **utility functions** essential for various tasks, including data loading, preprocessing, model training, and evaluation metrics.

---

## Project Report

A detailed explanation of the project's methodology, experiments, results, and conclusions can be found in the [Final Project Report](<Robust Brain MRI Classification on Latent Representations Leveraging Denoising Diffusion Probabilistic Models.pdf>).
