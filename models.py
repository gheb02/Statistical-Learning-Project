import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
import shap
from PIL import Image


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),    # (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*32*32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*32*32),
            nn.ReLU(),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # (3, 256, 256)
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

    
class TumorClassifier(nn.Module):
    def __init__(self, encoder, latent_dim=128):
        super().__init__()
        # freeze the encoder except for the last layer
        for param in encoder.parameters():
            param.requires_grad = False
        
        for param in encoder[-2:].parameters():
            param.requires_grad = True

        self.encoder = encoder  

        # Head di classificazione
        self.classifier = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(64, 1)
                )

    def forward(self, x):
        z = self.encoder(x)  # Extract compressed features from the input using the encoder

        if z.dim() > 2:  
            z = z.view(z.size(0), -1)  # Flatten the features if they have more than 2 dimensions (e.g., [B, C, H, W] -> [B, C*H*W])

        return self.classifier(z)  # Pass the flattened latent vector to the classifier for prediction



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(-torch.arange(half_dim, dtype=torch.float32, device=t.device) *
                         (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.block(x)  # Residual connection

class LatentDenoiserResidual(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = SinusoidalTimeEmbedding(latent_dim)
        self.input_norm = nn.LayerNorm(latent_dim * 2)
        self.input_linear = nn.Linear(latent_dim * 2, hidden_dim)

        self.res1 = ResidualBlock(hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 2 * latent_dim)  # Predict mean and log-variance

    def forward(self, z_t, t):
        t_emb = self.time_embed(t)                      # Compute time embedding
        x = torch.cat([z_t, t_emb], dim=1)              # Concatenate latent and time embedding
        x = self.input_norm(x)
        x = self.input_linear(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.output_norm(x)
        mean_logvar = self.output_linear(x)             # Predict both mean and log-variance

        mean_pred = mean_logvar[:, :self.latent_dim]    # First half: mean
        log_var_pred = mean_logvar[:, self.latent_dim:] # Second half: log-variance

        # Convert log-variance to standard deviation with numerical stability
        var_pred = torch.exp(log_var_pred.clamp(-10, 10))
        std_pred = torch.sqrt(var_pred)

        return mean_pred, std_pred  # Output noise distribution parameters



# Residual classifier model that jointly predicts class and latent noise
class LatentClassifierResidual(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim  # Needed for reshaping noise prediction

        # Time embedding for diffusion step t
        self.time_embed = SinusoidalTimeEmbedding(latent_dim // 2)
        
        # Project latent and time embeddings to a shared hidden space
        self.input_linear = nn.Linear(latent_dim, hidden_dim)
        self.time_linear = nn.Linear(latent_dim // 2, hidden_dim)

        self.initial_bn = nn.BatchNorm1d(hidden_dim)

        # Main residual block
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.act1 = nn.ReLU()

        self.middle_linear = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.act2 = nn.ReLU()

        # Classification head: predicts class (binary classification)
        self.classification_head = nn.Linear(hidden_dim // 2, 1)  # Output logit for BCEWithLogitsLoss

        # Noise prediction head: predicts mean and log-variance of the noise
        self.noise_mu_logvar_head = nn.Linear(hidden_dim // 2, 2 * latent_dim)

    def forward(self, z_t, t):
        # Embed timestep
        t_emb = self.time_embed(t)
        
        # Combine latent input and time embedding
        x = self.input_linear(z_t)
        t_proj = self.time_linear(t_emb)
        x = x + t_proj  # Time conditioning

        x = self.initial_bn(x)

        # Forward through backbone
        x = self.bn1(x)
        x = self.act1(self.dropout1(x))

        # Shared representation before heads
        shared_features = self.middle_linear(x)
        shared_features = self.bn2(shared_features)
        shared_features = self.act2(self.dropout2(shared_features))

        # Classification output (logits)
        classification_logits = self.classification_head(shared_features).squeeze(1)

        # Noise prediction output (mean and variance)
        mu_logvar_pred = self.noise_mu_logvar_head(shared_features)
        mu_pred = mu_logvar_pred[:, :self.latent_dim]
        logvar_pred = mu_logvar_pred[:, self.latent_dim:]

        # Numerical stability: clamp log-variance and compute std
        logvar_pred = logvar_pred.clamp(-10, 10)
        var_pred = torch.exp(logvar_pred)
        std_pred = torch.sqrt(var_pred)

        return classification_logits, mu_pred, std_pred



class ResNetBlock1D(nn.Module):
    """
    A simple ResNet-like block for 1D feature vectors.
    Consists of two linear layers with BatchNorm and ReLU,
    plus a dropout layer and a skip connection.
    """
    def __init__(self, in_features, out_features, stride=1, dropout_rate=0.2):
        super().__init__()
        
        # If input and output dimensions differ, or stride is not 1,
        # define a projection layer to match dimensions for the skip connection
        self.downsample = None
        if stride != 1 or in_features != out_features:
            # Here, stride means changing feature dimensions,
            # so downsample is a linear projection with batch normalization
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
        # Main residual block: two linear layers with BatchNorm, ReLU, and dropout in between
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # dropout added for regularization
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # Save input for skip connection
        
        out = self.block(x)  # Pass input through residual block
        
        # Apply projection on skip connection if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Add skip connection
        out = self.relu(out)  # Final activation
        
        return out



class DenoiserResNetClassifier(nn.Module):
    """
    Model combining an encoder, a denoiser, and a 1D ResNet classifier to classify noisy latent vectors.
    The encoder extracts latent features from input images.
    The denoiser predicts noise added at timestep t to recover the clean latent vector.
    The ResNet classifies the cleaned latent vector.
    """
    def __init__(self, encoder, denoiser, resnet_1d, latent_dim, T=1000, alphas_cumprod=None):
        super().__init__()
        self.encoder = encoder
        self.denoiser = denoiser
        self.resnet = resnet_1d
        self.T = T
        
        # Freeze encoder parameters so they are not updated during training
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Register alphas_cumprod as a persistent buffer (not a parameter)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def q_sample(self, z_0, t, noise):
        """
        Adds noise to the clean latent z_0 according to the diffusion schedule at timestep t.
        Uses cumulative product of alphas to scale clean latent and noise appropriately.
        """
        sqrt_alpha_cumprod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]) ** 0.5
        
        # Multiply by scalars expanded to match latent dimensions for broadcasting
        return sqrt_alpha_cumprod.unsqueeze(1) * z_0 + sqrt_one_minus_alpha_cumprod.unsqueeze(1) * noise

    def forward(self, x, t=None):
        # Encode input images into clean latent representation
        z_0 = self.encoder(x)
        
        # If timestep t is not provided, sample random timesteps for each batch element
        if t is None:
            t = torch.randint(0, self.T, (x.size(0),), device=x.device)
        
        # Sample noise with the same shape as latent vector
        noise = torch.randn_like(z_0)
        
        # Create noisy latent vector z_t by adding noise at timestep t
        z_t = self.q_sample(z_0, t, noise)
        
        # Predict noise mean and std at timestep t using the denoiser network
        mean_pred, std_pred = self.denoiser(z_t, t)
        
        # Recover estimate of clean latent vector z_0_hat by removing predicted noise
        sqrt_alpha_cumprod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]) ** 0.5
        
        # Adjust dimensions for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(1)
        
        # Formula for estimated clean latent vector
        z_0_hat = (z_t - sqrt_one_minus_alpha_cumprod * mean_pred) / sqrt_alpha_cumprod
        
        # Classify the cleaned latent vector using the ResNet classifier
        output = self.resnet(z_0_hat)
        
        return output


class TumorClassifierRes(nn.Module):
    def __init__(self, encoder, latent_dim=128):
        super().__init__()
        # Freeze the encoder, last layer excluded
        for param in encoder.parameters():
            param.requires_grad = False

        # Unfreeze from index 6 till the end of encoder
        for i in range(6, len(encoder)):
            for param in encoder[i].parameters():
                param.requires_grad = True

        # Trained econder
        self.encoder = encoder  

        self.classifier = nn.Sequential(
            # First block: latent_dim: 128
            ResNetBlock1D(latent_dim, 128),
            nn.Dropout(0.3),

            # Second block: 128 -> 64
            ResNetBlock1D(128, 64),
            nn.Dropout(0.3),

            # Final classification layer
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Get compressed features from encoder
        z = self.encoder(x)
        return self.classifier(z)
    


# Generic residual block used for fully connected layers
class ResidualBlockMulti(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),       # First linear layer
            nn.BatchNorm1d(dim),       # Normalization
            nn.ReLU(),                 # Activation
            nn.Dropout(dropout),       # Regularization
            nn.Linear(dim, dim),       # Second linear layer
            nn.BatchNorm1d(dim)        # Normalization again
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        return self.relu(out + x)  # Residual connection: input is added to the output
        

# Classifier using residual blocks on top of an encoder
class TumorMultiClassifier(nn.Module):
    def __init__(self, encoder, latent_dim=128, n_classes=4):
        super().__init__()

        # Freeze all encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last two child modules of the encoder (for fine-tuning)
        for param in list(encoder.children())[-2:]:
            for p in param.parameters():
                p.requires_grad = True

        self.encoder = encoder

        # Initial projection layer (latent_dim -> 256)
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 5 residual blocks for deep feature extraction (can be increased)
        self.res_blocks = nn.Sequential(
            ResidualBlockMulti(256),
            ResidualBlockMulti(256),
            ResidualBlockMulti(256),
            ResidualBlockMulti(256),
            ResidualBlockMulti(256)
        )

        # Output head: 256 -> 128 -> number of classes
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        z = self.encoder(x)  # Pass input through the encoder
        if z.dim() > 2:
            z = z.view(z.size(0), -1)  # Flatten if necessary

        x = self.input_layer(z)       # Initial projection
        x = self.res_blocks(x)        # Residual block stack
        return self.output_layer(x)   # Final prediction (logits)



# Wrapper class to avoid squeezing the output tensor
# Keeps the output shape as [batch_size, 1], which SHAP requires
class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Forward pass without squeezing
        return self.model(x)
    

class TumorImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)  
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        label = row["label_encoded"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)



class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, add_attention=True,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        """
        Customizable ResNet architecture.

        Args:
            block (nn.Module): Type of building block (e.g., BasicBlock or Bottleneck).
            layers (list): A list of integers indicating the number of blocks in each of the 4 stages.
            num_classes (int): Number of output classes for the final classification layer.
            add_attention (bool): Whether to add CBAM attention modules after each stage.
            zero_init_residual (bool): If True, zero-initialize the last BN in each
                                       residual branch, so that the residual branch starts
                                       with zeros, and each residual block behaves like an identity.
                                       This improves the accuracy of the model at the beginning of training.
            groups (int): Number of blocked connections from input channels to output channels.
            width_per_group (int): Base width of the output channels for each group.
            replace_stride_with_dilation (list, optional): If not None, a list of booleans
                                                            indicating whether to replace
                                                            the 2x2 stride with a dilated
                                                            convolution in stages 2, 3, and 4.
            norm_layer (callable, optional): Normalization layer to use (default: BatchNorm2d).
        """
        super(CustomResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Final pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last Bottleneck in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # Attention modules
        self.add_attention = add_attention
        if self.add_attention:
            # CBAM modules are instantiated based on the output channels of each layer
            self.cbam1 = CBAM(64 * block.expansion)
            self.cbam2 = CBAM(128 * block.expansion)
            self.cbam3 = CBAM(256 * block.expansion)
            self.cbam4 = CBAM(512 * block.expansion)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.add_attention:
            x = self.cbam1(x)

        x = self.layer2(x)
        if self.add_attention:
            x = self.cbam2(x)

        x = self.layer3(x)
        if self.add_attention:
            x = self.cbam3(x)

        x = self.layer4(x)
        if self.add_attention:
            x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        return x