import torch
import torch.nn as nn
import torchvision.models as models

class DermAIModel(nn.Module):
    def __init__(self, num_classes=8, model_type="efficientnet_b4", dropout_rate=0.5):
        """
        Primary Model: EfficientNet-B4
        Mobile Model: MobileNetV3 (Small or Large)
        """
        super(DermAIModel, self).__init__()
        self.model_type = model_type
        
        if self.model_type == "efficientnet_b4":
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            # Replace final classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        elif self.model_type == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            in_features = self.backbone.classifier[3].in_features
            # Replace final classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Note: Softmax is usually applied implicitly via CrossEntropyLoss (or FocalLoss).
        # We add it here explicitly if you need output probabilities during inference.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.backbone(x)
        # return logits for loss function calculation
        return features
        
    def predict_proba(self, x):
        """For inference only - returns probabilities with Softmax applied."""
        logits = self.forward(x)
        return self.softmax(logits)

def build_optimizer_and_scheduler(model, learning_rate=1e-3):
    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    # LR Scheduler (Cosine Annealing) + Early stopping logic happens in train loop
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return optimizer, scheduler
