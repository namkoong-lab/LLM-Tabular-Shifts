import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, original_layer, rank=1):
        super(LoRA, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        # Initialize LoRA layers
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        # Freeze the original layer's weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        # Move LoRA layers to the same device as the original layer
        device = next(original_layer.parameters()).device
        self.to(device)

    def forward(self, x):
        return self.original_layer(x) + self.lora_B(self.lora_A(x))
    
def apply_lora_to_linear_layers(model, rank):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRA(module, rank))
        else:
            apply_lora_to_linear_layers(module, rank)

    # Verify that the original layer's parameters are frozen
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            for param in module.original_layer.parameters():
                assert not param.requires_grad, f"Error! Parameter {name} requires grad"