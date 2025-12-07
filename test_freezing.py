
import torch
import sys
import os

# Add repo root to path
sys.path.append(os.getcwd())

from models import DiT_models

def verify_freezing():
    print("Initializing model...")
    model = DiT_models['DiT-XL/2'](
        input_size=32,
        num_classes=1000
    )
    
    print("Applying freezing logic...")
    # --- FREEZING LOGIC COPIED FROM SCRIPT ---
    for p in model.parameters():
        p.requires_grad = False
    
    # Check x2_embedder
    for p in model.x2_embedder.parameters():
        p.requires_grad = True
    
    # Check x2_vit_block
    for p in model.x2_vit_block.parameters():
        p.requires_grad = True
        
    if model.x2_vit_proj_in is not None:
        for p in model.x2_vit_proj_in.parameters():
            p.requires_grad = True
            
    if model.x2_vit_proj_out is not None:
        for p in model.x2_vit_proj_out.parameters():
            p.requires_grad = True
    # -----------------------------------------

    print("Verifying parameters...")
    
    # 1. Verify backbone is frozen
    frozen_params = [
        model.x_embedder.proj.weight,
        model.t_embedder.mlp[0].weight,
        model.blocks[0].attn.qkv.weight,
        model.final_layer.linear.weight
    ]
    for p in frozen_params:
        assert not p.requires_grad, f"Backbone parameter {p.shape} should be frozen!"
        
    # 2. Verify x2 branch is unfrozen
    unfrozen_params = [
        model.x2_embedder.proj.weight,
        model.x2_vit_block.norm1.weight
    ]
    if model.x2_vit_proj_in is not None:
        unfrozen_params.append(model.x2_vit_proj_in.weight)

    for p in unfrozen_params:
        assert p.requires_grad, f"x2 parameter {p.shape} should be unfrozen!"
        
    print("SUCCESS: Freezing logic verified correctly.")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,} ({trainable_params/total_params:.2%})")

if __name__ == "__main__":
    verify_freezing()
