"""
Quick local test to verify skip connection logging works
Run this locally: python test_skip_connection.py
"""
import torch
from models import DiT_XL_2

print("="*60)
print("Testing DiT with skip connection logging...")
print("="*60)

# Create model
print("\nCreating DiT-XL/2 model...")
model = DiT_XL_2(input_size=32, num_classes=1000)
model.eval()

# Create dummy inputs (simulating one diffusion step)
print("\nSimulating a forward pass...")
batch_size = 2
x = torch.randn(batch_size, 4, 32, 32)
t = torch.tensor([500, 500])  # Middle of diffusion process
y = torch.tensor([207, 360])  # Golden retriever, otter

print("\n" + "="*60)
with torch.no_grad():
    output = model.forward(x, t, y)
print("="*60)

print(f"\n✓ Forward pass completed! Output shape: {output.shape}")
print("\nIf you see '[DiT Forward]' messages above, the skip connection is working!")
print("="*60)

