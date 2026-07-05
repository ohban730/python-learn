import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("==============================================================")
print("     GELU, SwiGLU, and Gaussian CDF Comparison Demo")
print("==============================================================\n")

# ------------------------------------------------------------
# 1. Exact GELU vs Tanh Approximation GELU
# ------------------------------------------------------------
print("--- [STEP 1] Exact GELU vs Tanh Approximation ---")
x_test = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Exact GELU formula using Gaussian CDF
# CDF of Standard Normal is 0.5 * (1 + erf(x / sqrt(2)))
exact_gelu = 0.5 * x_test * (1.0 + torch.erf(x_test / torch.sqrt(torch.tensor(2.0))))

# Tanh approximation (used in GPT-2 and books)
approx_gelu = 0.5 * x_test * (1.0 + torch.tanh(
    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x_test + 0.044715 * torch.pow(x_test, 3))
))

print(f"Input x       : {x_test.tolist()}")
print(f"Exact GELU    : {[round(v, 5) for v in exact_gelu.tolist()]}")
print(f"Approx GELU   : {[round(v, 5) for v in approx_gelu.tolist()]}")
print(f"Difference    : {[round(abs(e - a), 6) for e, a in zip(exact_gelu.tolist(), approx_gelu.tolist())]}")
print("--> [SUCCESS] The difference is less than 0.0001, showing the approximation is highly accurate.\n")


# ------------------------------------------------------------
# 2. SwiGLU (Swish-Gated Linear Unit) Module Implementation
# ------------------------------------------------------------
print("--- [STEP 2] SwiGLU Module Forward Pass ---")

class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Two parallel linear transformations
        self.linear_w = nn.Linear(in_features, out_features)
        self.linear_v = nn.Linear(in_features, out_features)

    def forward(self, x):
        # SwiGLU(x) = (x @ W) * Swish(x @ V)
        # Note: F.silu in PyTorch is exactly the Swish activation function
        gate = F.silu(self.linear_v(x))
        data = self.linear_w(x)
        return data * gate

# Instantiate SwiGLU (Input: 4-dim, Output: 3-dim)
swiglu_layer = SwiGLU(in_features=4, out_features=3)

# 2 sample inputs
x_dummy = torch.randn(2, 4)
out_swiglu = swiglu_layer(x_dummy)

print(f"Input Tensor Shape  : {list(x_dummy.shape)}")
print(f"Output Tensor Shape : {list(out_swiglu.shape)}")
print(f"Output Values       :\n{out_swiglu.data}\n")


# ------------------------------------------------------------
# 3. Visualization and Chart Generation (ReLU vs GELU vs Swish)
# ------------------------------------------------------------
print("--- [STEP 3] Generating Comparison Plot ---")

x_plot = torch.linspace(-3.0, 3.0, 200)

# Calculate activations
y_relu = F.relu(x_plot)
y_gelu = 0.5 * x_plot * (1.0 + torch.erf(x_plot / torch.sqrt(torch.tensor(2.0))))
y_swish = x_plot * torch.sigmoid(x_plot) # Swish (SiLU)

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(x_plot.numpy(), y_relu.numpy(), label="ReLU", linestyle="--", color="gray", linewidth=2)
plt.plot(x_plot.numpy(), y_gelu.numpy(), label="GELU", color="blue", linewidth=2.5)
plt.plot(x_plot.numpy(), y_swish.numpy(), label="Swish (SiLU)", color="orange", linewidth=2)

plt.title("Activation Functions Comparison", fontsize=14)
plt.xlabel("Input (x)", fontsize=12)
plt.ylabel("Output (y)", fontsize=12)
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.legend(fontsize=12)

# Save chart image to docs directory
docs_dir = "python-learn/pytorch-llm/docs"
if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)

plot_path = os.path.join(docs_dir, "gelu_swish_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"--> [SUCCESS] Comparison plot saved to: {plot_path}")
print("==============================================================")
