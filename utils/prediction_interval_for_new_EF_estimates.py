#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ========== 1. Extract Data ==========
# Scores from model output
scores = np.array([1.3900, -1.3508, -2.6481, -1.3884, -1.0507, 0.8152, -0.0337, 1.3588])
# True log10(EF) values
log10_ef_true = np.array([7.31, 4.00, 1.48, 4.00, 3.97, 5.32, 6.50, 7.02])

# ========== 2. Linear Fitting ==========
a, b = 1.2564, 5.4057
log10_ef_pred = a * scores + b

# Calculate R² and RMSE
r2 = r2_score(log10_ef_true, log10_ef_pred)
rmse = np.sqrt(np.mean((log10_ef_true - log10_ef_pred)**2))

# 95% Prediction Interval (approximate)
pi_width = 2 * rmse

# Print numerical results
print(f"R² = {r2:.4f}")
print(f"RMSE = {rmse:.4f} log10 units")
print(f"95% Prediction Interval = ±{pi_width:.4f} log10 units")

# Convert to EF fold-change factor
ef_factor = 10 ** pi_width
print(f"Equivalent to EF multiplied/divided by {ef_factor:.1f} fold")

# ========== 3. Plotting ==========
plt.figure(figsize=(8, 6))
plt.scatter(log10_ef_true, log10_ef_pred, color='blue', s=100, label='Anchor samples')

# Ideal line (y=x)
min_val = min(min(log10_ef_true), min(log10_ef_pred))
max_val = max(max(log10_ef_true), max(log10_ef_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')

# Prediction interval band
plt.fill_between(log10_ef_true, 
                 log10_ef_pred - pi_width,
                 log10_ef_pred + pi_width,
                 alpha=0.2, color='gray', label=f'95% PI (±{pi_width:.2f})')

plt.xlabel('True log10(EF)', fontsize=12)
plt.ylabel('Predicted log10(EF)', fontsize=12)
plt.title(f'Linear Mapping (R² = {r2:.3f}, RMSE = {rmse:.3f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ef_mapping_with_pi.png', dpi=300)
plt.show()