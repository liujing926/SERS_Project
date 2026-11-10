#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.utils import resample
from scipy.stats import beta


def bootstrap_accuracy_from_scores(image_scores, ground_truth_pairs, n_iterations=1000):
    """
    Calculate bootstrap confidence intervals based on fixed scores for each image.
    
    Args:
        image_scores: dict, {'8_1': 0.0723, '8_2': -0.1619, '8_3': -1.1112}
        ground_truth_pairs: list of tuples, [('8_1', '8_2', True), ...]
                           True means the first image is better.
        n_iterations: number of bootstrap iterations.
    """
    n_pairs = len(ground_truth_pairs)
    accuracies = []
    
    for _ in range(n_iterations):
        # Resample comparison pairs with replacement
        indices = resample(range(n_pairs), n_samples=n_pairs)
        
        correct = 0
        for idx in indices:
            img_a, img_b, a_better = ground_truth_pairs[idx]
            score_a = image_scores[img_a]
            score_b = image_scores[img_b]
            
            if a_better:
                correct += (score_a > score_b)
            else:
                correct += (score_b > score_a)
        
        acc = correct / n_pairs
        accuracies.append(acc)
    
    lower = np.percentile(accuracies, 2.5)
    upper = np.percentile(accuracies, 97.5)
    mean = np.mean(accuracies)
    
    return mean, lower, upper


def calculate_clopper_pearson_ci(k, n):
    """
    Calculate Clopper-Pearson 95% confidence interval for a binomial proportion.
    
    Args:
        k: number of correct predictions.
        n: total number of comparison pairs.
    """
    lower = beta.ppf(0.025, k, n - k + 1)
    upper = beta.ppf(0.975, k + 1, n - k)
    return lower, upper


if __name__ == '__main__':

    print("--- Bootstrap 95% CI ---")
    
    # Test data
    image_scores = {
        '8_1': 0.7139,
        '8_2': 0.4441,
        '8_3': -0.92843
    }

    # Ground truth ranking (assuming: 8_1 > 8_2 > 8_3)
    ground_truth_pairs = [
        ('8_1', '8_2', True),   # 8_1 is better than 8_2
        ('8_1', '8_3', True),   # 8_1 is better than 8_3
        ('8_2', '8_3', True),   # 8_2 is better than 8_3
    ]

    mean_acc, ci_lower, ci_upper = bootstrap_accuracy_from_scores(
        image_scores, ground_truth_pairs
    )

    print(f"Accuracy: {mean_acc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")

    # Clopper-Pearson Confidence Interval
    print("\n--- Clopper-Pearson 95% CI ---")
    
    # Validation results for 8 pairs (e.g., k=7 correct out of n=8)
    k = 7   # Number of correctly predicted pairs
    n = 8   # Total number of pairs

    lower, upper = calculate_clopper_pearson_ci(k, n)

    print(f"Accuracy: {k}/{n} = {k/n:.1%}")
    print(f"95% CI: [{lower:.1%}, {upper:.1%}]")