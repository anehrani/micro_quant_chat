"""
Token analysis and visualization utilities.
Analyze token sequences and understand what the model learns.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import torch


class TokenAnalyzer:
    """Analyze token sequences and statistics"""

    def __init__(self, tokens: List[int]):
        """Initialize with a list of token IDs"""
        self.tokens = np.array(tokens)
        self.vocab_size = len(set(tokens))

    def vocabulary_stats(self) -> Dict:
        """Get vocabulary statistics"""
        counter = Counter(self.tokens)
        frequencies = np.array([counter[i] for i in range(self.vocab_size)])

        return {
            "vocab_size": self.vocab_size,
            "total_tokens": len(self.tokens),
            "unique_tokens": len(counter),
            "coverage_top_10": np.sum(frequencies[np.argsort(-frequencies)[:10]]) / len(self.tokens),
            "avg_token_frequency": frequencies.mean(),
            "entropy": self._entropy(frequencies),
        }

    def sequence_stats(self) -> Dict:
        """Get sequence statistics"""
        return {
            "mean_token": float(self.tokens.mean()),
            "std_token": float(self.tokens.std()),
            "min_token": int(self.tokens.min()),
            "max_token": int(self.tokens.max()),
            "median_token": float(np.median(self.tokens)),
        }

    def transition_matrix(self) -> np.ndarray:
        """Get token transition probabilities (n-gram)"""
        matrix = np.zeros((self.vocab_size, self.vocab_size))
        for i in range(len(self.tokens) - 1):
            curr_token = self.tokens[i]
            next_token = self.tokens[i + 1]
            matrix[curr_token, next_token] += 1

        # Normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums

        return matrix

    def most_common_transitions(self, top_k: int = 10) -> List[Tuple]:
        """Get most common token transitions"""
        transition_counts = {}
        for i in range(len(self.tokens) - 1):
            curr = self.tokens[i]
            next = self.tokens[i + 1]
            key = (curr, next)
            transition_counts[key] = transition_counts.get(key, 0) + 1

        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_transitions[:top_k]

    def perplexity_by_position(self, window_size: int = 100) -> np.ndarray:
        """Compute perplexity in sliding windows"""
        perplexities = []
        for i in range(len(self.tokens) - window_size):
            window = self.tokens[i : i + window_size]
            counter = Counter(window)
            probs = np.array([counter[j] / len(window) for j in range(self.vocab_size)])
            probs = probs[probs > 0]  # Remove zeros
            entropy = -np.sum(probs * np.log(probs))
            perplexities.append(np.exp(entropy))

        return np.array(perplexities)

    @staticmethod
    def _entropy(frequencies: np.ndarray) -> float:
        """Compute entropy of a frequency distribution"""
        probs = frequencies / frequencies.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def summary(self) -> str:
        """Get summary statistics"""
        vocab = self.vocabulary_stats()
        seq = self.sequence_stats()

        text = "Token Analysis Summary\n"
        text += "=" * 50 + "\n"
        text += "\nVocabulary Statistics:\n"
        for k, v in vocab.items():
            if isinstance(v, float):
                text += f"  {k}: {v:.4f}\n"
            else:
                text += f"  {k}: {v}\n"

        text += "\nSequence Statistics:\n"
        for k, v in seq.items():
            text += f"  {k}: {v:.4f}\n"

        return text


def load_tokens(filepath: str) -> List[int]:
    """Load tokens from space-separated file"""
    with open(filepath, "r") as f:
        tokens = list(map(int, f.read().strip().split()))
    return tokens


def save_tokens(tokens: List[int], filepath: str):
    """Save tokens to space-separated file"""
    with open(filepath, "w") as f:
        f.write(" ".join(map(str, tokens)))


class TokenPatternFinder:
    """Find interesting patterns in token sequences"""

    def __init__(self, tokens: List[int]):
        self.tokens = tokens

    def find_repeating_patterns(self, pattern_len: int = 3, min_repeats: int = 5) -> List[Tuple]:
        """Find repeating patterns of given length"""
        patterns = {}
        for i in range(len(self.tokens) - pattern_len + 1):
            pattern = tuple(self.tokens[i : i + pattern_len])
            if pattern not in patterns:
                patterns[pattern] = 0
            patterns[pattern] += 1

        # Filter by minimum repeats
        repeating = [(p, c) for p, c in patterns.items() if c >= min_repeats]
        repeating.sort(key=lambda x: x[1], reverse=True)

        return repeating

    def find_outliers(self, threshold: float = 3.0) -> List[int]:
        """Find tokens that are statistical outliers (unusual moves)"""
        tokens_arr = np.array(self.tokens)
        mean = tokens_arr.mean()
        std = tokens_arr.std()

        outlier_indices = np.where(np.abs(tokens_arr - mean) > threshold * std)[0]
        return list(outlier_indices)

    def token_bigram_entropy(self) -> float:
        """Compute entropy of bigram distribution"""
        bigrams = {}
        for i in range(len(self.tokens) - 1):
            bigram = (self.tokens[i], self.tokens[i + 1])
            bigrams[bigram] = bigrams.get(bigram, 0) + 1

        probs = np.array(list(bigrams.values())) / sum(bigrams.values())
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze token sequences")
    parser.add_argument("--tokens_file", type=str, default="data/all_tokens.txt", help="Token file to analyze")
    parser.add_argument("--patterns", action="store_true", help="Find repeating patterns")
    parser.add_argument("--outliers", action="store_true", help="Find outlier tokens")

    args = parser.parse_args()

    print("Loading tokens...")
    tokens = load_tokens(args.tokens_file)

    analyzer = TokenAnalyzer(tokens)
    print(analyzer.summary())

    if args.patterns:
        print("\nMost Common 3-gram Patterns:")
        patterns = TokenPatternFinder(tokens).find_repeating_patterns(pattern_len=3, min_repeats=10)
        for pattern, count in patterns[:10]:
            print(f"  {pattern}: {count} times")

    if args.outliers:
        print("\nFinding Outlier Tokens (3-sigma):")
        finder = TokenPatternFinder(tokens)
        outliers = finder.find_outliers(threshold=3.0)
        print(f"  Found {len(outliers)} outliers")
        if len(outliers) > 0:
            print(f"  First 10: {outliers[:10]}")
