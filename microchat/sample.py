from __future__ import annotations

from .ckpt import load_checkpoint


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="nanochat-style sampling")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--device", default="auto")

    # Backwards-compatible flags (src/generate.py used these names)
    p.add_argument("--seed", "--seed_tokens", dest="seed", default="80 81 83 89 66")
    p.add_argument("--num", "--num_generate", dest="num", type=int, default=50)
    p.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.8)
    p.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=50)
    p.add_argument("--samples", "--num_samples", dest="samples", type=int, default=3)
    args = p.parse_args()

    model, _, dev, _ = load_checkpoint(args.checkpoint, device=args.device)
    seed_tokens = list(map(int, args.seed.split()))

    for i in range(args.samples):
        out = seed_tokens.copy()
        for t in model.generate(seed_tokens, args.num, temperature=args.temp, top_k=args.top_k):
            out.append(t)
        print(f"sample {i + 1}: {out}")


if __name__ == "__main__":
    main()
