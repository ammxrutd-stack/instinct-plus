"""Train the thumbnail model. Thin wrapper so `python -m src.thumbnail.train`
works; the shared logic lives in src/modelling/train.py."""

from src.modelling.train import main

if __name__ == "__main__":
    main(["thumbnail"])
