from pathlib import Path
from PIL.Image import Image
from transformers import pipeline
from transformers.image_utils import load_images


def get_files():
    dir = Path("imgs")
    files = [str(f) for f in dir.iterdir() if f.is_file()]
    return files


def get_classifier():
    model = "google/siglip2-base-patch16-512"
    classifier = pipeline(model=model, task="zero-shot-image-classification")
    return classifier


def main():
    image_classifier = get_classifier()
    candidate_labels = [
        "happy",
        "sad",
        "laugh",
        "cry",
        "action",
        "fantasy",
        "comedy",
        "romance",
        "boys",
        "girls",
    ]
    imgs: list[Image] = load_images(get_files())  # pyright: ignore[reportAssignmentType]

    for img in imgs:
        outputs = image_classifier(img, candidate_labels)
        print(outputs)


if __name__ == "__main__":
    main()
