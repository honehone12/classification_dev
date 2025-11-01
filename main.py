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
        "For a Happy Day",
        "For a Sad Day",
        "Want to Laugh",
        "Want to Cry",
        "Battle Action",
        "Fantasy World",
        "Gag Comedy",
        "Love Romance",
        "Historical",
        "Heroic Drama",
        "For Boys",
        "For Girls",
        "Youth Ensemble",
        "Dark Fantasy",
        "Near-Future Sci-Fi",
        "Japanese Folklore",
        "Slice of Life",
        "School Setting",
        "Otherworldly Realm",
        "Futuristic City",
        "Warring States Era",
        "Outer Space",
    ]
    files = get_files()
    imgs: list[Image] = load_images(files)  # pyright: ignore[reportAssignmentType]
    iter = zip(files, imgs)

    for file, img in iter:
        with img:
            rgb = img.convert("RGB")
            outputs = image_classifier(rgb, candidate_labels)
            print("\n******")
            print(file)
            print("******")
            for kv in outputs:
                print(f"{kv['label']} = {kv['score']}")
            print("******\n")


if __name__ == "__main__":
    main()
