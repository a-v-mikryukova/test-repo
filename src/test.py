import hydra
import torch
import wandb
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.data import LibriSpeechDataset, TextTransform, collate_fn
from src.models import SpeechRecognitionModel, greedy_decode
from src.utils import WanDBLogger, cer, wer


@hydra.main(config_path="../configs", config_name="config")
def test(config) -> None:
    checkpoint = config["test"]["checkpoint"]
    logger = WanDBLogger(dict(config))
    logger.watch_model = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpeechRecognitionModel(**config["model"]).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    except Exception as e:
        msg = f"Error loading checkpoint: {e!s}"
        raise RuntimeError(msg)

    text_transform = TextTransform()
    test_datasets = []
    for url in config["data"]["urls"]["test"]:
        dataset = LibriSpeechDataset(
            root=config["data"]["data_dir"],
            url=url,
            download=False,
        )
        test_datasets.append(dataset)

    test_dataset = ConcatDataset(test_datasets)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=lambda x: collate_fn(x, text_transform, "test"),
        num_workers=config["train"]["num_workers"],
        shuffle=False,
    )

    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0
    examples = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels, _input_lengths, label_lengths) in tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc="Testing",
        ):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            decoded_preds, decoded_targets = greedy_decode(
                outputs.cpu(),
                labels.cpu(),
                label_lengths,
                text_transform,
            )

            batch_cer = sum(cer(t, p) for t, p in zip(decoded_targets, decoded_preds))
            batch_wer = sum(wer(t, p) for t, p in zip(decoded_targets, decoded_preds))

            total_cer += batch_cer
            total_wer += batch_wer
            total_samples += len(decoded_targets)

            if batch_idx == 0:
                examples = [
                    (target, pred, cer(target, pred), wer(target, pred))
                    for target, pred in zip(decoded_targets[:5], decoded_preds[:5])
                ]

    avg_cer = total_cer / total_samples
    avg_wer = total_wer / total_samples

    results_table = wandb.Table(
        columns=["Target", "Prediction", "CER", "WER"],
        data=examples,
    )

    logger.log_metrics({
        "test/cer": avg_cer,
        "test/wer": avg_wer,
        "test/examples": results_table,
    })

    for _target, _pred, _c, _w in examples:
        pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="ASR Model Testing")
    # parser.add_argument("--config", default="configs/config.yaml")
    # parser.add_argument("--checkpoint", required=True)
    # args = parser.parse_args()
    test()
