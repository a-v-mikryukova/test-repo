import torch
import yaml
import argparse
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from src.data.dataset import LibriSpeechDataset
from src.data.preprocess import TextTransform, collate_fn
from src.models.model import SpeechRecognitionModel
from src.models.decoder import greedy_decode
from src.utils.metrics import cer, wer
from src.utils.logger import WanDBLogger


def test(config_path, checkpoint):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = WanDBLogger(config)
    logger.watch_model = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpeechRecognitionModel(**config['model']).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint from {checkpoint}")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")

    text_transform = TextTransform()
    test_datasets = []
    for url in config['data']['urls']['test']:
        dataset = LibriSpeechDataset(
            root=config['data']['data_dir'],
            url=url,
            download=False
        )
        test_datasets.append(dataset)

    test_dataset = ConcatDataset(test_datasets)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=lambda x: collate_fn(x, text_transform, "test"),
        num_workers=4,
        shuffle=False
    )

    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0
    examples = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels, input_lengths, label_lengths) in tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc="Testing"
        ):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            decoded_preds, decoded_targets = greedy_decode(
                outputs.cpu(),
                labels.cpu(),
                label_lengths,
                text_transform
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
        data=examples
    )

    logger.log_metrics({
        "test/cer": avg_cer,
        "test/wer": avg_wer,
        "test/examples": results_table
    })

    print("\nTest Results:")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print("\nExample predictions:")
    for target, pred, c, w in examples:
        print(f"Target: {target}")
        print(f"Pred:   {pred}")
        print(f"CER: {c:.2f} | WER: {w:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Model Testing")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    test(args.config, args.checkpoint)