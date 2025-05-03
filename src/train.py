import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from src.data import LibriSpeechDataset
from src.data import TextTransform, get_featurizer, collate_fn
from src.models import SpeechRecognitionModel
from src.models import greedy_decode
from src.utils import cer, wer
from src.utils import WanDBLogger
import wandb
import torch.nn.functional as F


def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = WanDBLogger(config)
    text_transform = TextTransform()

    train_datasets = [LibriSpeechDataset(config['data']['data_dir'], url)
                      for url in config['data']['urls']['train']]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=lambda x: collate_fn(x, text_transform, "train"),
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechRecognitionModel(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['train']['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['train']['epochs']
    )
    criterion = torch.nn.CTCLoss(blank=28).to(device)

    best_wer = float('inf')
    for epoch in range(config['train']['epochs']):
        model.train()
        for batch_idx, (data, targets, input_len, target_len) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data.to(device))
            loss = criterion(outputs, targets, input_len, target_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            if batch_idx % 100 == 0:
                model.log_melspectrogram(data, logger, epoch * len(train_loader) + batch_idx)
                logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0]
                }, step=epoch * len(train_loader) + batch_idx)

        val_wer = validate(model, device, config, criterion, logger, epoch)
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(model.state_dict(), f"{config['train']['save_dir']}/best_model.pth")
            logger.log_checkpoint(f"{config['train']['save_dir']}/best_model.pth")


def validate(model, device, config, criterion, logger, epoch):
    model.eval()
    text_transform = TextTransform()
    val_loss = 0.0
    val_cer, val_wer = [], []

    val_datasets = [LibriSpeechDataset(config['data']['data_dir'], url)
                    for url in config['data']['urls']['dev']]
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=lambda x: collate_fn(x, text_transform, "valid"),
        num_workers=4,
        shuffle=False
    )

    with torch.no_grad():
        for batch_idx, (data, labels, input_lengths, label_lengths) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(
                F.log_softmax(outputs, dim=2).transpose(0, 1),
                labels,
                input_lengths,
                label_lengths
            )
            val_loss += loss.item() / len(val_loader)

            decoded_preds, decoded_targets = greedy_decode(
                outputs.cpu(),
                labels.cpu(),
                label_lengths,
                text_transform
            )

            for pred, target in zip(decoded_preds, decoded_targets):
                val_cer.append(cer(target, pred))
                val_wer.append(wer(target, pred))

            if batch_idx == 0:
                logger.log_metrics({
                    "val/examples": wandb.Table(
                        columns=["Target", "Prediction", "CER", "WER"],
                        data=[
                            [t, p, cer(t, p), wer(t, p)]
                            for t, p in list(zip(decoded_targets, decoded_preds))[:5]
                        ]
                    )
                }, step=epoch)

    avg_cer = sum(val_cer) / len(val_cer)
    avg_wer = sum(val_wer) / len(val_wer)
    avg_loss = val_loss

    logger.log_metrics({
        "val/loss": avg_loss,
        "val/cer": avg_cer,
        "val/wer": avg_wer
    }, step=epoch)

    print(f'\nValidation set: Average loss: {avg_loss:.4f}, Average CER: {avg_cer:.4f}, Average WER: {avg_wer:.4f}\n')

    model.train()
    return avg_wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
