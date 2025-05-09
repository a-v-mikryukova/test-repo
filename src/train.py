import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from src.data import LibriSpeechDataset, TextTransform, collate_fn
from src.models import SpeechRecognitionModel, greedy_decode
from src.utils import WanDBLogger, cer, wer

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path="../configs", config_name="config")
def train(config) -> None:
    logger = WanDBLogger(dict(config))
    text_transform = TextTransform()

    train_datasets = [LibriSpeechDataset(config["data"]["data_dir"], url)
                      for url in config["data"]["urls"]["train"]]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=lambda x: collate_fn(x, text_transform, "train")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechRecognitionModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["train"]["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=config["train"]["epochs"],
        anneal_strategy='linear')
    criterion = torch.nn.CTCLoss(blank=28).to(device)

    best_wer = float("inf")
    for epoch in range(config["train"]["epochs"]):
        model.train()
        data_len = len(train_loader.dataset)
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            optimizer.zero_grad()

            output = model(spectrograms)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)
            
            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()
            logger.log_metrics({
                    "train/loss": loss,
                    "train/lr": scheduler.get_last_lr()[0],
                })
            optimizer.step()
            scheduler.step()
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

    val_datasets = [LibriSpeechDataset(config["data"]["data_dir"], url)
                    for url in config["data"]["urls"]["dev"]]
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, text_transform, "valid")
    )

    with torch.no_grad():
        for i, _data in enumerate(val_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            output = model(spectrograms)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)
            loss = criterion(output, labels, input_lengths, label_lengths)
            val_loss += loss.item() / len(val_loader)

            decoded_preds, decoded_targets = greedy_decode(output.transpose(0, 1), labels, label_lengths, text_transform)

            for j in range(len(decoded_preds)):
                val_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                val_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(val_cer) / len(val_cer)
    avg_wer = sum(val_wer) / len(val_wer)
    avg_loss = val_loss

    logger.log_metrics({
        "val/loss": val_loss,
        "val/cer": avg_cer,
        "val/wer": avg_wer,
    })

    return avg_wer


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="configs/config.yaml")
    # args = parser.parse_args()
    train()
