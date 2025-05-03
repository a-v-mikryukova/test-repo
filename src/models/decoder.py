import torch
import wandb

def greedy_decode(output, labels, label_lengths, text_transform, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []

    example_pred = []
    example_target = []

    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())

        target_indices = labels[i][:label_lengths[i]].tolist()
        target_text = text_transform.int_to_text(target_indices)
        decoded_text = text_transform.int_to_text(decode)

        decodes.append(decoded_text)
        targets.append(target_text)

        if i == 0:
            example_pred = decode
            example_target = target_indices

    if wandb.run:
        try:
            pred_chars = [text_transform.index_map[idx] for idx in example_pred]
            target_chars = [text_transform.index_map[idx] for idx in example_target]

            table = wandb.Table(columns=["Type", "Text", "Indices", "Characters"])
            table.add_data("Prediction", " ".join(decoded_text.split()), example_pred, pred_chars)
            table.add_data("Target", " ".join(target_text.split()), example_target, target_chars)

            wandb.log({
                "predictions": table,
                "target_text": target_text,
                "predicted_text": decoded_text
            })
        except Exception as e:
            print(f"Failed to log predictions: {e}")

    return decodes, targets
