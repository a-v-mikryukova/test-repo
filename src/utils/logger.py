import wandb


class WanDBLogger:
    def __init__(self, config):
        self.config = config
        self.run = wandb.init(
            project="asr",
            config=config,
            magic=True
        )

    def log_metrics(self, metrics, step):
        wandb.log(metrics, step=step)

    def watch_model(self, model):
        wandb.watch(model, log="all", log_freq=100)

    def log_checkpoint(self, path):
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(path)
        self.run.log_artifact(artifact)