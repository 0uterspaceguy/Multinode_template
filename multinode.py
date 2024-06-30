import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from tqdm import tqdm

import argparse
import yaml




def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)

        targets = targets.type(torch.LongTensor).cuda()



        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in tqdm(self.train_data):
            source = source.to(self.local_rank).float()
            targets = targets.type(torch.LongTensor).to(self.local_rank)

            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(config):
    train_set = Dataset(data_path=config["train_path"],
                 names=config["names"]) 
    test_set = Dataset(data_path=config["test_path"],
                 names=config["names"])
    model = Model()  
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    return train_set, test_set, model, optimizer, criterion


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=16,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, 
         total_epochs: int, 
         batch_size: int, 
         snapshot_path: str = "./snapshots/snapshot.pt",
         config_path: str = "./config.yaml"):
    
    ddp_setup()

    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    train_set, test_set, model, optimizer, criterion = load_train_objs(config)

    train_data = prepare_dataloader(train_set, batch_size)
    test_data = prepare_dataloader(test_set, batch_size)

    trainer = Trainer(
        model=model,
        train_data=train_data,
        test_data=test_data,
        optimizer=optimizer,
        criterion=criterion,
        save_every=save_every,
        snapshot_path=snapshot_path)
    
    trainer.train(total_epochs)
    destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('config', type=str, help='Path to config')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.save_every, args.total_epochs, args.batch_size)