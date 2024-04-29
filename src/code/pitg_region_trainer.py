from password_dataset import *
from pitg_model import *
from tokenizer import *
import numpy as np
from tqdm import tqdm
import torch

DEFAULT_DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# DEFAULT_DEVICE = torch.device('cpu')

class Pass2PathTrainer:
    def __init__(
            self, 
            t1=KBDPasswordTokenizer(), 
            t2=TransTokenizer(), 
            t3=RegionTokenizer(), 
            t4=HostTokenizer(),
            epoch=50, 
            batch_size=4, 
            hidden_size=128, 
            embed_size=200, 
            num_layers=3,
            max_len=16, 
            dropout=0.4, 
            lr=0.001, 
            ratio="1:1",
            device=DEFAULT_DEVICE):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3 
        self.t4 = t4
        self.pad_token = self.t2.pad_token_id
        self.model = PITGModel(
            len(self.t1), 
            len(self.t3), 
            len(self.t4), 
            len(self.t2), 
            pad_token=self.t2.pad_token_id, 
            start_token=self.t2.start_token_id, 
            end_token=self.t2.end_token_id,
            hidden_size=hidden_size, 
            embed_size=embed_size, 
            num_layers=num_layers, 
            maxlen=max_len, 
            dropout=dropout,
            ratio=ratio, 
            device=device)
        self.max_len = max_len
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.model.set_mode("train")
        self.device = device
        self.model = self.model.to(self.device)
        pass

    def train(self, data_path, model_save):
        dataset = pitg_region_dataloader(
            data_path, 
            t1 = self.t1, 
            t2 = self.t2, 
            t3 = self.t3, 
            t4 = self.t4, 
            batch_size=self.batch_size)
        losses = np.full(self.epoch, np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        criterion = nn.NLLLoss(ignore_index = self.pad_token)

        self.model.train()
        for iter in range(self.epoch):
            batch_loss = []
            with tqdm(dataset, desc="Training", leave=False) as dataset_wrapper:
                for pwds, regions, hosts, edits in dataset_wrapper:
                    pwds = pwds.to(self.device)
                    edits = edits.to(self.device)
                    regions = regions.to(self.device) 
                    hosts = hosts.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(pwds, regions, hosts, edits)
                    batch_size = edits.size(0)
                    loss = criterion(outputs.reshape(-1, len(self.t2)), edits.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item() / batch_size)
                    avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
                    dataset_wrapper.set_postfix(loss=avg_loss)
            avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
            print(f">>> Epoch: {iter}, Loss: {avg_loss}")
            losses[iter] = avg_loss
        self.save(model_path=model_save)
        return losses
 
    def save(self, model_path):
        torch.save(self.model, model_path)
        print(f">>> Model saved in {model_path}")
        pass

def main():
    model_save = "/disk/yjt/PersonalTarGuess/model/pitg/t_collection_4kw_e_cos_m_pitg_region_v3.pt"
    dataset = "/disk/data/targuess/2_train/pitg/Collection1_cos_4kw.csv"

    trainer = Pass2PathTrainer(
        epoch=5, 
        batch_size=64, 
        hidden_size=128, 
        embed_size=200, 
        num_layers=3,
        max_len=16, 
        dropout=0.4, 
        lr=0.001, 
        ratio="1:1",
        device=DEFAULT_DEVICE
        )
    trainer.train(dataset, model_save)
    pass

if __name__ == '__main__':
    main()