from password_dataset import *
from seq2seq_model import *
from tokenizer import *
import numpy as np
from tqdm import tqdm
import torch

DEFAULT_DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class Pass2PathTrainer:
    def __init__(self, 
                 t1=KBDPasswordTokenizer(), 
                 t2=TransTokenizer(), 
                 epoch=50, 
                 batch_size=4, 
                 hidden_size=128, 
                 embed_size=200, 
                 max_len=16,
                 num_layers=1, 
                 lr=0.01, 
                 dropout=0.3, 
                 device=DEFAULT_DEVICE):
        self.t1 = t1
        self.t2 = t2
        self.pad_token = self.t2.pad_token_id
        self.model = Pass2PathModel(
            len(self.t1), 
            len(self.t2), 
            pad_token=self.t2.pad_token_id, 
            start_token=self.t2.start_token_id, 
            end_token=self.t2.end_token_id,
            hidden_size=hidden_size, 
            embed_size=embed_size,
            maxlen=max_len,
            num_layers=num_layers,
            dropout=dropout, 
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
        dataset = pass2path_dataloader(data_path, t1 = self.t1, t2 = self.t2, batch_size=self.batch_size)
        losses = np.full(self.epoch, np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        criterion = nn.NLLLoss(ignore_index = self.pad_token)

        self.model.train()
        for iter in range(self.epoch):
            batch_loss = []
            with tqdm(dataset, desc="Training", leave=False) as dataset_wrapper:
                for pwds, edits in dataset_wrapper:
                    pwds = pwds.to(self.device)
                    edits = edits.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(pwds, edits)
                    batch_size = edits.size(0)
                    loss = criterion(outputs[:, :].reshape(-1, len(self.t2)), edits[:, :].reshape(-1))
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item() / batch_size)
                    avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
                    dataset_wrapper.set_postfix(loss=avg_loss)
            avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
            print(f">>> Epoch: {iter}, Loss: {avg_loss}")
            losses[iter] = avg_loss
        self.save(model_save)
        return losses
 
    def save(self, model_path):
        torch.save(self.model, model_path)
        print(f">>> Model saved in {model_path}")
        pass

def main():
    model_save = "/disk/yjt/PersonalTarGuess/model/pass2path/t_collection_4kw_e_cos_m_pass2path_v4.pt"
    dataset = "/disk/data/targuess/2_train/pitg/Collection1_cos_4kw.csv"
    trainer = Pass2PathTrainer(
        epoch=5, 
        batch_size=64, 
        hidden_size=128, 
        embed_size=64, 
        num_layers=3,
        lr=0.001, 
        dropout=0.4, 
        device=DEFAULT_DEVICE)
    trainer.train(dataset, model_save)
    pass

if __name__ == '__main__':
    main()