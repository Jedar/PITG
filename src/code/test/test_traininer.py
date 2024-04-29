import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from password_dataset import *
from seq2seq_model import *
from tokenizer import *
from pitg_trainer import *
import numpy as np
from tqdm import tqdm
import torch

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def predict(model, inputs, regions, hosts):
    input_seq = inputs
    batch_size = input_seq.size(0)
    target_len = 6
    with torch.no_grad():
        encoder_hidden = model.encoder(input_seq, regions, hosts)
        decoder_input = torch.tensor([[1]] * batch_size).to(DEFAULT_DEVICE)  # 设置起始符号的索引为 0
        decoder_hidden = encoder_hidden
        outputs = torch.zeros(batch_size, target_len, model.output_size)
        for t in range(0, target_len):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output.squeeze(1)
            decoder_input = decoder_output.argmax(dim=2)
    predicted_seq = outputs.argmax(dim=2)
    print("Predicted Sequence:", predicted_seq.squeeze().tolist())

def predict_topk(model, inputs, topk, start_token=1):
    input_seq = inputs
    batch_size = input_seq.size(0)
    target_len = 6
    with torch.no_grad():
        decoder_hidden = model.encoder(input_seq)
        decoder_input = torch.tensor([[start_token]] * batch_size)  # 初始解码器输入为起始标记

        predicted_seq = []
        for _ in range(target_len):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.argmax(dim=2)

            # 应用Top-k采样策略
            topk_values, topk_indices = torch.topk(decoder_output.squeeze(), topk)

            predicted_seq.append(topk_indices.tolist())
    predicted_seq = torch.tensor(predicted_seq).transpose(0,1)
    print("Input Sequence:", inputs)
    print("Predicted Sequence:", predicted_seq)

def main():
    model_save = "/disk/yjt/PersonalTarGuess/model/pass2path/t_collection_100_m_pass2path.pt"
    dataset = "/disk/data/targuess/2_train/pitg/Collection1_cos_100_test.csv"

    trainer = PITGTrainer(batch_size=2, device=DEFAULT_DEVICE)
    trainer.train(dataset)

    # trainer.model = torch.load(model_save)
    pwds = [
        ("hello1", "lilcoach12345@yahoo.com"), 
        ("funtik44", "elena44.114@yandex.ru"),
        ("jebstone", "del7734@yahoo.com"),
        ("lerev1231", "verelius@gmail.com"), 
        ("a12345", "travel.m@hotmail.com"),
        ("hello1", ""), 
        ("funtik44", ""),
        ("jebstone", ""),
        ("lerev1231", ""), 
        ("a12345", "")
    ]

    test = trainer.t1([x[0] for x in pwds]).to(DEFAULT_DEVICE)
    emails = [parse_email(x[1]) for x in pwds]
    regions = torch.tensor([trainer.t3.encode(x[2]) for x in emails]).to(DEFAULT_DEVICE)
    hosts = torch.tensor([trainer.t4.encode(x[1]) for x in emails]).to(DEFAULT_DEVICE)

    print("Start: ",trainer.t2.start_token_id)
    # predict_topk(trainer.model, test, 5, trainer.t2.start_token_id)
    predict(trainer.model, test, regions, hosts)

    # trainer.save(model_save)
    pass

main()