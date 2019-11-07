import argparse
import json
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from data.create_record_data import InputFeatures
from data.record_utils import load_record_dataset, QADataset
from ALBERT.train.optimizer import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler
from model import QAModel


def main(args):
    with open(args.config_file, "r") as f:
        config = json.loads(f.read())
    train_config = config["train"]
    encoder_config = config["encoder"]
    decoder_config = config["decoder"]
    model = QAModel(encoder_config, decoder_config)
    device = torch.device('cuda:3')
    model.to(device)
    #print(list(model.named_parameters()))
    #return
    dataset = load_record_dataset(args.train_file, encoder_config["max_sequence_length"])
    dataloader = DataLoader(dataset=dataset, sampler= RandomSampler(dataset),
                            batch_size=train_config["batch_size"], num_workers=4)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': train_config["weight_decay"]},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    total_steps = int(len(dataset) / train_config["batch_size"]) * train_config["epoch"]

    print("total steps %d" % total_steps)

    optimizer = AdamW(optimizer_grouped_parameters, lr=train_config["lr"], eps=train_config["lr_decay"])
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps= int(total_steps * train_config["warm_up_proportion"]),
                                        t_total=total_steps)
    #optimizer = torch.optim.Adam(model.encoder.parameters())
    optimizer.zero_grad()
    model.train()
    start_loss_function = CrossEntropyLoss()
    end_loss_function = CrossEntropyLoss()

    for epoch in range(train_config["epoch"]):
        for i, batch in enumerate(dataloader):
            (tokens, masks, seg_ids, start_postions, end_positions) = tuple(x.to(device) for x in batch)
            pos_ids = torch.tensor(np.arange(tokens.shape[1]), dtype=torch.int64, device=tokens.device)
            start_pos_output, end_pos_output = model(tokens, seg_ids, pos_ids, masks)
            #print(start_pos_output, end_pos_output)
            start_loss = start_loss_function(start_pos_output, start_postions)
            end_loss = end_loss_function(end_pos_output, end_positions)
            loss = (torch.mean(start_loss) + torch.mean(end_loss)) / 2.0
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(str(i) +":"+ str(loss.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/QAconfig.json", type=str)
    parser.add_argument("--train_file", default="data/dataset/record/train.pkl", type=str)
    parser.add_argument("--dev_file", default="data/dataset/record/dev.pkl", type=str)
    parser.add_argument("--is_training", default=True, type=bool)
    args = parser.parse_args()
    main(args)
