import argparse
import json
import torch
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from data.tokenization import FullTokenizer
from data.knowlege_graph_utils import buid_graph
from record_utils import load_relation_dataset, load_relation_prediction_dataset
from create_relation_pretrain import RelationExample
from ALBERT.model.optimization import LAMB, get_linear_schedule_with_warmup
from ALBERT.model import AlbertModel, AlbertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

PRINT_EVERY = 50  # BATCH

class CLSModel(torch.nn.Module):
    def __init__(self, encoder_model_path, config , num_class = 2):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained(encoder_model_path, config=config)
        self.drop = torch.nn.Dropout(0.1)
        self.decoder = torch.nn.Linear(768, num_class)
        self.decoder.weight.data.normal_(mean=0.0, std=0.02)
        self.decoder.bias.data.zero_()

    def forward(self, token, mask, seg, pos):
        out = self.encoder(token, mask, seg, pos)[0]
        out = self.drop(out)
        return self.decoder(out)

def main(args):
    with open(args.config_file, "r") as f:
        config = json.loads(f.read())
    train_config = config["train"]
    encoder_model_path = config["encoder"]["pretrain_model_path"]
    encoder_config_path = config["encoder"]["config_path"]
    decoder_config = config["decoder"]
    prediction_config = config["prediction"]
    with open(encoder_config_path, "r") as f:
        encoder_config = json.loads(f.read())
    device = torch.device('cuda:1')
    config = AlbertConfig(**encoder_config)
    #
    # input_token = torch.LongTensor(np.random.randint(0, 30000, (4, 512)))
    # input_segment = torch.LongTensor(np.random.randint(0, 1, (4, 512)))
    # input_pos = torch.LongTensor(np.repeat(np.expand_dims(np.arange(512), 0), 4, axis=0))
    # input_mask = torch.LongTensor(np.random.randint(0, 1, (4, 512)))
    # divide = np.random.randint(100,200, (4,))
    # res = model(input_token, input_segment, input_pos, input_mask, divide)
    # print(res[0], res[1])
    # exit(0)
    # print(list(model.named_parameters()))
    # return
    if args.train_type == "is_related":
        train, eval = load_relation_dataset()
        model = CLSModel(encoder_model_path, config=config)
    else:
        train, eval = load_relation_prediction_dataset()
        model = CLSModel(encoder_model_path, config=config, num_class=46)

    model.to(device)
    # train = None
    # eval = None
    trainloader = DataLoader(dataset=train, sampler=RandomSampler(train),
                             batch_size=train_config["batch_size"], num_workers=4)

    evalloader = DataLoader(dataset=eval, sampler=SequentialSampler(eval),
                            batch_size=train_config["batch_size"], num_workers=4)

    # trainloader = None

    # evalloader = None

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': train_config["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    model.train()

    total_steps = int(len(train) / train_config["batch_size"]) * train_config["epoch"]

    # total_steps = 0

    print("total steps %d" % total_steps)

    optimizer = LAMB(optimizer_grouped_parameters, lr=train_config["lr"])
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        total_steps * train_config["warm_up_proportion"]),
                                                   num_training_steps=total_steps)
    # optimizer = torch.optim.Adam(model.encoder.parameters())
    optimizer.zero_grad()
    relation_loss_function = CrossEntropyLoss(ignore_index=-1)

    # ======================================================================================================================

    def process_data(i, batch, is_training):
        (tokens, labels, masks) = tuple(x for x in batch)
        tokens = tokens.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        seg_ids = torch.zeros_like(masks, device=device)
        pos_ids = torch.tensor(np.arange(tokens.shape[1]), dtype=torch.int64, device=tokens.device)
        output = model(tokens, masks, seg_ids, pos_ids)
        # start_pos_output -> [Batch, SeqLen]
        # end_pos_output -> [Batch, SeqLen]
        # ner_output -> [Batch, SeqLen, 3]
        # print(start_pos_output, end_pos_output)
        if is_training:
            #print(output.shape, labels.shape)
            output = torch.reshape(output, (output.shape[0] * output.shape[1], -1))
            labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1],))
            relation_loss = relation_loss_function(output, labels)
        else:
            relation_loss = None
        return relation_loss, output

    # ======================================================================================================================
    for epoch in range(train_config["epoch"]):
        cnt = 0
        total = 0.0
        for i, batch in enumerate(trainloader):
            loss, out = process_data(i, batch, True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            cnt += 1
            total += loss.item()
            if (i + 1) % PRINT_EVERY == 0:
                print(
                    "EPOCH %d/%d, BATCH %d, BATCH_LOSS: %.5f, TOTAL_AVERAGE_LOSS: %.5f" %
                    (epoch + 1, train_config["epoch"], i + 1, loss.item(), total / cnt))
        cnt = 0
        total = 0.0
        for i, batch in enumerate(evalloader):
            with torch.no_grad():
                loss, _ = process_data(i, batch, True)
                cnt += 1
                total += loss.item()
        print("VALID OF EPOCH %d LOSS IS %.5f" % (epoch, total / cnt))
        torch.save(model.encoder.state_dict(), "result/model_relation_prediction_pretrained-eph%d.pkl" % epoch)

    print("SAVE MODEL TO %s" % args.save_model_file)
    torch.save(model.encoder.state_dict(), args.save_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/RelationPrediction.json", type=str)
    parser.add_argument("--save_model_file", default="result/model_relation_prediction_pretrained.pkl")
    parser.add_argument("--vocab_model", default="ALBERT/pretrained_model/albert-base-v2-spiece.model",
                        type=str)
    parser.add_argument("--train_type", default="is_related", type=str)
    parser.add_argument("--is_training", default=True, type=bool)
    if not os.path.exists("result/"):
        os.makedirs("result/")
    args = parser.parse_args()
    main(args)
