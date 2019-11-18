import argparse
import json
import torch
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from data.create_record_data import InputFeatures, RecordExample
from data import tokenization
from data.tokenization import SentencePieceTokenizer
from data.record_utils import load_record_dataset, load_record_devset, write_predictions, QADataset, RawResult
from ALBERT.train.optimizer import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import QAModel

PRINT_EVERY = 50 # BATCH

def main(args):
    with open(args.config_file, "r") as f:
        config = json.loads(f.read())
    train_config = config["train"]
    encoder_config = config["encoder"]
    decoder_config = config["decoder"]
    prediction_config = config["prediction"]
    model = QAModel(encoder_config, decoder_config)
    device = torch.device('cuda:3')
    model.to(device)
    #print(list(model.named_parameters()))
    #return
    train, eval = load_record_dataset(args.train_file, encoder_config["max_sequence_length"],
                                      train_config["train_proportion"])
    #train = None
    #eval = None
    trainloader = DataLoader(dataset=train, sampler= RandomSampler(train),
                            batch_size=train_config["batch_size"], num_workers=4)

    evalloader = DataLoader(dataset=eval, sampler=RandomSampler(eval),
                            batch_size=train_config["batch_size"], num_workers=4)

    #trainloader = None

    #evalloader = None

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': train_config["weight_decay"]},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    model.train()


    #total_steps = int(len(train) / train_config["batch_size"]) * train_config["epoch"]

    total_steps = 0

    print("total steps %d" % total_steps)

    optimizer = AdamW(optimizer_grouped_parameters, lr=train_config["lr"], eps=train_config["lr_decay"])
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps= int(total_steps * train_config["warm_up_proportion"]),
                                        t_total=total_steps)
    #optimizer = torch.optim.Adam(model.encoder.parameters())
    optimizer.zero_grad()
    start_loss_function = CrossEntropyLoss()
    end_loss_function = CrossEntropyLoss()
#======================================================================================================================

    def process_data(i, batch, is_training):
        (tokens, masks, seg_ids, start_postions, end_positions, unique_ids) = tuple(x for x in batch)
        tokens = tokens.to(device)
        masks= masks.to(device)
        seg_ids = seg_ids.to(device)
        start_postions = start_postions.to(device)
        end_positions = end_positions.to(device)
        pos_ids = torch.tensor(np.arange(tokens.shape[1]), dtype=torch.int64, device=tokens.device)
        start_pos_output, end_pos_output = model(tokens, seg_ids, pos_ids, masks)
        # print(start_pos_output, end_pos_output)
        if is_training:
            start_loss = start_loss_function(start_pos_output, start_postions)
            end_loss = end_loss_function(end_pos_output, end_positions)
            loss = (torch.mean(start_loss) + torch.mean(end_loss)) / 2.0
        else: loss = None
        return loss, start_pos_output, end_pos_output, unique_ids

    def evaluate(model: QAModel, dev_path: str, encoder_config: dict, train_config: dict):
        print("======================START EVALUATION======================")
        model.eval()
        dev, dev_features, dev_examples = load_record_devset(dev_path, encoder_config["max_sequence_length"])
        tokenizer = SentencePieceTokenizer(args.vocab_model)
        devloader = DataLoader(dataset=dev, sampler=SequentialSampler(dev),
                               batch_size=train_config["batch_size"], num_workers=4)
        unique_id_list = []
        start_list = []
        end_list = []
        with torch.no_grad():
            for i, batch in enumerate(devloader):
                #print(len(batch[0]))
                _, start, end, unique_ids = process_data(i, batch, False)
                unique_id_list.append(unique_ids)
                start_list.append(start.cpu())
                end_list.append(end.cpu())
        unique_id_list = np.concatenate(unique_id_list)
        start_list = np.concatenate(start_list)
        end_list = np.concatenate(end_list)
        all_results = []
        #print(len(unique_id_list))
        for i, unique_id in enumerate(unique_id_list):
            #print(unique_id)
            #print("%d, %d" %(i, unique_id))
            all_results.append(RawResult(
                unique_id=unique_id,
                start_logits=start_list[i],
                end_logits=end_list[i]
            ))

        write_predictions(dev_examples, dev_features, all_results, prediction_config["nbest"],
                          prediction_config["max_answer_length"], prediction_config["do_lower_case"],
                          args.prediction_file, args.nbest_file ,None, True, False, tokenizer, 0)
#======================================================================================================================
    for epoch in range(train_config["epoch"]):
        cnt = 0
        total = 0.0
        for i, batch in enumerate(trainloader):
            loss, _, _, _ = process_data(i, batch, True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            cnt += 1
            total += loss.item()
            if (i+1) % PRINT_EVERY == 0:
                print("EPOCH %d/%d, BATCH %d, LOSS: %.5f" % (epoch + 1, train_config["epoch"], i + 1, total / cnt))
        cnt = 0
        total = 0.0
        for i, batch in enumerate(evalloader):
            with torch.no_grad():
                loss, _, _, _ = process_data(i, batch, True)
                cnt += 1
                total += loss.item()
        print("VALID OF EPOCH %d LOSS IS %.5f" % (epoch, total / cnt))

    print("SAVE MODEL TO %s" % args.save_model_file)
    torch.save(model.state_dict(), args.save_model_file)
    evaluate(model, args.dev_file, encoder_config, train_config) # evaluate model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/QAconfig.json", type=str)
    parser.add_argument("--train_file", default="data/dataset/record/", type=str)
    parser.add_argument("--dev_file", default="data/dataset/record/", type=str)
    parser.add_argument("--prediction_file", default="result/prediction.json", type=str)
    parser.add_argument("--save_model_file", default="result/model.pkl")
    parser.add_argument("--nbest_file", default="result/nbest.json", type=str)
    parser.add_argument("--vocab_model", default="data/vocab.model", type=str)
    parser.add_argument("--is_training", default=True, type=bool)
    if not os.path.exists("result/"):
        os.makedirs("result/")
    args = parser.parse_args()
    main(args)
