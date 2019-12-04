import argparse
import json
import torch
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from data.tokenization import FullTokenizer
from record_utils import load_record_dataset, load_record_devset, write_predictions, RawResult, InputFeatures
from ALBERT.model.optimization import LAMB, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import QAModel

PRINT_EVERY = 50  # BATCH


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
    model = QAModel(encoder_model_path, encoder_config, decoder_config)
    device = torch.device('cuda:1')
    model.to(device)

    # print(list(model.named_parameters()))
    # return
    train, eval = load_record_dataset(args.train_file, encoder_config["max_position_embeddings"],
                                      train_config["train_proportion"])
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
    start_loss_function = CrossEntropyLoss()
    end_loss_function = CrossEntropyLoss()
    ner_loss_function = CrossEntropyLoss(ignore_index=-1)

    # ======================================================================================================================

    def process_data(i, batch, is_training):
        (tokens, masks, seg_ids, start_postions, end_positions, ner_label, unique_ids) = tuple(x for x in batch)
        tokens = tokens.to(device)
        masks = masks.to(device)
        seg_ids = seg_ids.to(device)
        start_postions = start_postions.to(device)
        end_positions = end_positions.to(device)
        ner_label = ner_label.to(device)
        pos_ids = torch.tensor(np.arange(tokens.shape[1]), dtype=torch.int64, device=tokens.device)
        start_pos_output, end_pos_output, ner_output = model(tokens, seg_ids, pos_ids, masks)
        # start_pos_output -> [Batch, SeqLen]
        # end_pos_output -> [Batch, SeqLen]
        # ner_output -> [Batch, SeqLen, 3]
        # print(start_pos_output, end_pos_output)
        ner_output = torch.reshape(ner_output, (ner_output.shape[0] * ner_output.shape[1], -1))
        ner_label = torch.reshape(ner_label, (ner_label.shape[0] * ner_label.shape[1],))
        if is_training:
            start_loss = start_loss_function(start_pos_output, start_postions)
            end_loss = end_loss_function(end_pos_output, end_positions)
            ner_loss = ner_loss_function(ner_output, ner_label)
            qa_loss = (torch.mean(start_loss) + torch.mean(end_loss)) / 2.0
        else:
            qa_loss = None
            ner_loss = None
        return qa_loss, ner_loss, start_pos_output, end_pos_output, unique_ids

    def evaluate(model: QAModel, dev_path: str, encoder_config: dict, prediction_config: dict):
        print("======================START EVALUATION======================")
        model.eval()
        dev, dev_features, dev_examples = load_record_devset(dev_path, encoder_config["max_position_embeddings"])
        tokenizer = FullTokenizer(vocab_file=None, spm_model_file=args.vocab_model)
        devloader = DataLoader(dataset=dev, sampler=SequentialSampler(dev),
                               batch_size=prediction_config["evaluate_batch_size"])
        unique_id_list = []
        start_list = []
        end_list = []
        with torch.no_grad():
            for i, batch in enumerate(devloader):
                # print(len(batch[0]))
                _, _, start, end, unique_ids = process_data(i, batch, False)
                unique_id_list.append(unique_ids)
                start_list.append(start.cpu())
                end_list.append(end.cpu())
        unique_id_list = np.concatenate(unique_id_list)
        start_list = np.concatenate(start_list)
        end_list = np.concatenate(end_list)
        all_results = []
        # print(len(unique_id_list))
        for i, unique_id in enumerate(unique_id_list):
            # print(unique_id)
            # print("%d, %d" %(i, unique_id))
            all_results.append(RawResult(
                unique_id=unique_id,
                start_logits=start_list[i],
                end_logits=end_list[i]
            ))

        write_predictions(dev_examples, dev_features, all_results, prediction_config["nbest"],
                          prediction_config["max_answer_length"], prediction_config["do_lower_case"],
                          args.prediction_file, args.nbest_file, None, True, False, tokenizer, 0)

    # ======================================================================================================================
    for epoch in range(train_config["epoch"]):
        cnt = 0
        total = 0.0
        for i, batch in enumerate(trainloader):
            qa_loss, ner_loss, _, _, _ = process_data(i, batch, True)
            loss = (qa_loss + ner_loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            cnt += 1
            total += loss.item()
            if (i + 1) % PRINT_EVERY == 0:
                print(
                    "EPOCH %d/%d, BATCH %d, QA_LOSS: %.5f, NER_LOSS: %.5f, AVERAGE_LOSS: %.5f, TOTAL_AVERAGE_LOSS: %.5f" %
                    (
                    epoch + 1, train_config["epoch"], i + 1, qa_loss.item(), ner_loss.item(), loss.item(), total / cnt))
        cnt = 0
        total = 0.0
        for i, batch in enumerate(evalloader):
            with torch.no_grad():
                qa_loss, ner_loss, _, _, _ = process_data(i, batch, True)
                cnt += 1
                total += qa_loss.item()
        print("VALID OF EPOCH %d LOSS IS %.5f" % (epoch, total / cnt))

    print("SAVE MODEL TO %s" % args.save_model_file)
    torch.save(model.state_dict(), args.save_model_file)
    evaluate(model, args.dev_file, encoder_config, prediction_config)  # evaluate model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/QAconfig.json", type=str)
    parser.add_argument("--train_file", default="data/dataset/record/", type=str)
    parser.add_argument("--dev_file", default="data/dataset/record/", type=str)
    parser.add_argument("--prediction_file", default="result/prediction.json", type=str)
    parser.add_argument("--save_model_file", default="result/model.pkl")
    parser.add_argument("--nbest_file", default="result/nbest.json", type=str)
    parser.add_argument("--vocab_model", default="ALBERT/pretrained_model/albert-base-v2-spiece.model",
                        type=str)
    parser.add_argument("--is_training", default=True, type=bool)
    if not os.path.exists("result/"):
        os.makedirs("result/")
    args = parser.parse_args()
    main(args)
