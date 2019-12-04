import argparse
import json
import torch
import os
import numpy as np
from data.tokenization import FullTokenizer
from record_utils import load_record_devset, write_predictions, RawResult, InputFeatures, RecordExample
from torch.utils.data import DataLoader, SequentialSampler
from model import QAModel

def main(args):
    with open(args.config_file, "r") as f:
        config = json.loads(f.read())
    encoder_model_path = config["encoder"]["pretrain_model_path"]
    encoder_config_path = config["encoder"]["config_path"]
    decoder_config = config["decoder"]
    prediction_config = config["prediction"]
    with open(encoder_config_path, "r") as f:
        encoder_config = json.loads(f.read())
    model = QAModel(encoder_model_path, encoder_config, decoder_config, False)
    device = torch.device('cuda:1')
    model.to(device)
    model.load_state_dict(torch.load(args.save_model_file))

# ---------------------------------------------------------------------------------------------------------------------
    def process_data(i, batch):
        (tokens, masks, seg_ids, start_postions, end_positions, ner_label, unique_ids) = tuple(x for x in batch)
        tokens = tokens.to(device)
        masks = masks.to(device)
        seg_ids = seg_ids.to(device)
        pos_ids = torch.tensor(np.arange(tokens.shape[1]), dtype=torch.int64, device=tokens.device)
        start_pos_output, end_pos_output, _ = model(tokens, seg_ids, pos_ids, masks)
        # start_pos_output -> [Batch, SeqLen]
        # end_pos_output -> [Batch, SeqLen]
        # print(start_pos_output, end_pos_output)
        qa_loss = None
        ner_loss = None
        return qa_loss, ner_loss, start_pos_output, end_pos_output, unique_ids
# ---------------------------------------------------------------------------------------------------------------------

    print("======================START EVALUATION======================")
    model.eval()
    dev, dev_features, dev_examples = load_record_devset(args.dev_file, encoder_config["max_position_embeddings"])
    tokenizer = FullTokenizer(vocab_file=None, spm_model_file=args.vocab_model)
    devloader = DataLoader(dataset=dev, sampler=SequentialSampler(dev),
                           batch_size=prediction_config["evaluate_batch_size"])
    unique_id_list = []
    start_list = []
    end_list = []
    with torch.no_grad():
        for i, batch in enumerate(devloader):
            # print(len(batch[0]))
            _, _, start, end, unique_ids = process_data(i, batch)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config/QAconfig.json", type=str)
    parser.add_argument("--dev_file", default="data/dataset/record/", type=str)
    parser.add_argument("--prediction_file", default="result/prediction.json", type=str)
    parser.add_argument("--save_model_file", default="result/model.pkl")
    parser.add_argument("--nbest_file", default="result/nbest.json", type=str)
    parser.add_argument("--vocab_model", default="ALBERT/pretrained_model/albert-base-v2-spiece.model",
                        type=str)
    if not os.path.exists("result/"):
        os.makedirs("result/")
    args = parser.parse_args()
    main(args)