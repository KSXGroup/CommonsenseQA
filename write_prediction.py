import torch
from data import create_record_data
from data import tokenization
import numpy as np
from data.record_utils import QADataset
import argparse
import json


def main(args):
    passage_text = input('input passage>')
    question_text = input('input question>')
    id = 0

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    prev_is_whitespace = True
    for c in passage_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    example = create_record_data.RecordExample(id,question_text,doc_tokens)

    tokenizer = tokenization.SentencePieceTokenizer(args.vocab_model)
    feature = create_record_data.convert_examples_to_feature(list(example),tokenizer,512,512,256,args.is_training)

    with open(args.config_file, "r") as f:
        config = json.loads(f.read())
    train_config = config["train"]
    encoder_config = config["encoder"]
    data = QADataset(feature,encoder_config["max_sequence_length"])

    model = torch.load(args.model_path)
    pos_ids = torch.tensor(np.arange(data.tokens.shape[1]), dtype=torch.int64, device=data.tokens.device)
    start_pos_output, end_pos_output = model.forward(data.tokens, data.seg_ids, pos_ids, data.masks)

    print('answer is:%s'%(passage_text[start_pos_output,end_pos_output]))


if __name__ == '__main__':
    parser=argparse.ArgumentParser
    parser.add_argument("--config_file", default="config/QAconfig.json", type=str)
    parser.add_argument("--is_training", default=False, type=bool)
    parser.add_argument("--model_path", default='model.pkl',type=str)
    parser.add_argument("--vocab_model",default="data/vocab.model",type=str)
    args = parser.parse_args()
    main(args)
