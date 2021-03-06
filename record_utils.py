#taken from https://github.com/huggingface/transformers/blob/master/examples/utils_squad.py

import torch
import pickle
import collections
import json
import random
import numpy as np
from typing import List
from data.knowlege_graph_utils import *
from create_record_data import InputFeatures, RecordExample
from torch.utils.data.dataset import Dataset
from data.tokenization import BasicTokenizer

TRAIN = "train"
DEV = "dev"
FEATURE = "_feature"
EXAMPLE = "_example"
PKL = ".pkl"

class RelationDataset(Dataset):
    def __init__(self, example_list:List, seq_len:int):
        self.length = len(example_list)
        self.seq_len = seq_len
        self.data = example_list

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        example = self.data[item]
        token = example.tokens
        label = example.label
        mask = np.ones(len(token))
        itoken = np.zeros((self.seq_len,), dtype=np.int32)
        ilabel = np.ones((self.seq_len,), dtype=np.int32) * -1
        itoken[:len(token)] = token
        ilabel[:len(label)] = label
        imask = np.zeros((self.seq_len,), dtype=np.int32)
        imask[:len(mask)] = 1
        return (torch.tensor(itoken, dtype=torch.int64),
                torch.tensor(ilabel, dtype=torch.int64),
                torch.tensor(imask, dtype=torch.int64))

class QADataset(Dataset):
    def __init__(self, feature_list:List[InputFeatures], seq_len:int, is_training=True):
        self.length = len(feature_list)
        self.tokens = np.zeros((self.length, seq_len), dtype=np.int32)
        self.masks =  np.zeros((self.length, seq_len), dtype=np.bool)
        self.seg_ids = np.zeros((self.length, seq_len), dtype=np.bool)
        self.start_positions = np.zeros((self.length,), dtype=np.int32)
        self.end_positions = np.zeros((self.length, ), dtype=np.int32)
        self.ner_label = np.ones((self.length, seq_len), dtype=np.int32) * -1
        self.unique_id = np.zeros((self.length, ), dtype=np.int32)
        self.divide_pos = np.zeros((self.length,), dtype=np.int32)
        self.entity_start = []
        self.entity_end = []
        self.entity_node_index = []

        print("loading %d features" % self.length)
        print(len(self.ner_label))
        for i, feature in enumerate(feature_list):
            print("\rloading %d\t" % i, end='')
            self.tokens[i] = feature.input_ids
            self.masks[i] = feature.input_mask
            self.seg_ids[i] = feature.segment_ids
            #self.entity_start.append(feature.query_start_list)
            #self.entity_end.append(feature.query_end_list)
            #self.entity_node_index.append(feature.entity_node_index)

            for j, id in enumerate(self.seg_ids[i]):
                if id != 0:
                    self.divide_pos[i] = j
                    break
            if feature.start_position != None:
                self.start_positions[i] = feature.start_position
            else:
                self.start_positions[i] = -1

            if feature.end_position != None:
                self.end_positions[i] = feature.end_position
            else:
                self.end_positions[i] = -1

            if is_training:
                for j, qs in enumerate(feature.query_start_list):
                    #print(str(feature.doc_span_index)+"," + str(qs))
                    self.ner_label[i][qs] = 2
                    for k in range(qs+1, feature.query_end_list[j]):
                        self.ner_label[i][k] = 1
                    self.ner_label[i][feature.query_end_list[j]] = 3

                for j in range(np.sum(feature.input_mask)):
                    if self.ner_label[i][j] == -1:
                        self.ner_label[i][j] = 0

            self.unique_id[i] = int(feature.unique_id)

        print()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return (torch.tensor(self.tokens[item],dtype=torch.int64),
                torch.tensor(self.masks[item],dtype=torch.float32),
                torch.tensor(self.seg_ids[item], dtype=torch.int64),
                torch.tensor(self.start_positions[item], dtype=torch.int64),
                torch.tensor(self.end_positions[item], dtype=torch.int64),
                torch.tensor(self.ner_label[item], dtype=torch.int64),
                self.unique_id[item],self.divide_pos[item],
                self.entity_start, self.entity_end, self.entity_node_index)

class RawResult:
    def __init__(self, unique_id:int, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits

def load_relation_dataset(path:str='data/relation_data.pkl', max_len:int = 512 ,train_prop:float = 0.9):
    with open(path, "rb") as f:
        l = pickle.load(f)
        random.shuffle(l)
        dlen = int(len(l) * train_prop)
    return RelationDataset(l[:dlen], max_len), RelationDataset(l[dlen:], max_len)

def load_relation_prediction_dataset(path:str='data/relation_prediction_data.pkl', max_len:int = 512 ,train_prop:float = 0.9):
    with open(path, "rb") as f:
        l = pickle.load(f)
        random.shuffle(l)
        dlen = int(len(l) * train_prop)
    return RelationDataset(l[:dlen], max_len), RelationDataset(l[dlen:], max_len)

def load_record_dataset(path:str, max_len:int, train_prop:float) -> (QADataset,QADataset):
    with open(path + TRAIN + FEATURE + PKL, "rb") as f:
        feature_list = pickle.load(f)
    dlen = int(len(feature_list) * train_prop)
    return QADataset(feature_list[:dlen], max_len, True), QADataset(feature_list[dlen:], max_len)
    #return QADataset(feature_list[:100], max_len, True), QADataset(feature_list[:100], max_len)


def load_record_devset(path:str, max_len:int) -> (QADataset, List[InputFeatures], List[RecordExample]):
    with open(path + DEV + FEATURE + PKL, "rb") as f:
        feature_list = pickle.load(f)
    with open(path + DEV + EXAMPLE + PKL, "rb") as f:
        example_list = pickle.load(f)
    print(len(feature_list))
    print(len(example_list))
    #input("========================WAITING=========================")
    return QADataset(feature_list, max_len, True), feature_list, example_list


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes



def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        illegal_set = {" ", "\xc2\xa0", '\xa0', '\xad', '\u200e', '\u202c'}
        for (i, c) in enumerate(text):
            if c in illegal_set:
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            print(
                "Unable to find text: '%s' in '%s'" % (pred_text, tok_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print("Length not equal after stripping spaces: '%s' vs '%s'" % (orig_ns_text, tok_ns_text))
            #print("SUCK %s, %s" % (orig_text, tok_text))
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print(""
                  "Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = np.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative,tokenizer , null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        #print(result.unique_id)
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                #print("tok_tokens: " + str(tok_tokens))
                #print("origin_tokens: " + str(orig_tokens))
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
                tmp_token = BasicTokenizer(do_lower_case)
                tok_tokens = tmp_token.tokenize(tok_text)
                tok_text = ' '.join(tok_tokens).strip()

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = float(probs[i])
            output["start_logit"] = float(entry.start_logit)
            output["end_logit"] = float(entry.end_logit)
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions