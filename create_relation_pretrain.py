import json
import collections
import numpy as np
import pickle
from data.tokenization import FullTokenizer, BasicTokenizer
from typing import List

REPEAT = 2

relation_map = {'LocatedNear': 'located near', 'InstanceOf': 'instance of', 'DerivedFrom': 'derived from',
                'genre':'genre', 'HasA':'has a', 'DistinctFrom':'distinct from', 'Causes':'causes',
                'HasLastSubevent':'has last subevent', 'MannerOf':'manner of', 'MadeOf':'made of',
                'CreatedBy':'created by', 'Desires':'desires', 'HasSubevent':'has subevent', 'FormOf':'form of',
                'UsedFor':'used for', 'MotivatedByGoal':'motivated by goal', 'SimilarTo':'similar to',
                'NotHasProperty':'not has property', 'HasContext':'has context', 'genus':'genus', 'capital':'capital',
                'leader':'leader', 'AtLocation':'at location', 'language':'language', 'IsA':'is a',
                'NotDesires':'not desires', 'ReceivesAction':'receives action', 'HasPrerequisite':'has prerequisite',
                'SymbolOf':'symbol of', 'DefinedAs':'defined as', 'CapableOf': 'capable of', 'Synonym':'synonym',
                'knownFor':'known for', 'Entails':'entails', 'Antonym':'antonym', 'product':'product',
                'CausesDesire':'causes desire', 'field':'field', 'HasFirstSubevent':'has first subevent',
                'EtymologicallyRelatedTo':'etymologically related to', 'HasProperty':'has property',
                'occupation':'occupation','RelatedTo':'related to', 'influencedBy': 'influence by',
                'PartOf':'part of', 'NotCapableOf':'not capable of'}


class RelationExample:
    def __init__(self, tokens, label):
        self.tokens = tokens
        self.label = label

def filter_string(s):
    #print(s)
    s = s.replace('_', ' ', -1)
    return s

def read_relation_example(knowlegde_graph_path:str):
    graph = collections.defaultdict(list)
    random_set = set()
    is_related = set()
    with open(knowlegde_graph_path, "r") as f:
        relation_list = json.load(f)
    for relation in relation_list:
        relation = relation.strip()
        if relation != '':
            start, relation, end = relation.split(',')
            random_set.add(start)
            random_set.add(end)
            graph[start].append((relation, end))
            is_related.add(start+end)
            if end not in graph:
                graph[end] = []
    return list(random_set), graph, is_related

def create_is_related(kgp:str):
    tokenizer = FullTokenizer(vocab_file=None, do_lower_case=True,
                              spm_model_file='ALBERT/pretrained_model/albert-base-v2-spiece.model')
    relations_list = list(relation_map.values())
    nodes, graph, is_related = read_relation_example(kgp)
    examples = []
    for i, node in enumerate(graph):
        #print(i)
        all_data = []
        nei = graph[node]
        for rep in range(REPEAT):
            for tu in nei:
                r, n = tu
                #print(node)
                r0 = np.random.randint(100)
                replaced = False
                if r0 < 15:
                    while True:
                        n = relations_list[np.random.randint(len(relations_list))]
                        if node+n not in is_related and n+node not in is_related:
                            break
                    replaced = True
                cur_token = []
                token1 = tokenizer.tokenize(filter_string(node))
                token2 = tokenizer.tokenize(relation_map[r])
                token3 = tokenizer.tokenize(filter_string(n))
                cur_token.append('[CLS]')
                cur_token.extend(token1)
                cur_token.append('[SEP]')
                cur_token.extend(token2)
                cur_token.append('[SEP]')
                cur_token.extend(token3)
                if replaced:
                    cur_label = 0
                else:
                    cur_label = 1
                label  = np.ones(len(cur_token)) * -1
                label[0] = cur_label
                all_data.append((cur_token, label))
        example_token = []
        example_label = []
        for i in range(len(all_data)):
            if len(example_token) + len(all_data[i][0]) <= 512:
                example_token.extend(tokenizer.convert_tokens_to_ids(all_data[i][0]))
                example_label.extend(all_data[i][1])
            else:
                examples.append(RelationExample(example_token, example_label))
                example_token = []
                example_label = []
        if len(example_token) != 0:
             examples.append(RelationExample(example_token, example_label))

    with open("data/relation_data.pkl", "wb") as f:
        pickle.dump(examples, f)

def create_relation_prediction(kgp:str):
    tokenizer = FullTokenizer(vocab_file=None, do_lower_case=True,
                              spm_model_file='ALBERT/pretrained_model/albert-base-v2-spiece.model')
    relations_list = list(relation_map.keys())
    relation_index = dict()
    for (i,r) in enumerate(relations_list):
        relation_index[r] = i
    print(len(relation_index))
    nodes, graph, _ = read_relation_example(kgp)
    examples = []
    for i, node in enumerate(graph):
        # print(i)
        all_data = []
        nei = graph[node]
        for rep in range(REPEAT):
            for tu in nei:
                r, n = tu
                # print(node)
                cur_token = []
                token1 = tokenizer.tokenize(filter_string(node))
                token3 = tokenizer.tokenize(filter_string(n))
                cur_token.extend(token1)
                cur_token.append('[MASK]')
                cur_token.extend(token3)
                cur_token.append('[SEP]')
                label = np.ones(len(cur_token)) * -1
                label[len(token1)] = relation_index[r]
                all_data.append((cur_token, label))
                #print(cur_token)
                #print(label)
        example_token = []
        example_label = []
        for i in range(len(all_data)):
            if len(example_token) + len(all_data[i][0]) <= 512:
                example_token.extend(tokenizer.convert_tokens_to_ids(all_data[i][0]))
                example_label.extend(all_data[i][1])
            else:
                examples.append(RelationExample(example_token, example_label))
                example_token = []
                example_label = []
        if len(example_token) != 0:
            examples.append(RelationExample(example_token, example_label))

    with open("data/relation_prediction_data.pkl", "wb") as f:
        pickle.dump(examples, f)

if __name__ == "__main__":
    create_is_related('data/subgraph.json')
    #create_relation_prediction('data/subgraph.json')