import collections
import json
import numpy as np
def graph_to_idx():
    with open("graph.txt", "r") as f:
        ilist = f.read().strip().split('\n')
    cnt = 1
    node_to_idx = dict()
    for item in ilist:
        l = item.strip().split(',')
        if l[0] not in node_to_idx and len(l[0]) > 0:
            node_to_idx[l[0]] = cnt
            cnt += 1
        if l[2] not in node_to_idx and len(l[2]) > 0:
            node_to_idx[l[2]] = cnt
            cnt += 1
    return node_to_idx


def fliter_string(s):
    ret = ""
    for i in s:
        if (i >= 'a' and i <= 'z') or (i >= '0' and i <= '9'):
            ret += i
    return ret


def lcs_matching(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    s1 = fliter_string(s1)
    s2 = fliter_string(s2)
    dp = np.zeros((1000,1000))
    m = len(s1)
    n = len(s2)
    for p in range(m):
        for q in range(n):
            if s1[p] == s2[q]:
                dp[p + 1,q + 1] = dp[p,q] + 1
            else:
                dp[p + 1,q + 1] = max(dp[p, q+1], dp[p+1,q])
    res = dp[m, n]
    if res > 0.8 * n and res > 0.8 * m: return True, res / float(m) + res / float(n)
    else: return False, -1


def matching_enrich(nodes):
    with open("dataset/record/train.json", "r") as f:
        train = json.load(f)
    train = train["data"]
    node_dicted = dict()
    cache_dict = dict()
    for i in range(26):
        node_dicted[chr(ord('a') + i)] = []
    for i in range(10):
        node_dicted[chr(ord('0') + i)] = []
    for n in nodes:
        #print(n)
        if n.lower()[0] not in node_dicted :
            print(n)
            continue
        else: node_dicted[n.lower()[0]].append(n)
    result = dict()
    for i, ex in enumerate(train):
        text = ex["passage"]["text"]
        entities = ex["passage"]["entities"]
        unique_id = ex["id"]
        print("Example %d" % i)
        entity_to_graph = dict()
        for entity in entities:
            s = entity["start"]
            e = entity["end"]
            entity_text = text[s:e + 1]
            print("matching %s" % entity_text)
            p = 0
            if entity_text in cache_dict:
                print("cached node: %s, entity: %s" % (cache_dict[entity_text], entity_text))
                entity_to_graph[entity_text] = cache_dict[entity_text]
            else:
                for j, node in enumerate(node_dicted[entity_text[0].lower()]):
                    matched, score = lcs_matching(node, entity_text)
                    if matched and score > p:
                        entity_to_graph[entity_text] = node
                        print("%.5f node: %s, entity: %s"%(score, node, entity_text))
                        cache_dict[entity_text] = node
                        p = score
                        if p == 2.0: break
            if entity_text not in entity_to_graph:
                entity_to_graph[entity_text] = None
                cache_dict[entity_text] =  None
            print('done')
        result[unique_id] = entity_to_graph
        print("Exampe %d done" % i)
    return result

if __name__ == "__main__":
    node_idx = graph_to_idx()
    nodes = node_idx.keys()
    res = matching_enrich(nodes)
    enriched_entity = [node_idx, res]
    with open("enriched_entity.json", "w") as f:
        json.dump(enriched_entity,f)
