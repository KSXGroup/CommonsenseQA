import collections
import argparse
import json
import numpy as np
import pickle

WIDTH = 3
def graph_to_idx():
    with open("subgraph.json", "r") as f:
        ilist = json.load(f)
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

def build_graph():
    nodes = collections.defaultdict(dict)
    with open("graph.txt", "r") as f:
        ilist = f.read().strip().split('\n')
    print(len(ilist))
    for item in ilist:
        ls = item.split(',')
        if ls[1] not in nodes[ls[0]]: nodes[ls[0]][ls[1]] = [ls[2]]
        else: nodes[ls[0]][ls[1]].append(ls[2])
    return nodes

def fliter_string_1(s):
    s = s.lower()
    s = s.replace(' ', '_', -1)
    return s

def fliter_string_2(s):
    s = s.lower()
    s = s.replace(' ', '_', -1)
    s = s.replace('.', '', -1)
    return s

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

def bounded_bfs(start, boundary, graph):
    Q = []
    visted = set()
    visted.add(start)
    relations = set()
    Q.append((start, 0))
    while(len(Q) > 0):
        node, dep = Q.pop(0)
        if dep == boundary or node not in graph: continue
        else:
            avarelation = graph[node]
            for rel in avarelation:
                for nei in avarelation[rel]:
                    if nei not in visted:
                        visted.add(nei)
                        Q.append((nei, dep+1))
                        relations.add(','.join([node, rel, nei]))
    return relations


def matching_enrich(graph, args):
    filtered_node_to_node = graph.keys()
    subgraph = set()
    with open("dataset/record/train.json", "r") as f:
        train = json.load(f)
    train = train["data"]
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    maxen = 0
    avg = 0
    print(len(train))
    for i, ex in enumerate(train[args.begin:args.end]):
        text = ex["passage"]["text"]
        entities = ex["passage"]["entities"]
        avg += len(entities)
        # if len(entities) > 50:
        #     cnt3 += 1
        # if len(entities) > maxen:
        #     maxen = len(entities)
        #     print(maxen)
        cur = 0
        for entity in entities:
            cnt1 += 1
            s = entity["start"]
            e = entity["end"]
            entity_text = text[s:e + 1]
            matched1 = (fliter_string_1(entity_text) in filtered_node_to_node)
            if matched1:
                rels = bounded_bfs(fliter_string_1(entity_text), WIDTH, graph)
                for rel in rels: subgraph.add(rel)
            else:
                matched2 = (fliter_string_2(entity_text) in filtered_node_to_node)
                if matched2:
                    rels = bounded_bfs(fliter_string_2(entity_text), WIDTH, graph)
                    for rel in rels: subgraph.add(rel)
            print(len(rels))
            #print('done')
        print("Exampe %d done" % i)
    print("matching rate %.5f" % (cnt2 / cnt1))
    return subgraph

if __name__ == "__main__":
    nti = graph_to_idx()
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    graph = build_graph()
    print(len(graph))
    #res = matching_enrich(graph, args)
    #with open("related_graph-%d.json" % args.begin, "w") as f:
    #    json.dump(list(res), f)
