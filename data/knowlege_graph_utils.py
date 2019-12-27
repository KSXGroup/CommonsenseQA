import collections
import json

def get_index():
    idx = 1
    rel_idx = 1
    node_to_idx = collections.defaultdict(int)
    rel_to_idx = collections.defaultdict(int)
    with open("subgraph.json", "r") as f:
        relations = json.load(f)
    print(len(relations))
    for relation in relations:
        s, r, t = relation.split(',')
        if r not in rel_to_idx:
            rel_to_idx[r] = rel_idx
            rel_idx += 1

        if s not in node_to_idx:
            node_to_idx[s] = idx
            idx += 1

        if t not in node_to_idx:
            node_to_idx[t] = idx
            idx += 1
    return rel_to_idx, node_to_idx

def buid_graph():
    rel_to_idx, node_to_idx = get_index()
    graph = dict()
    with open("subgraph.json", "r") as f:
        relations = json.load(f)
    for relation in relations:
        s, r, t = relation.split(',')
        if s not in graph:
            graph[s] = [(rel_to_idx[r], node_to_idx[t])]
        else:
            graph[s].append((rel_to_idx[r], node_to_idx[t]))
    print(len(node_to_idx))
    return (node_to_idx, rel_to_idx, graph)


def filter_string_1(s):
    s = s.lower()
    s = s.replace(' ', '_', -1)
    return s


def filter_string_2(s):
    s = s.lower()
    s = s.replace(' ', '_', -1)
    s = s.replace('.', '', -1)
    return s


def get_kg_node(entity, kg):
    s1 = filter_string_2(entity)
    if s1 in kg: return s1
    s1 = filter_string_2(entity)
    if s1 in kg: return s1

if __name__ == "__main__":
    buid_graph()

# with open("subgraph.json", "r") as f:
#     ilist = json.load(f)
# nodes = collections.defaultdict(dict)
# visited = set()
# for item in ilist:
#     ls = item.split(',')
#
# def bfs(n):
#     Q = []
#     cnt = 0
#     Q.append(n)
#     while len(Q) > 0:
#         cur = Q.pop(0)
#         cnt += 1
#         if cur not in nodes:  continue
#         for relation in nodes[cur]:
#             for neighbour in nodes[cur][relation]:
#                 if neighbour  not in visited:
#                     visited.add(neighbour)
#                     Q.append(neighbour)
#     return cnt
# print(len(nodes))
# small_cnt = 0
# avg = 0
# for n in nodes:
#     if n not in visited:
#         sz = bfs(n)
#         if sz > 20:
#             print("Big: %d" % sz)
#         else:
#             #if sz <= 2:print(n)
#             small_cnt += 1
#             avg += sz
# print("Small count %d, avg %.5f" % (small_cnt, avg / float(small_cnt)))
