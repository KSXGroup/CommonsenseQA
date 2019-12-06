import collections
with open("graph.txt", "r") as f:
    ilist = f.read().strip().split('\n')
nodes = collections.defaultdict(dict)
visited = set()
for item in ilist:
    ls = item.split(',')
    if ls[1] not in nodes[ls[0]]: nodes[ls[0]][ls[1]] = [ls[2]]
    else: nodes[ls[0]][ls[1]].append(ls[2])

def bfs(n):
    Q = []
    cnt = 0
    Q.append(n)
    while len(Q) > 0:
        cur = Q.pop(0)
        cnt += 1
        if cur not in nodes:  continue
        for relation in nodes[cur]:
            for neighbour in nodes[cur][relation]:
                if neighbour  not in visited:
                    visited.add(neighbour)
                    Q.append(neighbour)
    return cnt

small_cnt = 0
avg = 0
for n in nodes:
    if n not in visited:
        sz = bfs(n)
        if sz > 20:
            print("Big: %d" % sz)
        else:
            if sz <= 2:print(n)
            small_cnt += 1
            avg += sz
print("Small count %d, avg %.5f" % (small_cnt, avg / float(small_cnt)))
