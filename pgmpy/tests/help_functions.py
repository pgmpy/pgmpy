from pgmpy.extern.six.moves import range


def recursive_sorted(li):
    li = list(li)
    for i in range(len(li)):
        li[i] = sorted(li[i])
    return sorted(li)


def recursive_frozenset(edges):
    for i in range(len(edges)):
        edges[i] = frozenset(edges[i])
    return edges
