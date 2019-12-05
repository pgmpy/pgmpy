def recursive_sorted(li):
    li = list(li)
    for i in range(len(li)):
        li[i] = sorted(li[i])
    return sorted(li)
