def recursive_sorted(li):
    for i in range(len(li)):
        li[i] = sorted(li[i])
    return sorted(li)
