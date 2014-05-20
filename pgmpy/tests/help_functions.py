def recursive_sorted(li):
    for i in range(len(li)):
        li[i] = sorted(li[i])
    return sorted(li)

def assertOrderedDictEqual(dict1, dict2):
    if not list(dict1.items()) == list(dict2.items()):
        raise AssertionError(str(dict1) + " is not equal to " + str(dict2))

