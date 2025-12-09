class DatasetRegistry:
    def __init__(self) -> None:
        self._by_name = {}
        self._by_tag = {}

    def register(self, cls):
        self._by_name[cls.name] = cls
        for tag in cls.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = set()
            self._by_tag[tag].add(cls.name)

    def list_all(self, tag=None):
        if tag is None:
            return list(self._by_name.keys())
        else:
            return list(self._by_tag.get(tag, []))

    def get(self, name):
        return self._by_name.get(name, None)


DATASETS = DatasetRegistry()


def dataset_class(cls):
    DATASETS.register(cls)
    return cls
