def list_models(**filter_tags) -> list[str]:
    """
    Lists all available example models.

    The models can be filtered based on their tags by providing keyword arguments. The available tags are:
    - name: str
    - n_nodes: No. of nodes in the model.
    - n_edges: No. of edges in the model.
    - is_parameterized: Whether it is just the network structure or also has parameters (CPDs) defined.
    - is_discrete: Whether the model has only discrete variables / parameterization.
    - is_continuous: Whether the model has only continuous variables / parameterization.
    - is_hybrid: Whether the model has both discrete and continuous variables / parameterization.

    Returns
    -------
    list
        List of names of all available example models.

    Examples
    --------
    >>> from pgmpy.example_models import list_models
    >>> list_models()
    ['bnlearn/alarm', 'bnlearn/arth150', ..... ]
    >>> list_models(is_discrete=True)
    ['bnlearn/alarm', 'bnlearn/asia', 'bnlearn/cancer', ..... ]
    >>> list_models(is_parameterized=False)
    ['dagitty/acid_1996', ...., ]
    """
    valid_tags = {"name", "n_nodes", "n_edges", "is_parameterized",
                  "is_discrete", "is_continuous", "is_hybrid"}
    invalid_tags = set(filter_tags.keys()) - valid_tags
    if invalid_tags:
        raise ValueError(
            f"Invalid filter tag(s): {invalid_tags}. "
            f"Valid tags are: {sorted(valid_tags)}."
        )

    return all_objects(
        object_types=_BaseExampleModel,
        package_name="pgmpy.example_models",
        filter_tags=filter_tags,
        return_names=True,
    )