# Plotting Models

```{meta}
:description: Visualize pgmpy graphs with Graphviz, daft, or networkx drawing tools.
```

pgmpy provides built-in plotting helpers for visualizing graph structures. These work
on all DAG-based models (since model classes inherit from `DAG`) and support multiple
backends for different use cases — quick inspection, publication figures, or file output.

## At a Glance

- **[Unified API](#api)**: Plotting methods are available directly on model objects.
- **[Graphviz Output](#graphviz)**: High-quality file output in PNG, PDF, SVG with automatic layout.
- **[Publication Figures](#publication-figures)**: Clean, paper-ready diagrams via daft.
- **[NetworkX Drawing](#networkx-drawing)**: Lightweight interactive plotting using the networkx ecosystem.

## API

Plotting helpers are called directly on the graph or model object:

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/sachs")

graphviz_graph = model.to_graphviz()
graphviz_graph.draw("sachs.png", prog="dot")

daft_graph = model.to_daft()
daft_graph.render()
```

## Graphviz

`to_graphviz()` converts the model to a PyGraphviz object with support for multiple
layout algorithms (`dot`, `neato`, `circo`, etc.) and output formats. Best for generating
high-quality file output for documentation and reports. Also works on `PDAG` objects for
visualizing partially directed graphs from discovery workflows.

## Publication Figures

`to_daft()` produces clean, publication-ready probabilistic graphical model diagrams
suitable for academic papers:

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/sachs")
daft_graph = model.to_daft(node_pos="circular")
daft_graph.render()
```

## NetworkX Drawing

Since pgmpy models are built on networkx, you can use `networkx.draw()` and related
functions directly for lightweight interactive plotting or integration with existing
networkx visualization pipelines:

```python
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.example_models import load_model

model = load_model("bnlearn/sachs")
pos = nx.spring_layout(model)
nx.draw(model, pos, with_labels=True, node_size=800, font_size=8)
plt.show()
```

## See Also

:::{seealso}
- {doc}`Defining a Custom Model <custom_model>` — Build the models you want to visualize.
- {doc}`Causal Discovery <causal_discovery>` — Learn graphs from data, then plot them.
:::

## API Reference

For the full list of graph and model classes with plotting support:

- {doc}`Graph Classes API <../api/base>`
- {doc}`Models API <../api/models>`
