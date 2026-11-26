# Plotting Models

```{eval-rst}
.. meta::
   :description: Draw pgmpy graphs with Graphviz, daft, or networkx.
```

pgmpy has methods that draw graph structures. They work on all models that are subclasses
of `DAG`. Select a backend for your task: a quick check, a publication figure, or an output
file.

## At a Glance

::::{grid} 1 1 2 2
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

Call the plot methods directly on the model object.
:::

:::{grid-item-card} Graphviz Output
:link: "#graphviz"
:link-type: url
:class-card: sd-card-hover

Write PNG, PDF, or SVG files with automatic layout.
:::

:::{grid-item-card} Publication Figures
:link: "#publication-figures"
:link-type: url
:class-card: sd-card-hover

Make diagrams for papers with daft.
:::

:::{grid-item-card} NetworkX Drawing
:link: "#networkx-drawing"
:link-type: url
:class-card: sd-card-hover

Draw interactive plots with networkx and matplotlib.
:::
::::

## API

Call the plot methods on the graph or model object:

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/sachs")

graphviz_graph = model.to_graphviz()
graphviz_graph.draw("sachs.png", prog="dot")

daft_graph = model.to_daft()
daft_graph.render()
```

## Graphviz

`to_graphviz()` converts the model to a PyGraphviz object. PyGraphviz supports many layout
algorithms (for example `dot`, `neato`, and `circo`) and output formats. Use it to make files
for documentation and reports. It also works on `PDAG` objects, for example the partially
directed graphs from causal discovery.

## Publication Figures

`to_daft()` makes probabilistic graphical model diagrams for academic papers:

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/sachs")
daft_graph = model.to_daft(node_pos="circular")
daft_graph.render()
```

## NetworkX Drawing

pgmpy models are networkx graphs. Thus, you can use `networkx.draw()` and the other networkx
draw functions directly. Use them for interactive plots or with your existing networkx code:

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
- {doc}`Defining a Custom Model <custom_model>` — Make the models that you want to draw.
- {doc}`Causal Discovery <causal_discovery>` — Learn graphs from data, then plot them.
:::

## API Reference

These pages list all graph and model classes that you can plot:

- {doc}`Graph Classes API <../api/base>`
- {doc}`Models API <../api/models>`
