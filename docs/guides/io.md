# Exporting / Importing Models

```{eval-rst}
.. meta::
   :description: Read and write pgmpy models with the save and load methods and the readwrite classes.
```

pgmpy can read and write models in several standard Bayesian network file formats. Use these
formats to exchange models with tools such as GeNIe and Hugin, and to save fitted models to
disk.

## At a Glance

::::{grid} 1 1 3 3
:gutter: 3
:class-container: pgmpy-card-grid pgmpy-glance-grid

:::{grid-item-card} Unified API
:link: "#api"
:link-type: url
:class-card: sd-card-hover

Use the `save(...)` and `load(...)` methods of the model for most tasks.
:::

:::{grid-item-card} Supported Formats
:link: "#supported-formats"
:link-type: url
:class-card: sd-card-hover

BIF, NET, XMLBIF, XDSL, XBN, UAI, and JSON.
:::

:::{grid-item-card} Reader/Writer Classes
:link: "#reader-writer-classes"
:link-type: url
:class-card: sd-card-hover

Control the options of each file format.
:::
::::

## API

Use the `save` and `load` methods of the model:

```python
from pgmpy.example_models import load_model
from pgmpy.models import DiscreteBayesianNetwork

model = load_model("bnlearn/alarm")
model.save("alarm.bif")

loaded = DiscreteBayesianNetwork.load("alarm.bif")
print(loaded)
```

## Supported Formats

The `save` and `load` methods support these formats. `pgmpy.readwrite.FORMATS` contains
the same list:

- `DiscreteBayesianNetwork`: BIF (default), NET, XMLBIF, XDSL, XBN, and UAI.
- `LinearGaussianBayesianNetwork`: bnlearn-compatible JSON (default).

pgmpy selects the format in this order:

1. The `filetype` argument, if you give it.
2. The file extension.
3. The default format of the model.

pgmpy gives all other keyword arguments to the reader or writer class of the format:

```python
model.save("alarm.bif", round_values=4)
loaded = DiscreteBayesianNetwork.load("alarm.bif", state_name_type=str)
```

`load` also accepts a file-like object (text or binary) instead of a path. A file object has
no file extension. If the file is not in the default format of the model, give `filetype`:

```python
with open("alarm.bif") as f:
    loaded = DiscreteBayesianNetwork.load(f, filetype="bif")
```

For PomdpX, use the `PomdpXReader` and `PomdpXWriter` classes. The `save` and `load` methods
do not support it.

## Reader/Writer Classes

For more control, use the reader and writer classes in `pgmpy.readwrite`. Each format has
its own classes, and all classes use the same interface:

- A reader takes `path` or `string` and the options of its format. `read()` returns the
  model.
- A writer takes the model and the options of its format. `write(filename)` writes the file.
  `str(writer)` returns the contents of the file as a string.

```python
from pgmpy.readwrite import BIFReader, BIFWriter

writer = BIFWriter(model, round_values=4)
writer.write("alarm.bif")
bif_string = str(writer)

model = BIFReader("alarm.bif", state_name_type=str).read()
model = BIFReader(string=bif_string).read()
```

The {doc}`Reading/Writing API <../api/readwrite>` gives the options of each class.

## See Also

:::{seealso}
- {doc}`Defining a Custom Model <custom_model>` — Define models in code.
:::

## API Reference

These pages list all supported formats and reader and writer classes:

- {doc}`Reading/Writing API <../api/readwrite>`
- {doc}`Models API <../api/models>`
