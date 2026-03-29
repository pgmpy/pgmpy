# Exporting / Importing Models

```{meta}
:description: Read and write pgmpy models using the model-level save/load helpers and readwrite classes.
```

pgmpy supports importing and exporting models in several standard Bayesian network file
formats. This enables interoperability with tools like GeNIe, Hugin, and others, as well
as persisting fitted models to disk for later use.

## At a Glance

- **[Unified API](#api)**: Model-level `save(...)` and `load(...)` methods for common workflows.
- **[Multiple Formats](#supported-formats)**: BIF, NET, XMLBIF, XDSL, and more.
- **[Reader/Writer Classes](#readerwriter-classes)**: Fine-grained control over format-specific behavior.

## API

The simplest way to save and load models is through the model-level methods:

```python
from pgmpy.example_models import load_model
from pgmpy.models import DiscreteBayesianNetwork

model = load_model("bnlearn/alarm")
model.save("alarm.bif", filetype="bif")

loaded = DiscreteBayesianNetwork.load("alarm.bif", filetype="bif")
print(loaded)
```

## Supported Formats

pgmpy supports several Bayesian network interchange formats including BIF, NET, XMLBIF,
XDSL, XBN, UAI, and PomdpX. The model-level `save`/`load` methods handle format
detection automatically.

## Reader/Writer Classes

For fine-grained control, each format has dedicated reader and writer classes in
`pgmpy.readwrite` that expose format-specific options:

```python
from pgmpy.readwrite import BIFReader

reader = BIFReader("alarm.bif")
model = reader.get_model()
```

## See Also

:::{seealso}
- {doc}`Defining a Custom Model <custom_model>` — Build models from scratch instead of loading from files.
:::

## API Reference

For the full list of supported formats and I/O classes:

- {doc}`Reading/Writing API <../api/readwrite>`
- {doc}`Models API <../api/models>`
