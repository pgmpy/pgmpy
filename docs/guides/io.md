# Exporting / Importing Models

```{eval-rst}
.. meta::
   :description: Read and write pgmpy models using the model-level save/load helpers and readwrite classes.
```

pgmpy supports importing and exporting models in several standard Bayesian network file
formats. This enables interoperability with tools like GeNIe, Hugin, and others, as well
as persisting fitted models to disk for later use.

## At a Glance

- **[Unified API](#api)**: Model-level `save(...)` and `load(...)` methods for common workflows.
- **[Multiple Formats](#supported-formats)**: BIF, NET, XMLBIF, XDSL, and more.
- **[Reader/Writer Classes](#reader-writer-classes)**: Fine-grained control over format-specific behavior.

## API

The simplest way to save and load models is through the model-level methods:

```python
from pgmpy.example_models import load_model
from pgmpy.models import DiscreteBayesianNetwork

model = load_model("bnlearn/alarm")
model.save("alarm.bif")

loaded = DiscreteBayesianNetwork.load("alarm.bif")
print(loaded)
```

## Supported Formats

The model-level `save`/`load` methods support the following formats (listed in
`pgmpy.readwrite.FORMATS`):

- `DiscreteBayesianNetwork`: BIF (default), NET, XMLBIF, XDSL, XBN, and UAI.
- `LinearGaussianBayesianNetwork`: bnlearn-compatible JSON (default).

The format is taken from `filetype` if given, otherwise from the file extension, and otherwise
falls back to the model's default format. Any other keyword arguments are passed to the reader or
writer class of the format:

```python
model.save("alarm.bif", round_values=4)
loaded = DiscreteBayesianNetwork.load("alarm.bif", state_name_type=str)
```

`load` also accepts a file-like object (text or binary) instead of a path. Since there is no file
extension to infer the format from, pass `filetype` unless the file is in the model's default format:

```python
with open("alarm.bif") as f:
    loaded = DiscreteBayesianNetwork.load(f, filetype="bif")
```

PomdpX is available only through the `PomdpXReader` and `PomdpXWriter` classes.

## Reader/Writer Classes

For fine-grained control, each format has dedicated reader and writer classes in
`pgmpy.readwrite`, and all of them share the same interface:

- Readers take either `path` or `string`, plus format-specific options, and `read()` returns the
  model.
- Writers take the model, plus format-specific options. `write(filename)` writes the file, and
  `str(writer)` returns the file contents as a string.

```python
from pgmpy.readwrite import BIFReader, BIFWriter

writer = BIFWriter(model, round_values=4)
writer.write("alarm.bif")
bif_string = str(writer)

model = BIFReader("alarm.bif", state_name_type=str).read()
model = BIFReader(string=bif_string).read()
```

See the {doc}`Reading/Writing API <../api/readwrite>` for the options of each class.

## See Also

:::{seealso}
- {doc}`Defining a Custom Model <custom_model>` — Build models from scratch instead of loading from files.
:::

## API Reference

For the full list of supported formats and I/O classes:

- {doc}`Reading/Writing API <../api/readwrite>`
- {doc}`Models API <../api/models>`
