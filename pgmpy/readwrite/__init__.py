from .BIF import BIFReader, BIFWriter
from .LGBNJSON import LGBNJSONReader, LGBNJSONWriter
from .NET import NETReader, NETWriter
from .PomdpX import PomdpXReader, PomdpXWriter
from .UAI import UAIReader, UAIWriter
from .XDSL import XDSLReader, XDSLWriter
from .XMLBeliefNetwork import XBNReader, XBNWriter
from .XMLBIF import XMLBIFReader, XMLBIFWriter

# Formats available to the models' `save` and `load` methods: format_name -> (Reader, Writer). The first format whose
# writer supports a model class is that model's default format.
FORMATS = {
    "bif": (BIFReader, BIFWriter),
    "net": (NETReader, NETWriter),
    "uai": (UAIReader, UAIWriter),
    "xmlbif": (XMLBIFReader, XMLBIFWriter),
    "xdsl": (XDSLReader, XDSLWriter),
    "xbn": (XBNReader, XBNWriter),
    "json": (LGBNJSONReader, LGBNJSONWriter),
}

__all__ = [
    "XMLBIFReader",
    "XMLBIFWriter",
    "XBNReader",
    "XBNWriter",
    "XDSLReader",
    "XDSLWriter",
    "PomdpXReader",
    "PomdpXWriter",
    "UAIReader",
    "UAIWriter",
    "BIFReader",
    "BIFWriter",
    "NETReader",
    "NETWriter",
    "LGBNJSONReader",
    "LGBNJSONWriter",
    "FORMATS",
]
