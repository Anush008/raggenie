from collections import OrderedDict
from app.models.request import ConnectionArgument

# Plugin Metadata
__version__ = "1.0.0"
__vectordb_name__ = "qdrant"
__display_name__ = "Qdrant"
__description__ = "Qdrant for Vector Storage"
__icon__ = "/assets/vectordb/logos/qdrant.svg"
__connection_args__ = [
    {
        "config_type": 1,
        "name": "Qdrant Host",
        "description": "Host to connect to Qdrant",
        "order": 1,
        "required": True,
        "slug": "host",
        "field": "host",
        "placeholder": "localhost",
    },
    {
        "config_type": 1,
        "name": "Qdrant Port",
        "description": "Port to connect to Qdrant",
        "order": 2,
        "required": True,
        "slug": "port",
        "field": "port",
        "placeholder": "6333",
    },
]

__all__ = [
    __version__,
    __vectordb_name__,
    __display_name__,
    __description__,
    __icon__,
    __connection_args__,
]
