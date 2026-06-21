from .gdb_networkx import NetworkXStorage
from .gdb_neo4j import Neo4jStorage
from .vdb_nanovectordb import NanoVectorDBStorage
from .kv_json import JsonKVStorage

try:
    from .vdb_hnswlib import HNSWVectorStorage
except ImportError as import_error:
    _HNSW_IMPORT_ERROR = import_error

    class HNSWVectorStorage:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "HNSWVectorStorage requires hnswlib and xxhash. "
                "Install optional dependencies with `pip install -r requirements-optional.txt`."
            ) from _HNSW_IMPORT_ERROR
