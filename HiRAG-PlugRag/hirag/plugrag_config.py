import os


GRAPH_ENCODER_ALIASES = {
    "rgcn": "rgcn",
    "r-gcn": "rgcn",
    "plugrag_rgcn": "rgcn",
    "gcn": "gcn",
    "simple_gcn": "gcn",
    "plugrag_simple_gcn": "gcn",
    "multi_edge": "multi_edge_rgcn_transe",
    "multi_edge_rgcn": "multi_edge_rgcn_transe",
    "multi_edge_rgcn_transe": "multi_edge_rgcn_transe",
    "rgcn_transe": "multi_edge_rgcn_transe",
}

DISABLED_GRAPH_ENCODERS = {
    "none",
    "off",
    "rank_only",
    "semantic_rank_only",
    "plugrag_semantic_rank_only",
}


def _default_mapping_file(output_pt: str | None) -> str | None:
    if not output_pt:
        return None
    stem, ext = os.path.splitext(output_pt)
    if ext:
        return f"{stem}_mapping.json"
    return f"{output_pt}_mapping.json"


def resolve_graph_encoder_files(config: dict, batch_config: dict | None = None) -> dict:
    """Resolve which trained graph encoder embeddings a query script should load."""
    batch_config = batch_config or {}
    raw_encoder = (
        os.environ.get("PLUGRAG_GRAPH_ENCODER")
        or batch_config.get("graph_encoder")
        or "rgcn"
    )
    encoder = str(raw_encoder).strip().lower()

    if encoder in DISABLED_GRAPH_ENCODERS:
        return {
            "encoder": encoder,
            "enabled": False,
            "embeddings_file": None,
            "mapping_file": None,
        }

    section_name = GRAPH_ENCODER_ALIASES.get(encoder)
    if section_name is None:
        valid = sorted(set(GRAPH_ENCODER_ALIASES) | DISABLED_GRAPH_ENCODERS)
        raise ValueError(
            f"Unknown graph encoder '{raw_encoder}'. Valid values: {', '.join(valid)}"
        )

    training_cfg = config.get(section_name, {}).get("training", {})
    embeddings_file = (
        os.environ.get("PLUGRAG_GRAPH_EMBEDDINGS_FILE")
        or training_cfg.get("output_pt")
    )
    mapping_file = (
        os.environ.get("PLUGRAG_GRAPH_MAPPING_FILE")
        or training_cfg.get("mapping_file")
        or _default_mapping_file(embeddings_file)
    )

    return {
        "encoder": section_name,
        "enabled": bool(embeddings_file and mapping_file),
        "embeddings_file": embeddings_file,
        "mapping_file": mapping_file,
    }
