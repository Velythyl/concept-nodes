import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from concept_graphs.mapping.similarity.semantic import CosineSimilarity01
from concept_graphs.inference.MapQueryEngine import MapQueryObjects

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="map_query")
def main(cfg: DictConfig):

    # Instantiate hydra objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction, device=cfg.device)
    verifier = hydra.utils.instantiate(cfg.vlm_verifier)
    semantic_sim = hydra.utils.instantiate(cfg.semantic_similarity)

    # Get inputs
    pickupable_names = dataset.get_pickupable_names()
    receptacles_bbox = dataset.get_receptacles_bbox()

    map_path = cfg.map_path

    # Initialize Engine
    engine = hydra.utils.instantiate(
        cfg.inference,
        ft_extractor=ft_extractor,
        semantic_sim_metric=semantic_sim,
        verifier=verifier,
    )

    # Run Query Logic
    results = engine.process_queries(
        queries=pickupable_names, receptacles_bbox=receptacles_bbox, top_k=cfg.top_k
    )

    # Save Results
    output_path = Path(map_path) / "query_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
