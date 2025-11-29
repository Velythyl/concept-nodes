import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="map_query")
def main(cfg: DictConfig):

    # Instantiate hydra objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction, device=cfg.device)
    verifier = hydra.utils.instantiate(cfg.vlm_verifier)
    semantic_sim = hydra.utils.instantiate(cfg.semantic_similarity)

    # Get inputs
    if cfg.name == "verifier":
        queries = dataset.get_pickupable_names()
        p2r_map = dataset.get_pickupable_to_receptacles()
    elif cfg.name == "map_receptacles":
        queries = dataset.get_receptacles_names()
        p2r_map = None

    map_path = cfg.map_path

    # Initialize Engine
    engine = hydra.utils.instantiate(
        cfg.inference,
        ft_extractor=ft_extractor,
        semantic_sim_metric=semantic_sim,
        verifier=verifier,
        receptacles_bbox=dataset.get_receptacles_bbox(),
        pickupable_to_receptacles=p2r_map,
    )

    # Run Query Logic
    results = engine.process_queries(queries=queries)

    output_path = Path(map_path) / "assignments"
    engine.save_results(results, output_path=output_path, sssd_data=dataset.sssd_data)
    log.info(f"Results saved to {output_path}")

    if cfg.debug:
        log.info("Visualizing map objects...")
        engine.visualize(output_path)
        


if __name__ == "__main__":
    main()
