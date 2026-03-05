"""Physical group assignment for Palace mesh generation.

This module builds the ``groups`` dict consumed by the config generator
from the ``pg_map`` produced by ``run_boolean_pipeline``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gmsh

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


def assign_physical_groups(
    kernel,
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
    entities: list[gmsh_utils.Entity],
    pg_map: dict[str, int],
    _stack: LayerStack,
) -> dict:
    """Build the ``groups`` dict from the boolean-pipeline result.

    Args:
        kernel: gmsh OCC kernel
        metal_tags: Metal layer tags from add_metals()
        dielectric_tags: Dielectric material tags from add_dielectrics()
        port_tags: Port surface tags (may have multiple surfaces for CPW)
        port_info: Port metadata including type info
        entities: Entity list used in run_boolean_pipeline
        pg_map: name → physical-group tag returned by run_boolean_pipeline
        _stack: Layer stack (reserved for future use)

    Returns:
        Dict with the same schema as before::

            {
                "volumes": {name: {"phys_group": int, "tags": [int]}},
                "conductor_surfaces": {name: {"phys_group": int, "tags": [int]}},
                "pec_surfaces": {name: {"phys_group": int, "tags": [int]}},
                "port_surfaces": {name: ...},
                "boundary_surfaces": {name: {"phys_group": int, "tags": [int]}},
            }
    """
    groups: dict[str, dict] = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    # Helper: entity name → (phys_group, surface_tags)
    entity_by_name: dict[str, gmsh_utils.Entity] = {e.name: e for e in entities}

    # --- Volumes (dielectrics + airbox) ---
    for material in dielectric_tags:
        entity = entity_by_name.get(material)
        pg = pg_map.get(material)
        if entity and pg is not None:
            vol_tags = [t for d, t in entity.dimtags if d == 3]
            if vol_tags:
                groups["volumes"][material] = {
                    "phys_group": pg,
                    "tags": vol_tags,
                }

    # --- PEC surfaces (planar conductors) ---
    for layer_name, tag_info in metal_tags.items():
        if tag_info["surfaces_xy"]:
            pec_name = f"{layer_name}_pec"
            entity = entity_by_name.get(pec_name)
            pg = pg_map.get(pec_name)
            if entity and pg is not None:
                surf_tags = [t for d, t in entity.dimtags if d == 2]
                if surf_tags:
                    groups["pec_surfaces"][layer_name] = {
                        "phys_group": pg,
                        "tags": surf_tags,
                    }

    # --- Port surfaces ---
    for port_name, tags in port_tags.items():
        port_num = int(port_name[1:])
        info = next(
            (p for p in port_info if p["portnumber"] == port_num),
            None,
        )

        if info and info.get("type") == "cpw":
            element_phys_groups = []
            for i in range(len(tags)):
                elem_name = f"{port_name}_E{i}"
                entity = entity_by_name.get(elem_name)
                pg = pg_map.get(elem_name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        element_phys_groups.append(
                            {
                                "phys_group": pg,
                                "tags": surf_tags,
                                "direction": info["elements"][i]["direction"],
                            }
                        )
            groups["port_surfaces"][port_name] = {
                "type": "cpw",
                "elements": element_phys_groups,
            }
        else:
            pg = pg_map.get(port_name)
            entity = entity_by_name.get(port_name)
            if entity and pg is not None:
                surf_tags = [t for d, t in entity.dimtags if d == 2]
                if surf_tags:
                    groups["port_surfaces"][port_name] = {
                        "phys_group": pg,
                        "tags": surf_tags,
                    }

    # --- Boundary surfaces (outer faces labelled *__None by the pipeline) ---
    boundary_pgs: list[int] = [
        pg for name, pg in pg_map.items() if name.endswith("__None")
    ]
    if boundary_pgs:
        groups["boundary_surfaces"]["absorbing"] = {
            "phys_group": boundary_pgs,
            "tags": [],  # tags not needed; pg_map is authoritative
        }

    # Label every remaining surface created by volume fragmentation -------
    # Surfaces already claimed by conductors, ports, or the absorbing
    # boundary keep their names.  Every other surface is labelled with the
    # sorted pair of volume names that share it, e.g. "air__substrate".
    groups["interface_surfaces"] = _assign_interface_surfaces(groups)

    kernel.synchronize()
    return groups


# ---------------------------------------------------------------------------
# Interface surface labelling
# ---------------------------------------------------------------------------


def _assign_interface_surfaces(groups: dict) -> dict:
    """Label volume-boundary surfaces not yet in a physical group.

    Each surface is named ``"vol1__vol2"`` when shared by two volumes, or
    ``"vol__None"`` when it belongs to only one volume.

    Returns:
        ``{label: {"phys_group": int, "tags": [int]}}``
    """
    # Collect surface tags that already have a physical group
    assigned: set[int] = set()
    for section in (
        "conductor_surfaces",
        "pec_surfaces",
        "port_surfaces",
        "boundary_surfaces",
    ):
        for info in groups[section].values():
            if info.get("type") == "cpw":
                for elem in info["elements"]:
                    assigned.update(elem["tags"])
            else:
                assigned.update(info.get("tags", []))

    # Build surface → owning-volume-name(s) map
    surf_to_names: dict[int, list[str]] = {}
    for vol_name, vol_info in groups["volumes"].items():
        for vol_tag in vol_info["tags"]:
            try:
                boundary = gmsh.model.getBoundary(
                    [(3, vol_tag)], combined=False, oriented=False, recursive=False
                )
            except Exception:
                logger.debug("Could not query boundary of volume %d", vol_tag)
                continue
            for bdim, btag in boundary:
                if bdim == 2:
                    surf_to_names.setdefault(btag, [])
                    if vol_name not in surf_to_names[btag]:
                        surf_to_names[btag].append(vol_name)

    # Group unassigned surfaces by their sorted owner-name combination
    label_to_surfs: dict[str, list[int]] = {}
    for stag, names in surf_to_names.items():
        if stag in assigned:
            continue
        label = "__".join(sorted(names)) if len(names) > 1 else f"{names[0]}__None"
        label_to_surfs.setdefault(label, []).append(stag)

    # Create physical groups
    result: dict[str, dict] = {}
    for label, stags in label_to_surfs.items():
        phys_group = gmsh_utils.assign_physical_group(2, stags, label)
        result[label] = {"phys_group": phys_group, "tags": stags}
        logger.debug(
            "Interface surface '%s': pg=%d, %d tags", label, phys_group, len(stags)
        )

    return result


__all__ = ["assign_physical_groups"]
