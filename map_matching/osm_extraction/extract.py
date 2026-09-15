"""OSM road graph extraction for map matching (Phase 5.1).

Extracts the road network for a corridor area from OpenStreetMap using osmnx
and saves it as a GraphML file for offline use. This script runs at BUILD TIME
only — never at inference/runtime.

Usage (from repo root):
    # IO-VNBD sessions (Coventry, UK)
    python map_matching/osm_extraction/extract.py \\
        --lat 52.4081 --lon -1.5106 --dist 5000 \\
        --output map_matching/osm_extraction/coventry.graphml

    # Indian corridor (replace with actual coordinates once available)
    python map_matching/osm_extraction/extract.py \\
        --lat <lat> --lon <lon> --dist 5000 \\
        --output map_matching/osm_extraction/corridor.graphml

The extracted graph is used by:
  - map_matching/hmm/viterbi_matcher.py (runtime map matching)
  - data/scripts/03_window.py road_signature builder (for global edge ID assignment)

IMPORTANT: Run this once per corridor, commit the GraphML file to the repo
(or an LFS-equivalent store), and never re-fetch at runtime. OSM rate-limits
bulk fetches and the graph changes slowly enough that a build-time snapshot is
appropriate for the project's 6-month horizon.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_corridor(
    lat: float,
    lon: float,
    dist_m: int = 5000,
    output_path: str = "map_matching/osm_extraction/corridor.graphml",
    network_type: str = "drive",
) -> None:
    """Fetch drive-mode road network around (lat, lon) within dist_m meters
    and save as GraphML.

    Args:
        lat: Latitude of the area center (decimal degrees).
        lon: Longitude of the area center (decimal degrees).
        dist_m: Radius in meters around the center point to include.
        output_path: Where to write the GraphML file.
        network_type: OSM network type ("drive" for roads only, which is what
            map matching needs — excludes footpaths, cycleways, etc.).
    """
    try:
        import osmnx as ox
    except ImportError:
        raise SystemExit(
            "osmnx is not installed. Install it with:\n"
            "  pip install osmnx\n"
            "or inside the ml container:\n"
            "  pip install osmnx shapely pyproj"
        )

    print(f"Fetching OSM {network_type} graph: center=({lat}, {lon}), radius={dist_m}m...")
    G = ox.graph_from_point((lat, lon), dist=dist_m, network_type=network_type)

    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    print(f"  Extracted: {n_nodes} nodes, {n_edges} edges")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, filepath=str(out_path))
    print(f"  Saved: {out_path}")

    # Print summary of road types for sanity check
    edge_types = {}
    for _, _, data in G.edges(data=True):
        highway = data.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0]
        edge_types[highway] = edge_types.get(highway, 0) + 1
    print("  Road type breakdown:")
    for rtype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"    {rtype}: {count}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract OSM road graph for map matching.")
    ap.add_argument("--lat", type=float, required=True, help="Center latitude (decimal degrees).")
    ap.add_argument("--lon", type=float, required=True, help="Center longitude (decimal degrees).")
    ap.add_argument("--dist", type=int, default=5000, help="Radius in meters (default: 5000).")
    ap.add_argument("--output", default="map_matching/osm_extraction/corridor.graphml",
                    help="Output GraphML path.")
    ap.add_argument("--network-type", default="drive",
                    choices=["drive", "walk", "bike", "all"],
                    help="OSM network type filter (default: drive).")
    args = ap.parse_args()

    extract_corridor(args.lat, args.lon, args.dist, args.output, args.network_type)


if __name__ == "__main__":
    main()
