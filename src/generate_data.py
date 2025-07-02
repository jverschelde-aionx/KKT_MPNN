import logging
import os

from ecole.instance import CombinatorialAuctionGenerator, IndependentSetGenerator
from tqdm import tqdm, trange

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Generation parameters
n_instances = 1000
# Independent Set parameters
is_nodes = [
    10,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    50000,
    100000,
    500000,
    1000000,
]
# Combinatorial Auction parameters
ca_items = [10, 50, 100, 200, 500, 1000]
distributions = ["uniform", "normal", "log-normal"]


# Generate Combinatorial Auction instances
for n_items in tqdm(ca_items, desc="Combinatorial Auction instance generation"):
    n_bids = n_items * 5  # Number of bids is typically a multiple of items
    logging.info(
        f"Generating Combinatorial Auction instances with {n_items} items and {n_bids} bidders"
    )

    CA_generator = CombinatorialAuctionGenerator(n_items=n_items, n_bids=n_bids)
    # emit instances
    for i in trange(n_instances, desc=f"Generating {n_items} items"):
        file_path = f"./instance/train/CA/ca-items{n_items}-bidders{n_bids}-{i:04}.lp"
        # Check if the file already exists to avoid overwriting
        if os.path.exists(file_path):
            continue
        # Generate instance
        instance = next(CA_generator)
        # write out to .lp
        instance.write_problem(file_path)


# Generate Indenpendent Set instances
for n_nodes in tqdm(is_nodes, desc="Independent Set instance generation"):
    logging.info(f"Generating Independent Set instances with {n_nodes} nodes")
    IS_generator = IndependentSetGenerator(
        n_nodes=n_nodes, edge_probability=0.25, graph_type="barabasi_albert"
    )

    for i in trange(n_instances, desc=f"Generating {n_nodes} nodes"):
        file_path = f"./instance/train/IS/independent-set-{n_nodes}-{i:04}.lp"
        # Check if the file already exists to avoid overwriting
        if os.path.exists(file_path):
            continue
        # Generate instance
        instance = next(IS_generator)
        # Write out to .lp file
        instance.write_problem(file_path)
