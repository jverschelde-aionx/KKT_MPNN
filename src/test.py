from models.policy_encoder import GraphDataset

if __name__ == "__main__":
    # Path from your prompt:
    graph_path = "../data/IS/BG/test/10/IS-10-0000.lp.bg"
    dataset = GraphDataset([graph_path])
    bg = dataset.process_sample(graph_path)
    graph = dataset.get(0)
    print(graph)
