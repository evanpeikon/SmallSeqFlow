def analyze_ppi_network(gene_list, score_threshold=0.7, species='human'):
    def fetch_string_ppi(genes, species):
        """Fetch PPI data from STRING database"""
        base_url = "https://string-db.org/api/tsv/network"
        genes_str = "\n".join(genes)
        params = {'identifiers': genes_str, 'species': species, 'limit': 1}
        response = requests.post(base_url, data=params)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch data from STRING: {response.status_code}")
            return None

    # Fetch and process PPI data
    ppi_data = fetch_string_ppi(gene_list, species)
    if ppi_data is None:
        raise ValueError("No PPI data retrieved. Check your input or STRING database connection.")
    
    # Parse data and filter by score
    ppi_df = pd.read_csv(StringIO(ppi_data), sep="\t")
    ppi_df_filtered = ppi_df[ppi_df['score'] > score_threshold]
    
    # Create network
    G = nx.Graph()
    for _, row in ppi_df_filtered.iterrows():
        G.add_edge(row['preferredName_A'], row['preferredName_B'], weight=row['score'])
    
    # Calculate network metrics
    network_metrics = {'num_nodes': G.number_of_nodes(),'num_edges': G.number_of_edges(),'density': nx.density(G),'num_components': len(list(nx.connected_components(G)))}
    
    # Calculate node centralities
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    clustering_coefficients = nx.clustering(G)
    
    # Get top nodes by different metrics
    top_nodes = {
        'degree': sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
        'betweenness': sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
        'clustering': sorted(clustering_coefficients.items(), key=lambda x: x[1], reverse=True)[:10]}
    
    def create_network_figures():
        fig = plt.figure(figsize=(20, 20))
        # Full network (top left)
        plt.subplot(2, 2, 1)
        nx.draw_networkx(G, node_size=50, with_labels=True, font_size=8, width=1, alpha=0.7)
        plt.title('Full PPI Network', pad=20, size=14)
        
        # Node degree distribution (top right)
        plt.subplot(2, 2, 2)
        degrees = [d for n, d in G.degree()]
        plt.hist(degrees, bins=max(10, max(degrees)), alpha=0.7)
        plt.xlabel('Node Degree')
        plt.ylabel('Frequency')
        plt.title('Node Degree Distribution', pad=20, size=14)
        
        # Top 20 nodes by degree centrality (bottom left)
        plt.subplot(2, 2, 3)
        top_degree_nodes = [node for node, _ in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]]
        subgraph_degree = G.subgraph(top_degree_nodes)
        nx.draw_networkx(subgraph_degree, node_size=500, node_color='skyblue', with_labels=True, font_size=10)
        plt.title("Top 20 Nodes by Degree Centrality", pad=20, size=14)
        
        # Top 20 nodes by betweenness centrality (bottom right)
        plt.subplot(2, 2, 4)
        top_betweenness_nodes = [node for node, _ in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]]
        subgraph_betweenness = G.subgraph(top_betweenness_nodes)
        nx.draw_networkx(subgraph_betweenness, node_size=500, node_color='lightgreen', with_labels=True, font_size=10)
        plt.title("Top 20 Nodes by Betweenness Centrality", pad=20, size=14)
        plt.tight_layout()
        return fig
    
    # Create visualization
    figure = create_network_figures()
    
    # Return results
    results = {'network': G,'metrics': network_metrics,'top_nodes': top_nodes,'figure': figure}
    return results

def print_ppi_results(results):
    # Print network metrics
    print(f"Number of nodes: {results['metrics']['num_nodes']} "
          f"Number of edges: {results['metrics']['num_edges']} "
          f"Network density: {results['metrics']['density']:.3f} "
          f"Number of connected components: {results['metrics']['num_components']}")
    
    # Print top nodes
    print("\nTop 10 nodes by degree centrality:")
    print([node for node, _ in results['top_nodes']['degree']])       
    print("\nTop 10 nodes by betweenness centrality:")
    print([node for node, _ in results['top_nodes']['betweenness']])
    print("\nTop 10 nodes by clustering coefficient:")
    print([node for node, _ in results['top_nodes']['clustering']])

# Example usage:
'''
gene_list = DEGs['gene'].tolist()  # Use the `gene` column (ENSEMBL IDs)
results = analyze_ppi_network(gene_list)
print_ppi_results(results)
plt.figure()
plt.show(results['figure'])
'''
