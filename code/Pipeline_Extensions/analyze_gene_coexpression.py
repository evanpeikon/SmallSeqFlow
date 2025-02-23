def analyze_gene_coexpression(filtered_normalized_countlist, treatment_pattern, control_pattern, n_genes=None, correlation_threshold=0.7):
    # Subset the data to exclude rows with NaN values in the "Gene_Name" column
    filtered_data = filtered_normalized_countlist.dropna(subset=["Gene_Name"])

    # Subset data for treatment and control
    treatment = filtered_data.filter(regex=treatment_pattern)
    control = filtered_data.filter(regex=control_pattern)

    # Ensure the expression data is numeric
    treatment_data_numeric = treatment.apply(pd.to_numeric, errors='coerce')
    control_numeric = control.apply(pd.to_numeric, errors='coerce')

    # Set Gene_Name as the index
    treatment_data_numeric = treatment_data_numeric.set_index(filtered_data["Gene_Name"])
    control_numeric = control_numeric.set_index(filtered_data["Gene_Name"])

    # Select the top n genes
    treatment_data_numeric = treatment_data_numeric.iloc[:n_genes, :]
    control_numeric = control_numeric.iloc[:n_genes, :]

    # Transpose the expression data
    treatment_transposed = treatment_data_numeric.T
    control_transposed = control_numeric.T

    def calculate_correlation_matrix(expression_data):
        return expression_data.corr(method="spearman")

    # Calculate correlation matrices
    treatment_corr_matrix = calculate_correlation_matrix(treatment_transposed)
    control_corr_matrix = calculate_correlation_matrix(control_transposed)

    def create_network(corr_matrix, threshold):
        G = nx.Graph()
        for i, gene1 in enumerate(corr_matrix.index):
            for j, gene2 in enumerate(corr_matrix.columns):
                if i < j:
                    correlation = corr_matrix.iloc[i, j]
                    if abs(correlation) >= threshold:
                        G.add_edge(gene1, gene2, weight=correlation)
        return G

    # Create networks
    treatment_network = create_network(treatment_corr_matrix, correlation_threshold)
    control_network = create_network(control_corr_matrix, correlation_threshold)

    # Create network visualization with subgraphs
    def plot_networks():
        fig = plt.figure(figsize=(15, 12))
        # Plot full networks
        plt.subplot(2, 2, 1)
        pos_treatment = nx.spring_layout(treatment_network, seed=42)
        nx.draw(treatment_network, pos_treatment, with_labels=False, node_size=20, edge_color="lightblue")
        plt.title(f"{treatment_pattern} Full Network")
        plt.subplot(2, 2, 2)
        pos_control = nx.spring_layout(control_network, seed=42)
        nx.draw(control_network, pos_control, with_labels=False, node_size=20, edge_color="lightgreen")
        plt.title(f"{control_pattern} Full Network")

        # Create and plot subgraphs of top 10 nodes
        def get_top_nodes_subgraph(G, n=10):
            # Get degrees and sort nodes
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:n]
            top_node_names = [node for node, _ in top_nodes]
            # Create subgraph
            subgraph = G.subgraph(top_node_names)
            return subgraph

        # Treatment subgraph
        treatment_sub = get_top_nodes_subgraph(treatment_network)
        plt.subplot(2, 2, 3)
        pos_treatment_sub = nx.spring_layout(treatment_sub, seed=42)
        nx.draw(treatment_sub, pos_treatment_sub, with_labels=True, node_size=500, edge_color="lightblue", font_size=8, font_weight='bold')
        plt.title(f"Top 10 {treatment_pattern} Genes Subnetwork")

        # Control subgraph
        control_sub = get_top_nodes_subgraph(control_network)
        plt.subplot(2, 2, 4)
        pos_control_sub = nx.spring_layout(control_sub, seed=42)
        nx.draw(control_sub, pos_control_sub, with_labels=True, node_size=500, edge_color="lightgreen", font_size=8, font_weight='bold')
        plt.title(f"Top 10 {control_pattern} Genes Subnetwork")
        plt.tight_layout()
        return fig

    # Analyze hub genes
    treatment_degrees = dict(treatment_network.degree())
    control_degrees = dict(control_network.degree())
    sorted_treatment_genes = sorted(treatment_degrees.items(), key=lambda x: x[1], reverse=True)
    sorted_control_genes = sorted(control_degrees.items(), key=lambda x: x[1], reverse=True)

    # Calculate network metrics
    density_treatment = nx.density(treatment_network)
    density_control = nx.density(control_network)

    # Analyze hub genes overlap
    hub_genes_treatment = set([gene for gene, degree in sorted_treatment_genes[:10]])
    hub_genes_control = set([gene for gene, degree in sorted_control_genes[:10]])
    common_hub_genes = hub_genes_treatment.intersection(hub_genes_control)
    unique_treatment_hub_genes = hub_genes_treatment - hub_genes_control
    unique_control_hub_genes = hub_genes_control - hub_genes_treatment

    # Create distribution plots
    def plot_network_distributions():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Plot 1: Cumulative Degree Distribution
        treatment_degrees_list = sorted([d for n, d in treatment_network.degree()], reverse=True)
        control_degrees_list = sorted([d for n, d in control_network.degree()], reverse=True)
        treatment_cumfreq = np.arange(1, len(treatment_degrees_list) + 1) / len(treatment_degrees_list)
        control_cumfreq = np.arange(1, len(control_degrees_list) + 1) / len(control_degrees_list)
        ax1.plot(treatment_degrees_list, treatment_cumfreq, 'r-', label=f'{treatment_pattern}', linewidth=2)
        ax1.plot(control_degrees_list, control_cumfreq, 'b-', label=f'{control_pattern}', linewidth=2)
        ax1.set_xlabel('Node Degree')
        ax1.set_ylabel('Cumulative Frequency')
        ax1.set_title('Cumulative Degree Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add statistics to first plot
        treatment_stats = f'{treatment_pattern}:\nMean degree: {np.mean(treatment_degrees_list):.2f}\nMax degree: {max(treatment_degrees_list)}'
        control_stats = f'{control_pattern}:\nMean degree: {np.mean(control_degrees_list):.2f}\nMax degree: {max(control_degrees_list)}'
        ax1.text(0.02, 0.98, treatment_stats, transform=ax1.transAxes,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.text(0.02, 0.78, control_stats, transform=ax1.transAxes,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: Edge Weight Distribution
        treatment_weights = [d['weight'] for (u, v, d) in treatment_network.edges(data=True)]
        control_weights = [d['weight'] for (u, v, d) in control_network.edges(data=True)]
        bins = np.linspace(min(min(treatment_weights, default=0), min(control_weights, default=0)),max(max(treatment_weights, default=1), max(control_weights, default=1)), 20)
        ax2.hist(treatment_weights, bins, alpha=0.5, label=treatment_pattern, color='red')
        ax2.hist(control_weights, bins, alpha=0.5, label=control_pattern, color='blue')
        ax2.set_xlabel('Edge Weight (Correlation Strength)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Edge Weight Distribution')
        ax2.legend()

        # Add statistics to second plot
        treatment_weight_stats = f'{treatment_pattern}:\nMean weight: {np.mean(treatment_weights):.3f}\nMedian weight: {np.median(treatment_weights):.3f}'
        control_weight_stats = f'{control_pattern}:\nMean weight: {np.mean(control_weights):.3f}\nMedian weight: {np.median(control_weights):.3f}'
        ax2.text(0.02, 0.98, treatment_weight_stats, transform=ax2.transAxes,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.text(0.02, 0.78, control_weight_stats, transform=ax2.transAxes,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        return fig

    # Calculate detailed network comparison metrics
    def calculate_network_metrics():
        metrics = {
            'Number of Nodes': lambda g: len(g.nodes()),
            'Number of Edges': lambda g: len(g.edges()),
            'Average Degree': lambda g: sum(dict(g.degree()).values())/len(g),
            'Network Density': lambda g: nx.density(g),
            'Average Clustering': lambda g: nx.average_clustering(g),
            'Average Path Length': lambda g: nx.average_shortest_path_length(g) if nx.is_connected(g) else 'Not connected',
            'Number of Connected Components': lambda g: nx.number_connected_components(g)}

        comparison_results = {}
        for metric_name, metric_func in metrics.items():
            try:
                treatment_value = metric_func(treatment_network)
                control_value = metric_func(control_network)
                comparison_results[metric_name] = {
                    'treatment': treatment_value,
                    'control': control_value}
            except Exception as e:
                comparison_results[metric_name] = {
                    'treatment': f"Error: {str(e)}",
                    'control': f"Error: {str(e)}"}
        return comparison_results

    network_comparison = calculate_network_metrics()

    # Create result dictionary
    results = {'networks': {'treatment': treatment_network,'control': control_network},
        'hub_genes': {'treatment_top10': sorted_treatment_genes[:10],'control_top10': sorted_control_genes[:10],'common': common_hub_genes,'unique_treatment': unique_treatment_hub_genes,'unique_control': unique_control_hub_genes},
        'network_metrics': {'treatment_density': density_treatment,'control_density': density_control,'detailed_comparison': network_comparison},
        'figures': {'networks': plot_networks(),'distributions': plot_network_distributions()}}
    return results

def print_analysis_results(results):
    # Print network comparison metrics
    comparison_metrics = results['network_metrics']['detailed_comparison']
    print("\nDetailed Network Comparison:")
    for metric_name, values in comparison_metrics.items():
        print(f"\n{metric_name}:")
        treatment_value = values['treatment']
        control_value = values['control']
        if isinstance(treatment_value, (int, float)):
            print(f"Treatment: {treatment_value:.3f}")
            print(f"Control: {control_value:.3f}")
        else:
            print(f"Treatment: {treatment_value}")
            print(f"Control: {control_value}")

    # Print hub genes
    print("\nTop 10 Hub Genes:")
    print("\nTreatment hub genes:")
    for gene, degree in results['hub_genes']['treatment_top10']:
        print(f"{gene}: {degree} connections")
    print("\nControl hub genes:")
    for gene, degree in results['hub_genes']['control_top10']:
        print(f"{gene}: {degree} connections")
    print("\nHub Gene Analysis:")
    print(f"Common hub genes between networks: {len(results['hub_genes']['common'])}")
    print("Common genes:", results['hub_genes']['common'])
    print(f"\nUnique to treatment: {len(results['hub_genes']['unique_treatment'])}")
    print("Genes:", results['hub_genes']['unique_treatment'])
    print(f"\nUnique to control: {len(results['hub_genes']['unique_control'])}")
    print("Genes:", results['hub_genes']['unique_control'])

# Example usage
''' #uncomment to use
results = analyze_gene_coexpression(
    filtered_normalized_count_matrix,
    treatment_pattern="AMIL", # Select columns with "AMIL" in their names (use appropriate keyword for your own data)
    control_pattern="CTRL", # Select columns with "CTRL" in their names (use appropriate keyword for your own data)
    n_genes = 100 # of genes to include in analysis
    )

print_analysis_results(results)

# Display figures (
plt.figure()
plt.show(results['figures']['networks'])
plt.figure()
plt.show(results['figures']['distributions'])
'''
