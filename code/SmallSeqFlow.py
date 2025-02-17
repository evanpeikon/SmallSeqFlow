!pip install mygene
!pip install gseapy
!pip install networkx
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import mygene
import gseapy as gp
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
import os
import subprocess
import gzip
import requests
import networkx as nx 
from io import StringIO
from sklearn.decomposition import PCA

def download_and_load_data(url, output_filename, sep="\t", column_filter=None):
    # Download the file using wget
    print(f"Downloading {output_filename} from {url}...")
    subprocess.run(["wget", "-O", output_filename + ".gz", url], check=True)

    # Unzip file using gunzip
    print(f"Unzipping {output_filename}.gz...")
    with gzip.open(output_filename + ".gz", "rb") as gz_file:
        with open(output_filename, "wb") as out_file:
            out_file.write(gz_file.read())

    # Load the data into a Pandas dataframe
    print(f"Loading {output_filename} into a pandas DataFrame...")
    df = pd.read_csv(output_filename, sep=sep, index_col=0)

    # Optionally,  filter columns based on keyword
    if column_filter:
        print(f"Filtering columns with keyword '{column_filter}'...")
        filtered_columns = [col for col in df.columns if column_filter in col]
        df = df[filtered_columns]

    # Return pandas data frame
    return df


# Example usage
''' Uncomment for use
count_matrix = download_and_load_data(url= "Your URL Here",
                                   output_filename= "Output filename here",
                                   column_filter= "Optional column filter")
                                   
count_matrix.head()
'''

def convert_ensembl_to_gene_symbols(count_matrix, species='human'):
    try:
        # Create a copy to avoid modifying the original
        count_matrix = count_matrix.copy()

        # Remove version numbers from Ensembl IDs
        cleaned_index = count_matrix.index.str.split('.').str[0]
        count_matrix.index = cleaned_index

        # Initialize MyGeneInfo object and query gene symbols
        mg = mygene.MyGeneInfo()
        ensembl_ids = count_matrix.index.unique().tolist()

        # Query gene information with error handling
        gene_info = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species=species, verbose=False)

        # Convert to DataFrame and clean results
        gene_df = pd.DataFrame(gene_info)
        gene_df = gene_df.dropna(subset=['symbol'])
        gene_df = gene_df.drop_duplicates(subset='query')

        # Map gene symbols to count matrix
        symbol_map = gene_df.set_index('query')['symbol']
        count_matrix['Gene_Name'] = count_matrix.index.map(symbol_map)

        # Reorganize columns with Gene_Name first
        cols = ['Gene_Name'] + [col for col in count_matrix.columns if col != 'Gene_Name']
        count_matrix = count_matrix[cols]

        # Log conversion statistics
        total_genes = len(ensembl_ids)
        mapped_genes = len(gene_df)
        print(f"Successfully mapped {mapped_genes} out of {total_genes} genes ({mapped_genes/total_genes*100:.1f}%)")

        return count_matrix

    except Exception as e:
        raise Exception(f"Error during gene ID conversion: {str(e)}")


# Example usage
'''
count_matrix_gene_names = convert_ensembl_to_gene_symbols(count_matrix, species='human')
count_matrix_gene_names.head()
'''

def visualize_rnaseq_qc(count_matrix, figure_size=(15, 12)):
    # Drop the Gene Name column for counting
    countlist_no_name = count_matrix.iloc[:, 1:]

    # Calculate total counts and log transform
    total_counts = countlist_no_name.sum(axis=0)
    log_counts = countlist_no_name.apply(lambda x: np.log2(x + 1))

    # Create main visualization figure
    fig1, axes = plt.subplots(2, 2, figsize=figure_size)

    # Panel 1: Total counts per sample
    sns.barplot(x=countlist_no_name.columns, y=total_counts, color='skyblue', ax=axes[0,0])
    axes[0,0].set_ylabel('Total Counts')
    axes[0,0].set_title('Total Counts per Sample')
    axes[0,0].tick_params(axis='x', rotation=85)

    # Panel 2: Log transformed counts distribution
    log_counts.boxplot(ax=axes[0,1])
    axes[0,1].set_ylabel('Log2(Counts + 1)')
    axes[0,1].set_title('Log Transformed Counts per Sample')
    axes[0,1].tick_params(axis='x', rotation=85)

    # Panel 3: Sample correlation heatmap
    correlation_matrix = log_counts.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0.5, vmin=0, vmax=1, ax=axes[1,0])
    axes[1,0].set_title('Sample Correlation Matrix')

    # Panel 4: PCA plot
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    pca_result = pca.fit_transform(scaler.fit_transform(log_counts.T))
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=log_counts.columns)
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', s=100, ax=axes[1,1])
    for idx, row in pca_df.iterrows():
        axes[1,1].annotate(idx, (row['PC1'], row['PC2']))
    axes[1,1].set_title(f'PCA Plot\nPC1 ({pca.explained_variance_ratio_[0]:.1%}) vs PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.tight_layout()

    # Create dendrogram figure
    fig2 = plt.figure(figsize=(8, 6))
    h_clustering = linkage(log_counts.T, 'ward')
    dendrogram(h_clustering, labels=countlist_no_name.columns)
    plt.xticks(rotation=90)
    plt.ylabel('Distance')
    plt.title('Sample Clustering Dendrogram')

    # Generate QC metrics
    qc_stats = {
        'total_reads': total_counts.sum(),
        'mean_reads_per_sample': total_counts.mean(),
        'cv_reads': total_counts.std() / total_counts.mean(),
        'min_sample_correlation': correlation_matrix.min().min(),
        'max_sample_correlation': correlation_matrix.max().min(),
        'pc1_variance': pca.explained_variance_ratio_[0],
        'pc2_variance': pca.explained_variance_ratio_[1]}
    print("\nRNA-seq Quality Control Metrics:")
    print(f"Total sequencing depth: {qc_stats['total_reads']:,.0f}")
    print(f"Mean reads per sample: {qc_stats['mean_reads_per_sample']:,.0f}")
    return fig1, fig2, qc_stats

# Example usage
'''
main_fig, dendrogram_fig, stats = visualize_rnaseq_qc(count_matrix=count_matrix_gene_names,figure_size=(15, 10))
plt.show()
'''

# plot the number of genes retained as a function of differnet CPM thresholds
def plot_genes_retained_by_cpm(data, min_samples=2):
    # convert raw counts to CPM to normalize the data
    cpm = data.apply(lambda x: (x / x.sum()) * 1e6) #convert raw counts to CPM to normalize
    # define a range of CPM thresholds to test, from 0 to 5 with increments of 0.1
    thresholds = np.arange(0, 5, 0.1)
    # initialize list to store the # of genes retained for ea/ threshold
    genes_retained = []

    # loop through ea/ threshold value to determine the # of genes retained
    for min_cpm in thresholds:
        # create mask where CPM > min_cpm in at least min_samples samples
        mask = (cpm > min_cpm).sum(axis=1) >= min_samples
        # count # of genes that meet the criteria and append to the list
        genes_retained.append(mask.sum())

    # plot # of genes retained as a function of CPM threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, genes_retained, marker='o', color='green')
    plt.axvline(x=1.0, color='red', linestyle='--', label='CPM = 1')
    plt.xlabel('Threshold (CPM)')
    plt.ylabel('Num Genes Retained')
    plt.legend()
    plt.show()

# Example useage
'''
# Drop the Gene Name column from count_matrix_gene_names for counting
countlist_no_name = count_matrix_gene_names.iloc[:, 1:]

# call plot_genes_retained_by_cpm function
plot_genes_retained_by_cpm(countlist_no_name)
'''

def filter_normalize(data, min_cpm=1.0, min_samples=2):
    # Extract structural components
    gene_names = data.iloc[:, 0]
    raw_counts = data.iloc[:, 1:]

    # Implement DESeq2-style filtering
    lib_sizes = raw_counts.sum(axis=0)
    cpm = raw_counts.div(lib_sizes, axis=1) * 1e6
    mask = (cpm > min_cpm).sum(axis=1) >= min_samples

    # Apply filtration criteria
    filtered_counts = raw_counts[mask]
    filtered_gene_names = gene_names[mask]

    # Calculate geometric means with DESeq2-inspired approach
    log_counts = np.log(filtered_counts.replace(0, np.nan))
    geometric_means = np.exp(log_counts.mean(axis=1))

    # Estimate size factors using DESeq2 methodology
    size_factor_ratios = filtered_counts.div(geometric_means, axis=0)
    size_factors = size_factor_ratios.median(axis=0)

    # Apply normalization transformation
    normalized_counts = filtered_counts.div(size_factors, axis=1)

    # Reconstruct data architecture
    normalized_data = pd.concat([filtered_gene_names, normalized_counts], axis=1)

    # Generate diagnostic metrics
    diagnostics = {'total_genes_initial': len(data),'genes_post_filtering': len(normalized_data),'size_factors': size_factors.to_dict(),'mean_size_factor': size_factors.mean(),'size_factor_variance': size_factors.var()}

    return normalized_data, diagnostics

# Example implementation with diagnostic output
'''
filtered_normalized_count_matrix, stats = filter_normalize(count_matrix_gene_names,  min_cpm=1.0, min_samples=2)
print(stats)
'''

# Plot the distribution of data after normalization
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Total normalized counts per sample
total_counts_normalized = filtered_normalized_count_matrix.iloc[:, 1:].sum(axis=0)  # Exclude gene_name column
axes[0].bar(filtered_normalized_count_matrix.columns[1:], total_counts_normalized, color='lightcoral')
axes[0].set_ylabel('Total Normalized Counts')
axes[0].set_title('Total Counts per Sample (Normalized)')
axes[0].tick_params(axis='x', rotation=85)

# Log-transformed normalized counts per sample
log_normalized_data = filtered_normalized_count_matrix.iloc[:, 1:].apply(lambda x: np.log2(x + 1), axis=0)  # Exclude gene_name column
log_normalized_data.boxplot(ax=axes[1])
axes[1].set_ylabel('Log2(Normalized Counts + 1)')
axes[1].set_title('Log Transformed Counts per Sample (Normalized)')
axes[1].tick_params(axis='x', rotation=85)

plt.tight_layout()
plt.show()

def analyze_differential_expression(expression_matrix, treatment_columns, control_columns,alpha=0.05, lfc_threshold=1.0):
    # Input validation
    if not all(col in expression_matrix.columns for col in treatment_columns + control_columns):
        raise ValueError("Specified columns not found in expression matrix")

    # Initialize results collection
    results = []

    # Perform gene-wise differential expression analysis
    for gene in expression_matrix.index:
        try:
            # Extract and validate group-wise expression values
            treated = pd.to_numeric(expression_matrix.loc[gene, treatment_columns], errors='coerce')
            control = pd.to_numeric(expression_matrix.loc[gene, control_columns], errors='coerce')

            # Remove missing values
            treated = treated.dropna()
            control = control.dropna()

            # Validate sufficient data points
            if treated.empty or control.empty:
                continue

            # Calculate expression statistics
            mean_control = np.mean(control)
            mean_treated = np.mean(treated)

            # Compute fold change with pseudo-count
            log2fc = np.log2((mean_treated + 1) / (mean_control + 1))

            # Perform Welch's t-test (equal_var=False)
            t_stat, p_val = ttest_ind(treated, control, equal_var=False)

            # Compile gene-wise results
            results.append({
                "gene": gene,
                "Gene_Name": expression_matrix.loc[gene, "Gene_Name"] if "Gene_Name" in expression_matrix.columns else gene,
                "log2fc": log2fc,
                "mean_treated": mean_treated,
                "mean_control": mean_control,
                "t_stat": t_stat,
                "p_val": p_val,
                "var_treated": np.var(treated),
                "var_control": np.var(control)})

        except Exception as e:
            print(f"Warning: Error processing gene {gene}: {str(e)}")
            continue

    # Convert to DataFrame and perform quality control
    results_df = pd.DataFrame(results)
    results_df['p_val'] = pd.to_numeric(results_df['p_val'], errors='coerce')
    results_df = results_df.dropna(subset=['p_val'])

    # Apply multiple testing correction
    results_df['p_adj'] = multipletests(results_df['p_val'], method='fdr_bh')[1]

    # Calculate absolute fold change
    results_df['abs_log2fc'] = results_df['log2fc'].abs()

    # Define significance criteria
    results_df['significant'] = (results_df['p_adj'] < alpha) & \
                               (results_df['abs_log2fc'] > lfc_threshold)

    # Generate summary statistics
    summary_stats = {
        'total_genes': len(results_df),
        'significant_genes': results_df['significant'].sum(),
        'up_regulated': sum((results_df['significant']) & (results_df['log2fc'] > 0)),
        'down_regulated': sum((results_df['significant']) & (results_df['log2fc'] < 0)),
        'mean_variance_ratio': np.mean(results_df['var_treated'] / results_df['var_control'])}

    # Sort by statistical significance
    results_df = results_df.sort_values('p_adj')

    print("\nDifferential Expression Analysis Summary:")
    print(f"Total genes analyzed: {summary_stats['total_genes']}")
    print(f"Significant genes: {summary_stats['significant_genes']}")
    print(f"Up-regulated: {summary_stats['up_regulated']}")
    print(f"Down-regulated: {summary_stats['down_regulated']}")
    print(f"Mean variance ratio (treated/control): {summary_stats['mean_variance_ratio']:.2f}")

    return results_df, summary_stats

# Example usage...
'''
treatment_samples = [...] # column identifiers for treatment condition samples
control_samples = [...] # column identifiers for control condition samples

welch_results, welch_stats = analyze_differential_expression(
    expression_matrix=filtered_normalized_count_matrix,
    treatment_columns=treatment_samples,
    control_columns=control_samples,
    alpha=0.05,  # default significance threshold
    lfc_threshold=1.0)  # default log2 fold change threshold

# Extract DEGs where 'significant' is True
DEGs = welch_results[welch_results['significant'] == True]
DEGs.head()
'''

def visualize_differential_expression_matrix(results_df, filtered_degs, expression_matrix, treatment_columns, control_columns, p_adj_threshold=0.05, abs_log2fc_threshold=1.0, figure_size=(10, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    scatter_params = {'alpha': 0.8,'edgecolor': None,'palette': 'viridis'}

    # Panel 1: Global Expression Landscape (Volcano Plot)
    sns.scatterplot(data=results_df, x='log2fc', y='p_adj', hue='log2fc',ax=axes[0,0], **scatter_params)
    axes[0,0].axhline(y=p_adj_threshold, color='red', linestyle='--', linewidth=1)
    axes[0,0].axvline(x=abs_log2fc_threshold, color='blue', linestyle='--', linewidth=1)
    axes[0,0].axvline(x=-abs_log2fc_threshold, color='blue', linestyle='--', linewidth=1)
    axes[0,0].set_xlabel('log2 Fold Change')
    axes[0,0].set_ylabel('Adjusted P-value')
    axes[0,0].set_title('Global Expression Landscape')

    # Panel 2: Fold Change Distribution (All Genes)
    sns.histplot(data=results_df, x='abs_log2fc',bins=50, kde=True,ax=axes[0,1])

    # Add vertical line at fold change threshold
    axes[0,1].axvline(x=abs_log2fc_threshold, color='red', linestyle='--', linewidth=1)

    axes[0,1].set_title('Distribution of Absolute log2FC (All Genes)')
    axes[0,1].set_xlabel('Absolute log2 Fold Change')
    axes[0,1].set_ylabel('Gene Frequency')

    # Panel 3: MA Plot
    results_df['mean_expression'] = np.log2((results_df['mean_treated'] + results_df['mean_control'])/2 + 1)

    sns.scatterplot(data=results_df, x='mean_expression', y='log2fc', hue='significant' if 'significant' in results_df.columns else None, ax=axes[1,0], **scatter_params)
    axes[1,0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1,0].set_title('MA Plot (Mean vs Fold Change)')
    axes[1,0].set_xlabel('Mean Expression (log2)')
    axes[1,0].set_ylabel('log2 Fold Change')

    # Panel 4: Distribution of Adjusted P-values
    sns.histplot(data=results_df,x='p_adj',bins=50, kde=True, ax=axes[1,1])

    # Add vertical line at significance threshold
    axes[1,1].axvline(x=p_adj_threshold, color='red', linestyle='--', linewidth=1)
    axes[1,1].set_title('Distribution of Adjusted P-values')
    axes[1,1].set_xlabel('Adjusted P-value')
    axes[1,1].set_ylabel('Gene Frequency')

    plt.tight_layout()

    # Generate comprehensive analytical metrics
    summary_stats = {
        'total_genes': len(results_df),
        'significant_genes': len(filtered_degs),
        'mean_fold_change_all': results_df['abs_log2fc'].mean(),
        'median_fold_change_all': results_df['abs_log2fc'].median(),
        'max_fold_change': results_df['abs_log2fc'].max(),
        'mean_fold_change_sig': filtered_degs['abs_log2fc'].mean(),
        'median_padj': results_df['p_adj'].median(),
        'genes_below_alpha': sum(results_df['p_adj'] < p_adj_threshold)}

    print("\nComprehensive Expression Analysis Metrics:")
    print(f"Total genes analyzed: {summary_stats['total_genes']}")
    print(f"Significant DEGs identified: {summary_stats['significant_genes']}")
    print(f"Mean absolute log2FC (all genes): {summary_stats['mean_fold_change_all']:.2f}")
    print(f"Mean absolute log2FC (significant): {summary_stats['mean_fold_change_sig']:.2f}")
    print(f"Median adjusted p-value: {summary_stats['median_padj']:.3f}")
    print(f"Genes below significance threshold: {summary_stats['genes_below_alpha']}")
    return fig, summary_stats

# example usage
'''
fig, stats = visualize_differential_expression_matrix(
    results_df=welch_results,          # Complete results from differential expression analysis
    filtered_degs=DEGs,    # Subset of significant DEGs
    expression_matrix=filtered_normalized_count_matrix,
    treatment_columns=treatment_samples,
    control_columns=control_samples)

# Display the plot
plt.show()
'''
