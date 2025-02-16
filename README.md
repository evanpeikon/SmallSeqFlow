# **Downloading Data**

```python
def download_and_load_data(url, output_filename, sep="\t", column_filter=None):
    '''
    Function for retrieving data from a URL and loading it into a pandas DataFrame.

    Parameters:
    - url (str): The URL to download the data from.
    - output_filename (str): The name of the file to save the downloaded data to.
    - coulumn_filter: An optional filtering criteria if you only wish to load certain columns from a  file.

    Returns:
    - pandas.DataFrame: A DataFrame containing the loaded data.

    Notes:
    - This function performs the followijng operations
      1. Download the file using wget
      2. Unzip the file
      3. Load the data into a pandas DataFrame
    '''

    # Import libraries
    import pandas as pd
    import subprocess
    import gzip

    # Download the file using wget
    print(f"Downloading {output_filename} from {url}...")
    subprocess.run(["wget", "-O", output_filename + ".gz", url], check=True)

    # Gunzip the file
    print(f"Unzipping {output_filename}.gz...")
    with gzip.open(output_filename + ".gz", "rb") as gz_file:
        with open(output_filename, "wb") as out_file:
            out_file.write(gz_file.read())

    # Load the data into a pandas DataFrame
    print(f"Loading {output_filename} into a pandas DataFrame...")
    df = pd.read_csv(output_filename, sep=sep, index_col=0)

    # Optional: Filter columns based on the keyword
    if column_filter:
        print(f"Filtering columns with keyword '{column_filter}'...")
        filtered_columns = [col for col in df.columns if column_filter in col]
        df = df[filtered_columns]

    return df
```
```python
# Example usage
url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE95077&format=file&file=GSE95077%5FNormalized%5FCount%5FMatrix%5FJJN3%5FAmiloride%5Fand%5FCTRL%2Etxt%2Egz'
output_filename = "GSE95077_Normalized_Count_Matrix_JJN3_Amiloride_and_CTRL.txt"
column_keyword = ""
countlist = download_and_load_data(url, output_filename, column_filter=column_keyword)

# View first 5 rows of data
countlist.head()
```

# **Ensemble ID to Gene Symbol**
```python
def convert_ensembl_to_gene_symbols(countlist, species='human'):
    """
    Convert Ensembl gene IDs to gene symbols in a count matrix.

    Parameters:
    - countlist : pandas.DataFrame (Count matrix with Ensembl IDs as index)
    - species : str, optional (Species for gene annotation (default: 'human')

    Returns:
    - pandas.DataFrame: Count matrix with added gene symbols and reordered columns

    Notes:
    This function performs the following operations:
      1. Removes version numbers from Ensembl IDs
      2. Queries MyGeneInfo for gene symbols
      3. Handles duplicate and missing entries
      4. Reorganizes the dataframe with gene symbols
    """

    # Import libraries
    !pip install mygene
    import mygene
    import pandas as pd

    try:
        # Create a copy to avoid modifying the original
        countlist = countlist.copy()

        # Remove version numbers from Ensembl IDs
        cleaned_index = countlist.index.str.split('.').str[0]
        countlist.index = cleaned_index

        # Initialize MyGeneInfo object and query gene symbols
        mg = mygene.MyGeneInfo()
        ensembl_ids = countlist.index.unique().tolist()

        # Query gene information with error handling
        gene_info = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species=species, verbose=False)

        # Convert to DataFrame and clean results
        gene_df = pd.DataFrame(gene_info)
        gene_df = gene_df.dropna(subset=['symbol'])
        gene_df = gene_df.drop_duplicates(subset='query')

        # Map gene symbols to count matrix
        symbol_map = gene_df.set_index('query')['symbol']
        countlist['Gene_Name'] = countlist.index.map(symbol_map)

        # Reorganize columns with Gene_Name first
        cols = ['Gene_Name'] + [col for col in countlist.columns if col != 'Gene_Name']
        countlist = countlist[cols]

        # Log conversion statistics
        total_genes = len(ensembl_ids)
        mapped_genes = len(gene_df)
        print(f"Successfully mapped {mapped_genes} out of {total_genes} genes ({mapped_genes/total_genes*100:.1f}%)")

        return countlist

    except Exception as e:
        raise Exception(f"Error during gene ID conversion: {str(e)}")
```
```python
# Example usage:
countlist = convert_ensembl_to_gene_symbols(countlist, species='human')
countlist.head()
```

# **Filtering and Normalization
```python
def filter_normalize(data, min_cpm=1.0, min_samples=2):
    """
    Implements a DESeq2-inspired normalization strategy for RNA-seq count data.

    Key methodological distinctions from canonical DESeq2:
    1. Geometric mean calculation handling
    2. Size factor estimation approach
    3. Filtering paradigm implementation

    Parameters
    ----------
    data : pd.DataFrame
        Count matrix with gene identifiers in first column
    min_cpm : float
        Minimum counts per million threshold for filtering
    min_samples : int
        Minimum number of samples meeting CPM threshold

    Returns
    -------
    pd.DataFrame
        Normalized count matrix with preserved gene identifiers
    """
    # Import libraries
    import numpy as np
    import pandas as pd

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
    diagnostics = {
        'total_genes_initial': len(data),
        'genes_post_filtering': len(normalized_data),
        'size_factors': size_factors.to_dict(),
        'mean_size_factor': size_factors.mean(),
        'size_factor_variance': size_factors.var()
    }

    return normalized_data, diagnostics
```
```python
# Example implementation with diagnostic output
filtered_normalized_count_matrix, stats = filter_normalize(countlist,  min_cpm=1.0, min_samples=2)
print(stats)
```

# **Differential Expression Analysis**
```python
def analyze_differential_expression(expression_matrix, treatment_columns, control_columns,
                                 alpha=0.05, lfc_threshold=1.0):
    """
    Implements a robust differential expression analysis framework utilizing
    parametric statistical testing with multiple comparison correction.

    Parameters
    ----------
    expression_matrix : pd.DataFrame
        Normalized expression matrix with genes as rows and samples as columns
    treatment_columns : list
        Column identifiers for treatment condition samples
    control_columns : list
        Column identifiers for control condition samples
    alpha : float, optional
        Significance threshold for adjusted p-values (default: 0.05)
    lfc_threshold : float, optional
        Log2 fold change threshold for biological significance (default: 1.0)

    Returns
    -------
    pd.DataFrame
        Comprehensive differential expression results including:
        - Gene identifiers
        - Log2 fold changes
        - Statistical metrics (t-statistics, p-values, adjusted p-values)
        - Expression magnitudes
    """

    # Import libraries
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests


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

            # Perform statistical testing
            t_stat, p_val = ttest_ind(treated, control)

            # Compile gene-wise results
            results.append({
                "gene": gene,
                "Gene_Name": expression_matrix.loc[gene, "Gene_Name"] if "Gene_Name" in expression_matrix.columns else gene,
                "log2fc": log2fc,
                "mean_treated": mean_treated,
                "mean_control": mean_control,
                "t_stat": t_stat,
                "p_val": p_val
            })

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
        'down_regulated': sum((results_df['significant']) & (results_df['log2fc'] < 0))
    }

    # Sort by statistical significance
    results_df = results_df.sort_values('p_adj')

    print("\nDifferential Expression Analysis Summary:")
    print(f"Total genes analyzed: {summary_stats['total_genes']}")
    print(f"Significant genes: {summary_stats['significant_genes']}")
    print(f"Up-regulated: {summary_stats['up_regulated']}")
    print(f"Down-regulated: {summary_stats['down_regulated']}")

    return results_df, summary_stats
```
```python
# Example usage:
treatment_samples = ['JJ_AMIL_141050_INTER-Str_counts', 'JJ_AMIL_141056_INTER-Str_counts', 'JJ_AMIL_141062_INTER-Str_counts']
control_samples = ['JJ_CTRL_141048_INTER-Str_counts', 'JJ_CTRL_141054_INTER-Str_counts', 'JJ_CTRL_141060_INTER-Str_counts']
expression_matrix = filtered_normalized_count_matrix
significance_threshold = 0.05
abs_log2FC = 1.0

# Perform differential expression analysis
de_results, stats = analyze_differential_expression(
    expression_matrix=expression_matrix,
    treatment_columns=treatment_samples,
    control_columns=control_samples,
    alpha=significance_threshold,
    lfc_threshold=abs_log2FC)

# Access results
significant_genes = de_results[de_results['significant']]
```


# **Visualization of DEGs**
```python
def visualize_differential_expression_matrix(results_df, filtered_degs, 
                                          expression_matrix,
                                          treatment_columns, 
                                          control_columns,
                                          p_adj_threshold=0.05, 
                                          abs_log2fc_threshold=1.0,
                                          figure_size=(20, 16)):
    """
    Generates sophisticated four-panel visualization suite incorporating global expression
    landscape, fold change distribution, mean-difference relationships, and sample-wise
    correlations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Complete differential expression results
    filtered_degs : pd.DataFrame
        Filtered differential expressed genes
    expression_matrix : pd.DataFrame
        Original expression matrix containing all samples
    treatment_columns : list
        Treatment sample identifiers
    control_columns : list
        Control sample identifiers
    p_adj_threshold : float
        Statistical significance threshold
    abs_log2fc_threshold : float
        Biological significance threshold
    figure_size : tuple
        Dimensions for visualization matrix
        
    Returns
    -------
    matplotlib.figure.Figure
        Complete figure object containing all visualizations
    dict
        Comprehensive analytical metrics
    """
    # Import libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # Configure visualization architecture
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Define aesthetic parameters
    scatter_params = {
        'alpha': 0.8,
        'edgecolor': None,
        'palette': 'viridis'
    }
    
    # Panel 1: Global Expression Landscape (Volcano Plot)
    sns.scatterplot(data=results_df, 
                   x='log2fc', 
                   y='p_adj', 
                   hue='log2fc',
                   ax=axes[0,0], 
                   **scatter_params)
    
    axes[0,0].axhline(y=p_adj_threshold, color='red', linestyle='--', linewidth=1)
    axes[0,0].axvline(x=abs_log2fc_threshold, color='blue', linestyle='--', linewidth=1)
    axes[0,0].axvline(x=-abs_log2fc_threshold, color='blue', linestyle='--', linewidth=1)
    axes[0,0].set_xlabel('log2 Fold Change')
    axes[0,0].set_ylabel('Adjusted P-value')
    axes[0,0].set_title('Global Expression Landscape')
    
    # Panel 2: Fold Change Distribution
    sns.histplot(filtered_degs['abs_log2fc'], 
                bins=50, 
                kde=True,
                ax=axes[0,1])
    
    axes[0,1].set_title('Distribution of Absolute log2FC')
    axes[0,1].set_xlabel('Absolute log2 Fold Change')
    axes[0,1].set_ylabel('Gene Frequency')
    
    # Panel 3: MA Plot
    results_df['mean_expression'] = np.log2((results_df['mean_treated'] + 
                                           results_df['mean_control'])/2 + 1)
    
    sns.scatterplot(data=results_df,
                   x='mean_expression',
                   y='log2fc',
                   hue='significant' if 'significant' in results_df.columns else None,
                   ax=axes[1,0],
                   **scatter_params)
    
    axes[1,0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1,0].set_title('MA Plot (Mean vs Fold Change)')
    axes[1,0].set_xlabel('Mean Expression (log2)')
    axes[1,0].set_ylabel('log2 Fold Change')
    
    # Panel 4: Sample Correlation Heatmap
    sample_columns = treatment_columns + control_columns
    correlation_matrix = expression_matrix[sample_columns].corr()
    
    # Generate correlation heatmap
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0.5,
                vmin=0,
                vmax=1,
                ax=axes[1,1])
    
    axes[1,1].set_title('Sample-wise Correlation Matrix')
    
    # Optimize layout architecture
    plt.tight_layout()
    
    # Generate comprehensive analytical metrics
    summary_stats = {
        'total_genes': len(results_df),
        'significant_genes': len(filtered_degs),
        'mean_fold_change': filtered_degs['abs_log2fc'].mean(),
        'median_fold_change': filtered_degs['abs_log2fc'].median(),
        'max_fold_change': filtered_degs['abs_log2fc'].max(),
        'mean_sample_correlation': correlation_matrix.mean().mean(),
        'min_sample_correlation': correlation_matrix.min().min(),
        'treatment_correlation': correlation_matrix.loc[treatment_columns, 
                                                      treatment_columns].mean().mean(),
        'control_correlation': correlation_matrix.loc[control_columns, 
                                                    control_columns].mean().mean()
    }
    
    print("\nComprehensive Expression Analysis Metrics:")
    print(f"Total genes analyzed: {summary_stats['total_genes']}")
    print(f"Significant DEGs identified: {summary_stats['significant_genes']}")
    print(f"Mean absolute log2FC: {summary_stats['mean_fold_change']:.2f}")
    print(f"Mean sample correlation: {summary_stats['mean_sample_correlation']:.3f}")
    print(f"Treatment group correlation: {summary_stats['treatment_correlation']:.3f}")
    print(f"Control group correlation: {summary_stats['control_correlation']:.3f}")
    
    return fig, summary_stats
```
```python
# Execute comprehensive visualization framework
fig, stats = visualize_differential_expression_matrix(
    results_df=de_results,
    filtered_degs=significant_genes,
    expression_matrix=filtered_normalized_count_matrix,
    treatment_columns=treatment_samples,
    control_columns=control_samples
)

# Display visualization matrix
plt.show()
```
