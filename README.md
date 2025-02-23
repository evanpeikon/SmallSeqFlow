# 🧬 SmallSeqFlow Overview
Small-sample RNA sequencing experiments present unique analytical challenges yet remain common in many research settings due to cost constraints, limited biological material, or rare experimental conditions. While several established RNA-seq analysis pipelines exist, many are optimized for larger datasets or require significant computational expertise on behalf of the user. To address this gap, I developed a streamlined, Python-based RNA-seq analysis pipeline specifically optimized for small-sample studies that can be easily executed in Jupyter or Google Colab notebooks. 

This pipeline integrates essential RNA-seq analysis steps into a series of modular functions, starting from initial data acquisition through to differential expression analysis and visualization. The workflow begins with a robust data loading function that handles compressed files from common repositories, followed by automated conversion of Ensembl IDs to gene symbols using the MyGene.info service - a crucial step for biological interpretation that is often challenging in public datasets. The pipeline then implements a comprehensive quality control visualization suite that helps researchers identify potential technical biases or batch effects before proceeding with analysis.

The core analytical components of the pipeline are carefully chosen to address the statistical challenges of small-sample studies. The filtering and normalization steps use a DESeq2-inspired approach that works well with limited replication, while the differential expression analysis employs Welch's t-test and Benjamini-Hochberg FDR correction - methods that maintain statistical rigor while acknowledging the limitations of small sample sizes. 

The final visualization module generates presentation-ready figures that capture both global expression patterns and detailed statistical metrics. Following the initial differential expression analysis, the pipeline enables seamless integration with downstream analyses including weighted gene co-expression network analysis (WGCNA) to identify gene modules, mapping of differentially expressed genes to protein-protein interaction networks, and comprehensive functional enrichment analyses to reveal biological pathways and processes (these functionalities are included in the "Extensions" section of this repository). 

By packaging these components into a user-friendly notebook format, this pipeline makes robust RNA-seq analysis accessible to researchers working with limited samples, without requiring extensive computational resources or bioinformatics expertise.

# 🧬 DIY Guide To Using SmallSeqFlow
SmallSeqFlow is designed with user-friendliness in mind, making RNA-seq analysis accessible to researchers regardless of their computational background. The complete pipeline is available as a Jupyter notebook [HERE](https://github.com/evanpeikon/SmallSeqFlow/blob/main/code/SmallSeqFlow.ipynb), containing all the necessary functions and documentation. To analyze your own dataset, download the notebook named SmallSeqFlow.ipynb and import it into Google Collab or Juypter. You only need to modify a few key parameters at the beginning of the notebook: the URL for your dataset, your experimental groups (treatment and control sample names), and any desired threshold values for filtering and differential expression analysis.

In the following sections, I'll demonstrate the practical application of SmallSeqFlow using data from a study investigating the therapeutic potential of amiloride in multiple myeloma (GEO: GSE95077). This step-by-step walkthrough will serve as both a tutorial and a real-world example of how SmallSeqFlow handles RNA-seq data analysis, from initial data loading through to functional interpretation. By following along with this example, you'll learn how to interpret the various quality control visualizations, understand the filtering and normalization steps, and make informed decisions about analysis parameters for your own data. The cells in the notebook are designed to run sequentially, with each step building upon the previous one, making it straightforward to follow the analytical workflow while understanding the biological implications of each step.

## Dependencies
Before using SmallSeqFlow you'll need to import the following libraries:

```python
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
```

## Load, Inspect, and Prepare Data
### Downloading Data
The following function downloads a compressed data file from a specified URL, saves and unzips it locally, and loads it into a pandas DataFrame. It takes a required URL and output filename, along with optional parameters for the data separator (defaulting to tab) and an optional column filter keyword. The function returns a pandas DataFrame containing either all columns from the file or only those columns whose names match the filter keyword if one is provided.

```python
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
```
```python
# Example usage
count_matrix = download_and_load_data(url= "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE95077&format=file&file=GSE95077%5FNormalized%5FCount%5FMatrix%5FJJN3%5FAmiloride%5Fand%5FCTRL%2Etxt%2Egz",
                                   output_filename= "GSE95077_Normalized_Count_Matrix_JJN3_Amiloride_and_CTRL.txt",
                                   column_filter= "")
                                   
count_matrix.head()
```
Which produces the following output:

<img width="1404" alt="Screenshot 2025-02-17 at 12 54 23 PM" src="https://github.com/user-attachments/assets/e9cf5eed-ee80-482c-b69a-c64d421ee126" />

> Note: After executing the code block above, I recommend checking for missing values using the code ```print(count_matrix.isnull().sum())``` and then removing missing values to the extent they exist.

### Converting Ensemble IDs to Gene Names
This function converts Ensembl gene IDs to human-readable gene symbols in a count matrix using the MyGene.info service. It takes a pandas DataFrame with Ensembl IDs as the index and an optional species parameter (defaulting to human). The function cleans Ensembl IDs by removing version numbers, queries the MyGene.info database for corresponding gene symbols, and adds these as a new column while retaining the original count data. Notably, when analyzing GEO (Gene Expression Omnibus) data, it's common to encounter datasets where not all Ensembl IDs can be mapped to gene symbols - this function handles this by logging the mapping success rate, which can often be less than 100% due to outdated or non-standard identifiers in the original data.

```python
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
```
```python
# Example usage
count_matrix_gene_names = convert_ensembl_to_gene_symbols(count_matrix, species='human')
count_matrix_gene_names.head()
```
Which produces the following output:

<img width="1381" alt="Screenshot 2025-02-17 at 12 56 20 PM" src="https://github.com/user-attachments/assets/aeda5217-0cfc-43bd-92c6-91909c8dd3ff" />

### Exploratory Data Analysis

This function creates a comprehensive quality control visualization suite for RNA-seq data that helps researchers identify potential issues or batch effects before proceeding with downstream analysis. It takes a count matrix (with genes as rows and samples as columns) and produces several key visualizations and metrics that serve different analytical purposes. The function returns two figure objects containing multiple plots and a dictionary of QC statistics.

The visualizations and metrics are crucial for several reasons. The total counts per sample plot helps identify potential sequencing depth biases or failed libraries, as samples with unusually low counts may need to be excluded. The log-transformed count distributions (boxplots) reveal potential systematic biases in gene expression levels across samples and can highlight outlier samples with unusual expression patterns.

The correlation heatmap and PCA plot are particularly important for detecting batch effects or unexpected sample groupings - high correlation between samples within a given condition typically indicates good technical replication, while distinct clusters in the PCA plot might reveal either biological differences of interest or unwanted technical variation that needs to be addressed during normalization. The hierarchical clustering dendrogram provides another perspective on sample similarities that can corroborate patterns seen in the PCA and correlation analyses (for this visualization we want to see that samples in the treatment condition group together, and that they are distinct from samples in the control condition). 

These visualizations collectively inform crucial decisions about filtering and normalization strategies. For example, if certain samples show consistently low correlation with others or appear as outliers in multiple plots, they might need to be excluded from further analysis. Similarly, if the PCA plot reveals clear batch effects, these can be addressed through appropriate normalization methods or by including batch as a covariate in differential expression analysis. Understanding these patterns before normalization is critical because different normalization methods make different assumptions about data distribution, and choosing the wrong method could either fail to correct real technical biases or inadvertently remove biological signal of interest.

```python
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
```
```python
main_fig, dendrogram_fig, stats = visualize_rnaseq_qc(count_matrix=count_matrix_gene_names,figure_size=(15, 10))
plt.show()
```
Which produces the following output:
```
RNA-seq Quality Control Metrics:
Total sequencing depth: 190,804,621
Mean reads per sample: 31,800,770
```

<img width="700" alt="Screenshot 2025-02-17 at 12 59 33 PM" src="https://github.com/user-attachments/assets/128c809e-99aa-41f7-9f3d-ff680972e01b" />

## **Quality Control, Filtering, and Normalization**

The next step in our analysis is to filter out genes with low expression levels across all samples, which can introduce noise in the data. By filtering these out, you can make your results more reliable and improve your statistical power, making detecting real biological differences between conditions easier. Additionally, filtering out genes with low expression counts decreases computational load by reducing the number of genes in your dataset, making future downstream analyses faster.

To determine the optimal filtering criteria, we'll use the following function to plot the number of genes retained with different filtering criteria.

```python
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
```
```python
# Example useage
# Drop the Gene Name column from count_matrix_gene_names for counting
countlist_no_name = count_matrix_gene_names.iloc[:, 1:]

# call plot_genes_retained_by_cpm function
plot_genes_retained_by_cpm(countlist_no_name)
```

<img width="650" alt="Screenshot 2025-02-17 at 1 00 54 PM" src="https://github.com/user-attachments/assets/c34efa08-9544-49f4-9046-2d4d9d9fa360" />

Based on the data in the chart above, we'll filter genes with an expression threshold of <0.75 CPM. For many bulk RNA-seq datasets, a CPM threshold of 1 is a common filtering point, but 0.75 is slightly more lenient is justifiable given the distribution of our data. Now, In the code block below, I'll show you how perform basic filtering and normalization.

Now, using the function below we'll performs essential preprocessing of our RNA-seq count data using a DESeq2-inspired approach, combining both filtering and normalization steps. It first filters out lowly expressed genes by requiring a minimum counts-per-million (CPM) threshold in at least a specified number of samples, then normalizes the remaining counts using size factors calculated from geometric means - a method that helps account for differences in sequencing depth and composition biases across samples. The function returns both the normalized count matrix (with gene identifiers preserved) and diagnostic metrics about the filtering and normalization process, making it a crucial step in preparing RNA-seq data for differential expression analysis.

```python
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
```
```python
# Example implementation with diagnostic output
filtered_normalized_count_matrix, stats = filter_normalize(count_matrix_gene_names,  min_cpm=1.0, min_samples=2)
print(stats)
```
Which produces the following output, demonstrating how many genes were removed from the dataset, as well as providing information on the mean size factors and size factor variances, which are metrics that help assess the technical variability in sequencing depth across samples:
```
{'total_genes_initial': 23044, 'genes_post_filtering': 13557, 'size_factors': {'JJ_AMIL_141050_INTER-Str_counts': 1.0027688405136876, 'JJ_AMIL_141056_INTER-Str_counts': 1.0048870898107296, 'JJ_AMIL_141062_INTER-Str_counts': 1.0011096285714167, 'JJ_CTRL_141048_INTER-Str_counts': 1.0071996800856784, 'JJ_CTRL_141054_INTER-Str_counts': 0.999004128898606, 'JJ_CTRL_141060_INTER-Str_counts': 0.9990038091556457}, 'mean_size_factor': 1.002328862839294, 'size_factor_variance': 1.0811867242898648e-05}
```
The mean_size_factor represents the average scaling factor applied to normalize libraries to a common scale. A value close to 1, as is the case in the example above, indicates that samples had similar sequencing depths and required minimal adjustment. If this value were substantially different from 1 (e.g., 0.5 or 2.0), it would suggest more dramatic scaling was needed to make samples comparable. The size_factor_variance, on the other hand, measures how much the individual size factors vary between samples. The very small variance in the example above indicates highly consistent sequencing depth across samples - essentially, all the samples had very similar library sizes. Higher variance (e.g., >0.1) would indicate more variable sequencing depth between samples, which could potentially impact the reliability of downstream analyses.

After completing the filtering and normalization steps, it's crucial to validate the effectiveness of these procedures by re-examining our sample distributions. By visualizing the total counts and log-transformed counts per sample at this stage, we can confirm that we've achieved consistent sequencing depth across all samples. Additionally, revisiting our dimensionality reduction analyses (PCA and hierarchical clustering) at this point serves as a quality check - we should now see even clearer separation between treatment conditions, as the normalization process should have reduced technical noise while preserving true biological signal (these visualizations are not included in the code below).

```python
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
```
Which produces the following output:

<img width="1267" alt="Screenshot 2025-02-17 at 1 07 45 PM" src="https://github.com/user-attachments/assets/c59cf2c0-8db4-40f1-9bfe-68b732e92de4" />

## Differential Expression Analysis

This function performs differential expression analysis on RNA-seq data, comparing treatment versus control conditions for each gene while implementing statistical approaches specifically chosen for small sample sizes. The core of the analysis uses Welch's t-test, which is particularly suitable for RNA-seq studies with limited replication because it doesn't assume equal variances between groups - a common issue in RNA-seq data where expression variability can differ substantially between conditions. The function calculates log2 fold changes with a pseudo-count addition to handle zero values, and returns comprehensive statistics including means, variances, and test results for each gene.

The statistical approach is notable for its use of the Benjamini-Hochberg (BH) false discovery rate (FDR) correction method ('fdr_bh') for multiple testing correction. This method is preferred over more stringent approaches like Bonferroni correction because it provides a better balance between Type I and Type II errors, which is crucial when working with small sample sizes where statistical power is already limited. The function combines both statistical significance (adjusted p-value < alpha) and biological significance (absolute log2 fold change > threshold) to identify differentially expressed genes, a dual-criterion approach that helps control for both statistical and biological relevance.

The implementation includes robust error handling and data validation steps, essential for dealing with the messy reality of RNA-seq data where missing values or computational issues might occur for individual genes. The function outputs both detailed gene-level results and summary statistics, providing a comprehensive view of the differential expression landscape while maintaining transparency about the analysis process. This approach is ideal for small sample sizes because it balances statistical rigor with biological relevance, while using methods (Welch's t-test and BH correction) that are appropriate for limited replication scenarios where traditional assumptions about equal variances and stringent multiple testing corrections might be too conservative.

```python 
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
```
```python 
# Example usage...
treatment_samples = ['JJ_AMIL_141050_INTER-Str_counts', 'JJ_AMIL_141056_INTER-Str_counts', 'JJ_AMIL_141062_INTER-Str_counts'] # column identifiers for treatment condition samples
control_samples = ['JJ_CTRL_141048_INTER-Str_counts', 'JJ_CTRL_141054_INTER-Str_counts', 'JJ_CTRL_141060_INTER-Str_counts'] # column identifiers for control condition samples

welch_results, welch_stats = analyze_differential_expression(
    expression_matrix=filtered_normalized_count_matrix,
    treatment_columns=treatment_samples,
    control_columns=control_samples,
    alpha=0.05,  # default significance threshold
    lfc_threshold=1.0)  # default log2 fold change threshold

# Extract DEGs where 'significant' is True
DEGs = welch_results[welch_results['significant'] == True]
DEGs.head()
```
Which produces the following output:

<img width="1058" alt="Screenshot 2025-02-17 at 1 10 25 PM" src="https://github.com/user-attachments/assets/74641498-c40d-4bc5-977b-8465582301f4" />

The code above performs a differential expression analysis on gene expression data. The final output welch_results contain the results from differential expression analysis across all genes, and the dataframe DEGs contains the differentially expressed genes between the treatment and control samples.

Now, that we've identified our DEGs, it's time to explore this data. The function below creates a comprehensive four-panel visualization suite that effectively communicates different aspects of differential expression analysis results. The choice of plots is carefully designed to provide complementary views of the data, helping researchers identify both global patterns and potential technical biases in their RNA-seq analysis.

The volcano plot (Panel 1) and MA plot (Panel 3) are particularly complementary in their information content. The volcano plot shows the relationship between statistical significance and fold change magnitude, helping identify genes that meet both statistical and biological significance thresholds. The red and blue dashed lines clearly demarcate these significance boundaries. The MA plot, meanwhile, reveals any potential intensity-dependent biases in the fold change estimates by plotting the relationship between mean expression level and fold change. This is crucial for quality control as it can reveal systematic biases that might need to be addressed in the normalization step - ideally, the plot should show a relatively symmetric distribution around the zero fold-change line (red dashed line) across all expression levels.

The distribution plots (Panels 2 and 4) provide important context about the overall patterns in the data. The fold change distribution helps assess the global magnitude of expression changes and can reveal whether the chosen fold change threshold is appropriate for the dataset. The adjusted p-value distribution helps validate whether the statistical testing is detecting genuine biological signal - we typically expect to see a right-skewed distribution with a peak near zero, indicating the presence of truly differentially expressed genes. A uniform distribution might suggest insufficient power to detect differential expression or lack of biological signal, while an extreme skew (especially with an overwhelming majority of near-zero p-values) could indicate technical artifacts. Together, these visualizations provide a comprehensive view of the differential expression analysis results while helping identify potential technical issues that might affect interpretation.

```python
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
```
```python
# example usage
fig, stats = visualize_differential_expression_matrix(
    results_df=welch_results,          # Complete results from differential expression analysis
    filtered_degs=DEGs,    # Subset of significant DEGs
    expression_matrix=filtered_normalized_count_matrix,
    treatment_columns=treatment_samples,
    control_columns=control_samples)

# Display the plot
plt.show()
```

<img width="854" alt="Screenshot 2025-02-17 at 1 11 32 PM" src="https://github.com/user-attachments/assets/b4a9755e-cc91-45e8-a81e-d752217b21bb" />

With our differentially expressed genes (DEGs) identified, we can now delve into deeper biological interpretation through several complementary approaches. By mapping these DEGs to [protein-protein interaction (PPI) networks](https://github.com/evanpeikon/PPI_Network_Analysis), we can understand how the affected genes interact with each other and identify potential hub genes or regulatory networks that might be central to the biological response. This network-based analysis can reveal functional modules and biological pathways that might not be apparent from examining individual genes in isolation.

Additionally, we can perform [functional enrichment analysis](https://github.com/evanpeikon/functional_enrichment_analysis) using tools like Gene Ontology (GO) terms or pathway databases (such as KEGG or Reactome) to understand the biological processes, molecular functions, and cellular components that are significantly affected in our treatment condition. For more complex insights into gene behavior, [weighted gene co-expression network analysis (WGCNA)](https://github.com/evanpeikon/co_expression_network) can be employed to identify modules of co-expressed genes and their relationships to experimental conditions. These downstream analyses transform our list of DEGs into meaningful biological insights, helping to generate hypotheses about the underlying mechanisms of the observed effects.

# 🧬 Extensions
Building on the core functionalities of SmallSeqFlow, I developed several extensions to enable deeper biological insights from small-sample RNA-seq data (you can get the code for the pipeline extensions [HERE](https://github.com/evanpeikon/SmallSeqFlow/tree/main/code/Pipeline_Extensions)). The co-expression analysis module implements a correlation-based approach that constructs gene networks by calculating pairwise Spearman correlations between genes and applying significance thresholds to identify meaningful gene-gene relationships. This module includes functions for network visualization, hub gene identification, and comparative network analysis between conditions, helping reveal how treatment affects the broader organization of gene regulatory networks even in datasets with limited samples.

To bridge transcriptional changes with protein-level interactions, I created a protein-protein interaction (PPI) network analysis extension that maps differentially expressed genes onto known protein interaction networks using the STRING database. This module includes functions for network construction, community detection, and centrality analysis to identify key proteins that might serve as important regulators or intervention points. The visualization components highlight both global network structure and focused subnetworks of highly connected proteins, making it easier to identify potential therapeutic targets or mechanistic insights.

The functional enrichment analysis extension provides comprehensive pathway and gene set analysis capabilities. This module interfaces with multiple annotation databases (GO, KEGG, Reactome) to identify biological processes, molecular functions, and pathways enriched in differentially expressed genes or network modules. Together, these extensions transform the basic pipeline into a comprehensive toolset for extracting biological meaning from small-sample RNA-seq experiments.

## Gene Co-Expression Network Analysis

This function performs a comparative gene co-expression network analysis between two conditions (treatment and control) in a gene expression dataset. As input, it takes the ```filtered_normalized_count_matrix``` DataFrame containing normalized gene expression data with gene names and sample columns, along with patterns to identify treatment and control samples. For each condition, it creates a correlation network where genes are nodes, and edges represent strong correlations (above a specified threshold, default 0.7) between gene pairs. The correlation is calculated using Spearman's method on the transposed expression data (genes as columns, samples as rows).

The function outputs a comprehensive analysis including: network visualizations (full networks and subnetworks of top 10 hub genes), network metrics (density, clustering, path length, etc.), hub gene analysis (top 10 most connected genes in each condition and their overlap), and distribution plots showing the degree distribution and edge weight distribution in both networks. All these results are returned in a structured dictionary and can be displayed using the accompanying print_analysis_results() function. The visualizations help identify differences in network structure between conditions, while the metrics and hub gene analysis provide quantitative measures of network differences and key genes that might be important in each condition.

```python
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
```
```python
# example usage
results = analyze_gene_coexpression(
    filtered_normalized_count_matrix,
    treatment_pattern="AMIL", # Select columns with "AMIL" in their names (use appropriate keyword for your own data)
    control_pattern="CTRL", # Select columns with "CTRL" in their names (use appropriate keyword for your own data)
    n_genes = 100 # of genes to include in analysis (use n_genes = NONE to include all genes in your dataset)
    )

print_analysis_results(results)

# Display figures (
plt.figure()
plt.show(results['figures']['networks'])
plt.figure()
plt.show(results['figures']['distributions'])
```
Which produces the following outputs:
```
Detailed Network Comparison:

Number of Nodes:
Treatment: 100.000
Control: 100.000

Number of Edges:
Treatment: 1634.000
Control: 1703.000

Average Degree:
Treatment: 32.680
Control: 34.060

Network Density:
Treatment: 0.330
Control: 0.344

Average Clustering:
Treatment: 1.000
Control: 1.000

Average Path Length:
Treatment: Not connected
Control: Not connected

Number of Connected Components:
Treatment: 3.000
Control: 3.000

Top 10 Hub Genes:

Treatment hub genes:
NFYA: 37 connections
STPG1: 37 connections
LAS1L: 37 connections
ENPP4: 37 connections
BAD: 37 connections
MAD1L1: 37 connections
DBNDD1: 37 connections
RBM5: 37 connections
ARF5: 37 connections
SARM1: 37 connections

Control hub genes:
FIRRM: 43 connections
FUCA2: 43 connections
GCLC: 43 connections
NFYA: 43 connections
STPG1: 43 connections
LAS1L: 43 connections
SEMA3F: 43 connections
ANKIB1: 43 connections
KRIT1: 43 connections
LASP1: 43 connections

Hub Gene Analysis:
Common hub genes between networks: 3
Common genes: {'STPG1', 'LAS1L', 'NFYA'}

Unique to treatment: 7
Genes: {'DBNDD1', 'ENPP4', 'ARF5', 'BAD', 'SARM1', 'MAD1L1', 'RBM5'}

Unique to control: 7
Genes: {'LASP1', 'GCLC', 'ANKIB1', 'FUCA2', 'FIRRM', 'SEMA3F', 'KRIT1'}
```

![Unknown](https://github.com/user-attachments/assets/dab0e208-328c-44ee-a405-f3e0f3c92ba5)
![Unknown-1](https://github.com/user-attachments/assets/6e384b53-4518-4656-8296-450f94251b96)

As you can see, the function generates two main sets of visualizations that help understand the network structure and properties in both treatment and control conditions:

The first figure shows a 2x2 panel of network visualizations. The top row displays the full networks for both conditions, where each node represents a gene and each edge represents a strong correlation (>0.7 by default) between two genes. These full networks give an overview of the global connectivity patterns. The bottom row shows focused subnetworks containing only the top 10 most connected genes (hub genes) and their interactions with each other, making it easier to visualize the key players in each network. The node size in the subnetworks is larger and includes labels to clearly identify these important genes. These visualizations help identify whether the treatment condition leads to different connectivity patterns or hub genes compared to control.

The second figure shows two distribution plots that provide quantitative insights into network properties. The left plot shows the cumulative degree distribution for both networks, indicating how node connectivity is distributed (e.g., whether most genes have few connections or if there are many highly connected genes). The right plot shows the distribution of edge weights (correlation strengths), helping understand if the correlations in one condition tend to be stronger or weaker than in the other. Both plots include summary statistics in text boxes, making it easy to compare numerical differences between conditions. Together, these visualizations help identify whether the treatment affects the overall organization of gene co-expression patterns and which genes might be playing central roles in each condition.

## Protein-Protein Interaction (PPI) Network Analysis
This function performs protein-protein interaction (PPI) network analysis using gene lists from differential expression results. It takes as input a list of gene identifiers (ENSEMBL IDs), an optional confidence score threshold (default 0.7) for filtering interactions, and a species identifier (default 'human'). The function queries the STRING database to fetch known protein-protein interactions for the input genes, filtering out low-confidence interactions based on the specified threshold.

The function then constructs a network where proteins are nodes and interactions are edges, then performs several network analyses including calculating basic network metrics (number of nodes, edges, density, connected components) and different centrality measures (degree, betweenness, and clustering coefficients). These measures help identify important proteins in the network - degree centrality identifies highly connected hub proteins, betweenness centrality finds proteins that serve as important bridges between different parts of the network, and clustering coefficients identify proteins that form tight-knit groups.

The function outputs both numerical results and visualizations. The numerical results include network statistics and lists of the top 10 proteins ranked by each centrality measure. The visualizations include four plots: the complete PPI network, subnetworks focusing on the top 20 proteins by degree and betweenness centrality (showing their direct interactions), and a histogram showing the distribution of node degrees in the network. All results are returned in a structured dictionary that can be easily accessed for further analysis or visualization customization.

```python
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
```
```python
# Example usage:
gene_list = DEGs['gene'].tolist()  # Use the `gene` column (ENSEMBL IDs)
results = analyze_ppi_network(gene_list)
print_ppi_results(results)
plt.figure()
plt.show(results['figure'])
```
Which produces the following outputs:
```
Number of nodes: 62 Number of edges: 67 Network density: 0.035 Number of connected components: 18

Top 10 nodes by degree centrality:
['HLA-DRA', 'MT-CO1', 'HLA-DOA', 'HLA-DQA1', 'HLA-DMB', 'HLA-DMA', 'NDUFB1', 'MT-ND6', 'MT-ND3', 'MT-ND4']

Top 10 nodes by betweenness centrality:
['EEF1A1', 'MT-CO1', 'HLA-DRA', 'CENPM', 'CALM3', 'AKAP5', 'TK1', 'CDC45', 'RAPGEF3', 'HLA-DQA1']

Top 10 nodes by clustering coefficient:
['SPC24', 'CENPH', 'CIITA', 'MSMO1', 'MVD', 'DHCR7', 'PTPN22', 'NDUFB1', 'MT-ND6', 'MT-ND3']
```

![Unknown-2](https://github.com/user-attachments/assets/9f08e49a-0023-4e3f-a346-16df27467169)

## Functional Enrichment Analysis
This code performs a comprehensive gene set enrichment analysis using the Enrichr tool through the GSEAPY package. It takes a list of differentially expressed genes and analyzes them against four different databases: Gene Ontology (GO) Biological Processes to understand what biological processes these genes are involved in, GO Molecular Functions to identify what molecular activities these genes perform, GO Cellular Components to determine where in the cell these genes' products are active, and KEGG pathways to understand which biological pathways these genes participate in.

For each of these analyses, the code creates separate result sets (biological_processes, molecular_functions, cellular_components, and pathways) and displays the top results. This helps identify which biological processes, functions, locations, and pathways are statistically overrepresented in the input gene list, giving insights into the biological significance of these genes.

```python
# Define the gene lists for each model (DEGs) here
gene_list = DEGs['Gene_Name'].dropna().astype(str).tolist()

# Perform GO enrichment analysis for Biological Process (BP), Molecular Function (MF), Cellular Components (CC), and pathways 
biological_processes = gp.enrichr(gene_list, gene_sets=['GO_Biological_Process_2018'], organism='human')
biological_processes = biological_processes.results
molecular_functions = gp.enrichr(gene_list, gene_sets=['GO_Molecular_Function_2018'], organism='human')
molecular_functions = molecular_functions.results
cellular_components = gp.enrichr(gene_list, gene_sets=['GO_Cellular_Component_2018'], organism='human')
cellular_components = cellular_components.results
pathways = gp.enrichr(gene_list, gene_sets=['KEGG_2016'], organism='human')
pathways = pathways.results

# View results
biological_processes
molecular_functions
cellular_components
pathways
```
Which produces the following output:

<img width="1351" alt="Screenshot 2025-02-23 at 1 30 19 PM" src="https://github.com/user-attachments/assets/9dd2e37a-3b2e-4fe0-a1c3-52f2b451fabf" />

# 🧬 Contributing and Support
SmallSeqFlow is an open-source project and welcomes contributions from the community. If you encounter issues, have suggestions for improvements, or would like to contribute to the project, feel free to reach out: evanpeikon@gmail.com. 
