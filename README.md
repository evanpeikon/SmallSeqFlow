# 🧬 SmallSeqFlow Overview
Small-sample RNA sequencing experiments present unique analytical challenges yet remain common in many research settings due to cost constraints, limited biological material, or rare experimental conditions. While several established RNA-seq analysis pipelines exist, many are optimized for larger datasets or require significant computational expertise on behalf of the user. To address this gap, I developed a streamlined, Python-based RNA-seq analysis pipeline specifically optimized for small-sample studies that can be easily executed in Jupyter or Google Colab notebooks. 

This pipeline integrates essential RNA-seq analysis steps into a series of modular functions, starting from initial data acquisition through to differential expression analysis and visualization. The workflow begins with a robust data loading function that handles compressed files from common repositories, followed by automated conversion of Ensembl IDs to gene symbols using the MyGene.info service - a crucial step for biological interpretation that is often challenging in public datasets. The pipeline then implements a comprehensive quality control visualization suite that helps researchers identify potential technical biases or batch effects before proceeding with analysis.

The core analytical components of the pipeline are carefully chosen to address the statistical challenges of small-sample studies. The filtering and normalization steps use a DESeq2-inspired approach that works well with limited replication, while the differential expression analysis employs Welch's t-test and Benjamini-Hochberg FDR correction - methods that maintain statistical rigor while acknowledging the limitations of small sample sizes. 

The final visualization module generates presentation-ready figures that capture both global expression patterns and detailed statistical metrics. Following the initial differential expression analysis, the pipeline enables seamless integration with downstream analyses including weighted gene co-expression network analysis (WGCNA) to identify gene modules, mapping of differentially expressed genes to protein-protein interaction networks, and comprehensive functional enrichment analyses to reveal biological pathways and processes (note: these additional functionalities have been added to the SmallSeqFlow pipeline as of 2/23/25 and can be found in the section of this repository titled "Extensions"). 

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

# 🧬 Contributing and Support

SmallSeqFlow is an open-source project and welcomes contributions from the community. If you encounter issues, have suggestions for improvements, or would like to contribute to the project, feel free to reach out: evanpeikon@gmail.com. 
