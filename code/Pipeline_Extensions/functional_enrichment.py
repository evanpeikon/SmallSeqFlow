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
