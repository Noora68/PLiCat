
## <b>Data Acquisition and Annotation Pipeline for Disease Mutation Databases</b>
Finally, we named the mutation dataset: disease_merged_35_500_filtered.csv

<div style="text-align: right;"><em><b>by Feitong Dong</b><br>
June 2025</em></div>

## 1. Data Acquisition from Multiple Disease Mutation Databases

### 1.1 ClinVar
- **Download and Extract Data**  
  Download from:  
  [https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/)<br/>
  Locate and download `variant_summary.txt.gz`.  
  In a Jupyter Notebook, read, decompress, and save the file as `ClinVar_data_summary.csv` (7,131,811 records).

- **Filter Lipid-Related Disease Data**  
  Filter data using a gene list related to lipid diseases and save as `ClinVar_lipid.csv` (1,045,883 records).

- **Extract Key Fields**  
  Extract relevant columns and save as `ClinVar_lipid_filtered.csv`.

### 1.2 cBioPortal
- **Study Identifiers:**  
  `ucec_tcga_pan_can_atlas_2018, skcm_tcga_pan_can_atlas_2018, coadread_tcga_pan_can_atlas_2018, luad_tcga_pan_can_atlas_2018, stad_tcga_pan_can_atlas_2018, lusc_tcga_pan_can_atlas_2018, blca_tcga_pan_can_atlas_2018, brca_tcga_pan_can_atlas_2018, hnsc_tcga_pan_can_atlas_2018, cesc_tcga_pan_can_atlas_2018, gbm_tcga_pan_can_atlas_2018, lihc_tcga_pan_can_atlas_2018, ov_tcga_pan_can_atlas_2018, lgg_tcga_pan_can_atlas_2018, esca_tcga_pan_can_atlas_2018, prad_tcga_pan_can_atlas_2018, paad_tcga_pan_can_atlas_2018, kirp_tcga_pan_can_atlas_2018, kirc_tcga_pan_can_atlas_2018, sarc_tcga_pan_can_atlas_2018, thca_tcga_pan_can_atlas_2018, acc_tcga_pan_can_atlas_2018, ucs_tcga_pan_can_atlas_2018, laml_tcga_pan_can_atlas_2018, dlbc_tcga_pan_can_atlas_2018, thym_tcga_pan_can_atlas_2018, meso_tcga_pan_can_atlas_2018, kich_tcga_pan_can_atlas_2018, tgct_tcga_pan_can_atlas_2018, chol_tcga_pan_can_atlas_2018, pcpg_tcga_pan_can_atlas_2018, uvm_tcga_pan_can_atlas_2018, wt_target_2018_pub, all_phase2_target_2018_pub, aml_target_2018_pub, nbl_target_2018_pub, rt_target_2018_pub`

- **Download and Extract Data**  
  Download the tar.gz files for the above Study IDs from [cBioPortal](https://www.cbioportal.org/) and save to `./pathogenic_dataset/cbioportal_downloads`.  
  Use a Jupyter Notebook to decompress and merge all files into a single CSV: `cbioportal_data.csv` (3,532,778 records).

- **Filter Lipid-Related Disease Data**  
  Filter with gene list to get lipid-related diseases (374,223 records), save as `cbioportal_lipid.csv`.

- **Extract Key Fields**  
  Save filtered important columns as `cbioportal_lipid_filtered.csv`.

---

## 2. Convert Genome Coordinates from GRCh37 to GRCh38 Using LiftOver

- **Installation and Execution**  
  Install and run LiftOver on WSL command line (details in `LiftOver.txt`).

- **Data Structure Inspection**  
  Review the output format after LiftOver conversion.

> LiftOver is a genome coordinate conversion tool that translates genomic coordinates or annotation files (e.g., BED, VCF) between different reference genome versions due to coordinate system differences.  
> GRCh37 and GRCh38 are versions of the human genome reference sequences.

---

## 3. Annotate Data Using Ensembl VEP

- **VEP Installation**  
  See installation instructions in `Esembl VEP code.txt`.

- **Convert CSV to VCF for VEP Input**  
  Create input VCF files:  
  - `clinvar_vep_input.vcf`  
  - `cbioportal_vep_input.vcf`

- **Run VEP Annotation**  
  Execute VEP annotation on WSL command line. Output files:  
  - `clinvar_vep_output.txt`  
  - `cbioportal_vep_output.txt`

- **Post-Processing VEP Output**  
  Remove header comment lines and save as CSV:  
  - `clinvar_vep_head.csv`  
  - `cbioportal_vep_head.csv`

---

## 4. Obtain Protein Sequence Mutation Information

- **Download Uniprot_sprot.fasta**  
  [https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz)

- **Clean Synonymous and Non-Protein-Altering Mutations**  
  Remove synonymous mutations, duplicates, intronic/UTR/regulatory region mutations. Save cleaned files:  
  - `clinvar_vep_cleaned.csv`  
  - `cbioportal_vep_cleaned.csv`

- **Remove Duplicates and Extract Sequences**  
  Using `Uniprot_sprot.fasta`, retrieve wild-type and mutant sequences, add columns `"WT_seq"` and `"Mut_Seq"`. Save as:  
  - `clinvar_vep_with_sequences.csv`  
  - `cbioportal_vep_with_sequences.csv`

- **De-duplication by “Amino_acids” and “SWISSPROT”**  
  Extract wild-type and mutant protein sequences, save in columns `"WT_Seq"` and `"Mut_Seq"`.

---

## 5. Data Integration

- **Merge Datasets**  
  Filter out rows with empty `"WT_seq"` and remove duplicate `"WT_seq"` rows.  
  Vertically concatenate the two datasets, add a new column `"Source"`.  
  Save as:  
  - `clinvar_unique_WT_seq.csv`  
  - `cbioportal_unique_WT_seq.csv`  
  - `merged_unique_WT_seq_with_source.csv`

- **Filter Sequence Length**  
  Keep sequences with length between 35 and 500, save as `merged_35_500.csv`.

- **Extract Useful Fields**  
  Save filtered key fields as `merged_35_500_filtered.csv`.

---



