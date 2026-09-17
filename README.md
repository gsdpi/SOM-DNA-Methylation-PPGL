# User-Guided Visual Analytics of Genome-Wide DNA Methylation Data Based on Self-Organizing Maps

Submitted to: *IEEE Transactions on Computational Biology and Bioinformatics*

## Authors

**Ignacio Díaz, José M. Enguita, Abel A. Cuadrado, Diego García,
Sara Roos-Hoefgeest, Tamara Cubiella, Nuria Valdés, María D. Chiara**

(C) Universidad de Oviedo, [GSDPI research group](https://gsdpi.edv.uniovi.es/webpage/ "website")

Contact us by email at [gsdpi@uniovi.es](mailto:gsdpi@uniovi.es)

---

## 📄 Overview

This repository contains the code and resources required to **fully reproduce the experiments, figures, and analyses** associated with the article:

> **User-Guided Visual Analytics of Genome-Wide DNA Methylation Data Based on Self-Organizing Maps**

The goal of this work is to provide an interactive and computationally efficient analytical framework for exploring high‑dimensional **DNA methylation** data using **Self-Organizing Maps (SOMs)**, enabling users to:

- generate *epigenetic portraits* of tumor samples,
- explore gene–CpG associations through activation maps,
- dynamically reorganize samples using conditional projections (PCA, t‑SNE, UMAP),
- compute *influence maps* derived from sparse logistic regression,
- and iteratively guide biomedical discovery through a visual, user-driven workflow.

---

## 🧬 Abstract

DNA methylation is a key epigenetic modification with diagnostic and prognostic relevance across a wide range of diseases, particularly cancer. Modern array-based technologies enable high-throughput quantification of methylation states at hundreds of thousands of CpG sites, yielding high-dimensional datasets that pose significant challenges for exploratory analysis and feature prioritization. Existing visualization tools often lack interactivity, integration with machine learning methods, or flexible mechanisms for dynamic dimensionality reduction and biological interpretation.

This work presents an interactive analytical framework that extends the Self-Organizing Map approach for epigenomic data exploration. Our method introduces metasites—representative prototypes of CpG site clusters—enabling interpretable, real-time visualization and machine learning over reduced feature spaces. Through conditional sample projections (e.g., via PCA, t-SNE, or UMAP), user-driven region selection, and the integration of sparsity-controlled logistic regression, we generate metasite relevance maps that reveal discriminative epigenetic patterns and guide downstream analysis.

The proposed approach supports iterative, visually driven discovery of coregulated modules and disease-associated methylation signatures, offering a powerful and intuitive interface for multidimensional exploration of complex methylation landscapes. Its utility is demonstrated through the analysis of DNA methylation in pheochromocytomas and paragangliomas, focusing on SDHB mutation status and the role of protocadherin gene clusters.

---

## 🧬 Dataset

The dataset used in this work consists of **DNA methylation profiles from 34 patients** diagnosed with pheochromocytomas and paragangliomas (PPGL). The cohort includes:

- **14 SDHB‑mutated samples**
- **19 non‑SDHB‑mutated samples**
- **1 sample pending classification** (depending on metadata source)

The original methylation matrix contains:

- **n = 754,581 CpG sites**
- **m = 34 tumor samples**

To increase biological interpretability and focus on CpG loci most relevant for transcriptional regulation, the analysis is restricted to **promoter-associated CpG sites**, including:

- Transcription Start Sites (TSS)
- First exons (1st exon)
- 5' untranslated regions (5'UTRs)

After filtering, the final dataset includes:

- **n = 259,559 promoter-associated CpG sites**

Before SOM training, β-values undergo **rank normalization** to mitigate saturation effects at the extremes and enhance the SOM’s capacity to distinguish subtle methylation variations across CpGs.

Raw data are **not included** in this repository, only a small filtered subset.  The original data can be obtained from ArrayExpress:
https://www.ebi.ac.uk/biostudies/ArrayExpress/studies/E-MTAB-15178?query=E-MTAB-15178

For using this code with another dataset, users must provide their own IDAT or β-value matrices following the expected structure described in the preprocessing scripts.

---

## 🗺️ Interactive Application

A demo with an interactive app demonstrating these techniques available here:
👉 https://avispe.edv.uniovi.es/epigenomics

## 📚 Citation

If you use this code, please cite the article:

> Díaz, I., Enguita, J. M., Cuadrado, A. A., García, D., Roos-Hoefgeest, S., Cubiella, T., Valdés, N., & Chiara, M. D. (2026). User-Guided Visual Analytics of Genome-Wide DNA Methylation Data Based on Self-Organizing Maps. IEEE Transactions on Computational Biology and Bioinformatics, 23(4), 1734-1746. https://doi.org/10.1109/TCBBIO.2026.3696739

## 📄 License

This project is released under the **MIT License** unless otherwise specified.

## Acknowledgement

This work was supported by the Ministerio de Ciencia e Innovación / Agencia Estatal de Investigación (MCIN/AEI/10.13039/501100011033), grants PID2020-115401GB-I00 and PID2023-151388OB-I00.
