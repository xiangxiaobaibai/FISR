# FISR
Official implementation of "FSIR: A Fungal Sequence Classification Model Based on Sequence Alignment and HNSW". FSIR synergistically combines the WFA algorithm with HNSW and BWT. It achieves a substantial reduction in retrieval time and memory consumption for large-scale fungal sequence identification.
# FSIR: Fungal Sequence Identification and Retrieval

FSIR is a novel computational model designed to improve the retrieval efficiency and classification accuracy of large-scale fungal Internal Transcribed Spacer (ITS) sequences. By synergistically integrating the Wavefront Alignment (WFA) algorithm with the Hierarchical Navigable Small World (HNSW) graph structure, FSIR achieves substantial reductions in both computational time and memory consumption while maintaining high precision.

## 🌟 Key Features

* **Efficient Preprocessing:** Utilizes the Burrows-Wheeler Transform (BWT) to compress data and optimize sequence indexing. This process significantly reduces the memory footprint for sequence storage and enhances subsequent retrieval speed.
* **Rapid Sequence Alignment:** Implements the WFA algorithm with an affine gap penalty model. WFA dynamically constructs optimal alignment paths using incremental updates, avoiding the redundant computation of traditional full-matrix traversal.
* **High-Speed Retrieval:** Constructs a multi-layered HNSW graph to perform efficient approximate nearest neighbor searches. This structure enables fast global jumps on sparse upper layers and fine-grained local searches on denser lower layers.
* **Superior Performance:** Compared to traditional BLAST, FSIR reduces Average Response Time (ART) by 59.82% and memory consumption by 16.71%. Additionally, it achieves a 95.35% classification accuracy at the species level.

## 📊 Model Architecture

The FSIR framework consists of two core modules.

* **Sequence Alignment Module:** Rapidly aligns variable-length ITS sequences and estimates similarity based on mismatches and gaps. This approach effectively circumvents the substantial memory consumption typically associated with traditional k-mer feature extraction methods.
* **HNSW Construction & Indexing Module:** Employs a top-down insertion strategy based on probability to ensure optimal graph sparsity and density. During retrieval, it leverages this structure for efficient local optimization, significantly reducing unnecessary search steps.

![FSIR Model Architecture](images/architecture.png)

The FSIR framework consists of two core modules...
