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

<p align="center">
  <img width="750" height="350" alt="image" src="https://github.com/user-attachments/assets/66344691-6a82-4fca-ad27-46d6f0959653" />
</p>


<p align="center">
  <img width="39%" alt="image" src="https://github.com/user-attachments/assets/e7753650-5389-40c8-a0ca-d212170856ae" />
  <img width="36.5%" alt="image" src="https://github.com/user-attachments/assets/8db283f5-0ec8-46dc-a2cc-1763eb308769" />
</p>

## 📊 Model Experiments

To evaluate the overall performance of FSIR in fungal ITS sequence retrieval and classification tasks, we conducted a comparative experiment against the widely adopted BLAST algorithm. As shown in Table 1, FSIR outperforms BLAST in both retrieval efficiency and classification accuracy. The performance gains are particularly significant at fine-grained taxonomic ranks, such as family, genus, and species.

In terms of performance metrics, FSIR demonstrates a distinct advantage. Its Average Response Time (ART) is 0.592 seconds, a 68.02% reduction compared to BLAST’s 1.4728 seconds. Its memory footprint is 229.2 MB, also representing a 16.71% decrease. Regarding classification accuracy, FSIR achieved improvements at all taxonomic levels. Notably, at the fine-grained “species” level, the accuracy increased from 94.33% to 95.35%, a gain of 1.08%.

Overall, compared to the traditional BLAST algorithm, FSIR achieves faster query responses, higher classification accuracy, and lower memory consumption, making it particularly suitable for large-scale fungal ITS sequence retrieval and classification tasks
