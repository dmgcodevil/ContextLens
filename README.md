# A BFS-Based Algorithm for Merging Rows with Selected-Match Gain and Path Penalty

## Abstract

We present a graph driven algorithm for merging partially overlapping rows into a coherent table, designed for scenarios where entities extracted from unstructured or semi-structured data sources need to be consolidated into a unified structure. Each row contains a set of label-value tuples, and the algorithm resolves missing labels by traversing a graph of connected rows based on shared tuples. Our method introduces a custom scoring function with three components: `local strength, selected-match gain, tuple weight`, and `path penalty`, ensuring fine-grained control over the merging process. By leveraging a weighted BFS, we guarantee optimal label assignments under this scoring framework.

This algorithm provides a cost-effective, fine-tunable alternative to Retrieval-Augmented Generation (RAG) for entity consolidation tasks, especially in domains requiring interpretable, deterministic outputs and minimal reliance on pre-trained models. Its two-phase approach—prioritizing "perfect rows" (those fully matching partial merges) before exploring other candidates—achieves efficient and high-quality merges. We include examples, pseudo-code, and a proof of correctness, demonstrating its practical applicability in use cases such as entity resolution, log analysis, and structured data enrichment.

### For more detail information please read the [paper]()


Glossary:

```
R - Row
L - Label
V - Value
T - Tuple (L,V)
B - Bucket
```


Example:

```
R1=[(L1,V1)]
R2=[(L1,V2)]
R3=[(L3,V3), (L4,V4), (L5,V5)]
R4=[(L3,V3), (L4,V5), (L5,V6)]
R5=[(L1,V1), (L3,V3), (L4,V4)]
R6=[(L1,V2), (L3,V3), (L4,V5)]
```

R1 connected to R5 via (L1,V1)
R5 connected to R3 via (L3,V3), (L4,V4)

merging R1,R3,R5 will produce `[(L1,V1), (L3,V3), (L4,V4), (L5, V5)]`

R2 connected to R6 via (L1,V2)
R6 connected to R4 via (L3,V3), (L4,V5)
merging R2,R4,R6 will produce  `[(L1,V2), (L3,V3), (L4,V5), (L5,V6)]`


expected output:

````
R1: [(L1,V1), (L3,V3, (L4,V4), (L5, V5)]
R2: [(L1,V2), (L3,V3), (L4,V5), (L5,V6)]
````



## How to run demo

1. clone the repo
2. run `python3 ./main.py --entity-files ./demo/transactions_entities.json,./demo/log_entities.json,./demo/inventory_entities.json --labels TRANSACTION_ID,TRANSACTION_DATE,TRANSACTION_AMOUNT,USER,USER_ID,PRODUCT_MAKE,PRODUCT_MODEL,PRODUCT_PRICE`
