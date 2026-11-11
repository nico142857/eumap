# eumap

UMAP with its nearest-neighbour graph built by a [Faiss](https://github.com/facebookresearch/faiss)
IVF index whose centroids are trained with
[e-means](https://e-means.github.io/) (Bessa et al., 2026).

**Status: experimental.** Whether this is faster than stock UMAP is an open question that the
benchmark in this repository is designed to answer. See [Benchmark](#benchmark) before using it.

## What it is, and what it is not

UMAP has no k-means step: it builds an approximate k-nearest-neighbour graph (NN-descent), turns
it into a fuzzy graph and optimises a layout. e-means is a faster k-means. The only place the two
meet is the nearest-neighbour stage, where a k-means-partitioned index (IVF) is an alternative to
NN-descent.

eumap does exactly that and nothing else:

1. train the IVF centroids with e-means (`faiss.contrib.e_means`),
2. build a Faiss `IndexIVFFlat` on those centroids and search it for every point's neighbours,
3. hand the neighbours to an unmodified `umap.UMAP` through its `precomputed_knn` argument.

It is a thin wrapper around [umap-learn](https://github.com/lmcinnes/umap), not a fork. The graph
construction, layout and `transform` are UMAP's own.

## Install

```bash
pip install git+https://github.com/nico142857/eumap
```

Requires Python 3.12 or newer, `umap-learn >= 0.6.0` and `faiss-cpu >= 1.16.0` (the first release
that ships e-means).

## Usage

```python
from eumap import EUMAP

model = EUMAP(n_neighbors=15, metric="euclidean", random_state=0)
embedding = model.fit_transform(X)        # X: (n_samples, n_features) array
new_points = model.transform(X_new)       # works like umap.UMAP.transform

model.knn_info_    # how the graph was built: method, nlist, nprobe, estimated recall, stage times
model.umap_        # the fitted umap.UMAP object
```

Any other keyword is passed to `umap.UMAP` unchanged (`min_dist`, `n_components`, `n_epochs`, ...).

The neighbour graph can also be used on its own:

```python
from eumap import ivf_knn

graph = ivf_knn(X, n_neighbors=15, metric="cosine", target_recall=0.95)
graph.indices, graph.distances   # in the format umap.UMAP(precomputed_knn=...) expects
graph.info
```

### Parameters specific to eumap

| Parameter | Default | Meaning |
|---|---|---|
| `metric` | `"euclidean"` | `"euclidean"` or `"cosine"`. Other metrics raise an error; use `umap.UMAP` for them. |
| `nlist` | about `4 * sqrt(n)` | Number of IVF cells (centroids). |
| `nprobe` | `"auto"` | Cells searched per point. `"auto"` doubles it until the recall measured on a sample of points reaches `target_recall`. |
| `target_recall` | `0.95` | Target for the automatic `nprobe`. |
| `trainer` | `"emeans"` | `"emeans"`, or `"lloyd"` for Faiss' ordinary k-means. |
| `exact_below` | `100000` | Below this many points the graph is computed exactly by brute force. |
| `random_state` | `None` | An int seeds the trainer, the `nprobe` tuning sample and UMAP. |
| `n_jobs` | `-1` | Threads for Faiss and UMAP. |

## Behaviour worth knowing

- **Small data uses exact neighbours.** e-means' automatic batch-size and step-size rules were
  fitted by its authors for at least 100,000 points, and brute force takes seconds below that.
- **Missing neighbours are repaired.** An IVF search can return fewer than `n_neighbors` results
  for a point; those points are re-searched exactly, so the graph handed to UMAP is complete.
- **Euclidean data is centred internally** before indexing. This does not change any distance; it
  avoids float32 cancellation in Faiss when the data sits far from the origin.
- **`transform` works**, through a small adapter around the Faiss index. `update` is not supported.
- Dense input only.

## Benchmark

`benchmarks/` compares, on the same node and with the same threads:

- stock UMAP's NN-descent (both the default and the newer settings of umap-learn 0.6),
- an IVF graph trained with ordinary k-means (to separate "using Faiss IVF" from "using e-means"),
- the same IVF graph trained with e-means,
- exact brute force, as the reference.

It records per-stage wall times, recall against the exact graph, end-to-end time and embedding
quality over three seeds, and compares methods at matched recall. The protocol, including the
criteria for "worth using" and "e-means contributes", was written down before any result existed:
[`benchmarks/PROTOCOL.md`](benchmarks/PROTOCOL.md).

Datasets: a synthetic size sweep (10,000 to 1,000,000 points), MNIST, Fashion-MNIST, Covertype,
SIFT1M and GloVe-100.

### Results

The benchmark is running. This section will be replaced by the measured results.

The expectation recorded in the protocol beforehand, from a single probe run, is a marginal or
negative result on CPU: quantizer training, the part e-means speeds up, is a small share of the
total, and NN-descent is already fast.

### Running it

The scripts assume a Slurm cluster; `slurm/submit.sh` submits each job with a clean environment.

```bash
uv sync --group bench
slurm/submit_all.sh --dry-run core     # show what would be submitted
slurm/submit_all.sh core               # ground truth + benchmark per dataset, at most 3 at once
.venv/bin/python -I benchmarks/aggregate.py   # tables, verdicts and figures under results/
```

A single dataset can be run without Slurm:

```bash
python benchmarks/run_benchmark.py --dataset fashion_mnist --seeds 0 1 2
```

## Tests

```bash
uv sync
uv run pytest
```

## References

- McInnes, Healy and Melville. *UMAP: Uniform Manifold Approximation and Projection for Dimension
  Reduction.* arXiv:1802.03426.
- Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze and Jégou. *e-means: outperforming k-means in
  one-tenth the epochs.* arXiv:2611.09501.
- Douze et al. *The Faiss library.* arXiv:2401.08281.

## Licence

BSD 3-Clause. See [LICENSE](LICENSE).
