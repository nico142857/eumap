# eumap benchmark protocol

This file was written before any benchmark result existed, and the criteria in it were fixed then.
It must not be edited to fit results. Changes made after the first real result go in the change log
at the end, with the date and the reason.

## Question

Does eUMAP beat stock UMAP on CPU? How much of any difference comes from e-means rather than from
merely using a Faiss IVF index? A negative result is acceptable and is reported as plainly as a
positive one.

## Code

| file | role |
|---|---|
| `benchmarks/datasets.py` | downloads into `data/<name>/raw/`, verifies, caches `data/<name>/X.npy` (+ `y.npy`, `meta.json`) |
| `benchmarks/run_benchmark.py` | runs every arm of one dataset in one process; appends JSONL records |
| `benchmarks/aggregate.py` | reads `results/raw/*.jsonl` and writes `results/summary.csv`, `results/RESULTS.md`, `results/figures/*.png` and the verdict lines |
| `slurm/bench.sbatch` | one dataset on one exclusive 36-core broadwell node |
| `slurm/submit_all.sh` | ground truth, then benchmark, per dataset; at most 3 jobs running at once |

The IVF arms call eumap's own building blocks (`prepare`, `train_centroids`, `build_ivf_index`,
`search_self`), so the benchmark times the package's code path. The end-to-end arms call
`eumap.EUMAP` and `umap.UMAP`.

## Hardware and software

- Slurm partition `normal`, `--constraint=broadwell --exclusive`, 36 CPUs, 360 GB. These are
  Xeon E5-2697 v4 nodes: 2 x 18 cores, no hyperthreading, AVX2, no AVX-512, 386 GB RAM.
- One job per dataset (one per synthetic size group). All its arms run sequentially in one process,
  on one node.
- Every record stores the hostname, CPU model, job id, affinity size, Faiss/numba/BLAS thread
  counts and package versions. Times are compared only between records with the same hostname,
  so never across nodes. A dataset that needs more than one 23.5 h job continues in a chained
  part-2 job; when part 1 starts, it pins the pending part 2 to its own node, and both parts
  hold that node exclusively. If the records of one (dataset, n) still come from more than one
  host (the pin was released, or a manual resubmission ran elsewhere), `aggregate.py` uses only
  the host with the most records for every table and verdict of that (dataset, n), and names the
  excluded records on its verdict lines.
- Software: Python 3.12, umap-learn 0.6.0 (pynndescent 0.7.0 backend, no RINND), faiss-cpu 1.16.0,
  numba 0.68.1, numpy 2.5.4, scikit-learn 1.9.1. The exact versions are in every record.

## Datasets

| name | n x d | metric | labels | source |
|---|---|---|---|---|
| `synthetic` | n in {1e4, 3e4, 1e5, 3e5, 1e6, 3e6} x 128 | euclidean | component id (100) | generated, see below |
| `mnist` | 70,000 x 784 | euclidean | 10 | IDX files (ossci-datasets S3 mirror), MD5-checked |
| `fashion_mnist` | 70,000 x 784 | euclidean | 10 | IDX files (fashion-mnist S3), MD5-checked |
| `covtype` | 581,012 x 54, each column standardised | euclidean | 7 | figshare file used by scikit-learn, SHA256-checked |
| `sift1m` | 1,000,000 x 128 | euclidean | none | ann-benchmarks `sift-128-euclidean.hdf5`, field `train` |
| `glove100` | 1,183,514 x 100 | cosine | none | ann-benchmarks `glove-100-angular.hdf5`, field `train` |
| `gist1m` (extended) | 1,000,000 x 960 | euclidean | none | ann-benchmarks `gist-960-euclidean.hdf5` |
| `deep10m` (extended, `--knn-only`) | 9,990,000 x 96 | cosine | none | ann-benchmarks `deep-image-96-angular.hdf5` |

- **Checks on every load:** the expected shape, float32, C-contiguous, and all values finite.
- **Cosine datasets:** rows with zero norm would be dropped and counted in `meta.json`. GloVe has none.
- **Synthetic data** is a mixture of 100 Gaussian 10-dimensional subspaces in d = 128:
  `x = mu_c + z B_c + 1.0 e`, with `mu_c ~ N(0, I)`, `B_c ~ N(0,1)^(10 x 128)`, `z ~ N(0, I_10)`
  and `e ~ N(0, I_128)`. The noise (sigma = 1, mean scale 1) is chosen so that recall is not
  trivially 1.0. Rows are generated in fixed chunks of 65,536, each with its own seed, so every
  size is a prefix of the 3e6 array.
- **Subsets:** `--n` on a real dataset takes a nested, fixed random subset (seed 0). The core run
  uses every real dataset in full.

## Arms

`k = n_neighbors = 15` everywhere, counting the point itself, as UMAP does. Seeds are 0, 1 and 2.
A seed drives the trainer, the NN-descent `random_state` and the UMAP `random_state` together.

### kNN stage

Each arm's recall is measured against the exact graph.

| arm | what is timed |
|---|---|
| `exact` | `eumap.exact_knn(X, 16)` (Faiss brute force), computed once by the ground-truth job and cached in `data/<dataset>/gt_k16.npz` (`gt_n<n>_k16.npz` for subsets and synthetic sizes). Its time is recorded, but it is the reference, not a competitor. |
| `nnd_modern` | `umap.umap_.nearest_neighbors(X, 15, metric, {}, angular, RandomState(seed), low_memory=True, n_jobs=-1, compatibility_layout=False)` |
| `nnd_legacy` | the same call with `compatibility_layout=True` (UMAP 0.6's default NN-descent settings) |
| `ivf_lloyd` | `train_centroids(trainer="lloyd")`: Faiss IVF defaults, 10 iterations, 256 points per centroid |
| `ivf_lloyd25` | Lloyd, `niter=25`, `max_points_per_centroid=n` (no subsampling; the e-means paper's reference) |
| `ivf_emeans` | `train_centroids(trainer="emeans")`, e-means defaults (5 epochs) |
| `ivf_emeans2` | e-means, `nepochs=2` |

- **Every IVF arm and seed:**
  - `nlist = eumap.default_nlist(n)`.
  - `prepare`, `train_centroids` and `build_ivf_index` each run once and are timed separately
    (`t_prepare`, `t_train`, `t_add`).
  - Then nprobe = 1, 2, 4, ... is swept while recall < 0.999 and nprobe <= min(nlist, 256). Each
    step times `search_self(..., repair=False)` alone (`t_search`). With `repair=False`, missing
    neighbours are left in place, so their fraction is recorded rather than repaired inside the
    timer.
  - Also recorded: the centroid MSE (mean squared distance to the assigned centroid) and the list
    imbalance `nlist * sum(size^2) / n^2`, where 1 means perfectly balanced. Their maximum list
    size relative to the mean and their number of empty lists are recorded too.
- **IVF kNN time** at an operating point is `t_train + t_add + t_search(nprobe)`, as the spec
  defines it. `t_prepare` (dtype and finiteness check, plus the normalised copy for cosine) is
  reported in its own column.
- **NN-descent kNN time** is the single `nearest_neighbors` call. NN-descent output gets its
  self-first fix outside the timer, as bookkeeping.

### End to end and embedding quality

Every arm uses UMAP with `optimizer="adam"`, `compatibility_layout=False`, `n_jobs=-1` and
`random_state=seed`. A seeded run therefore stays multi-threaded.

| arm | what is timed |
|---|---|
| `umap_e2e` | `umap.UMAP(...).fit(X)` |
| `eumap_e2e` | `eumap.EUMAP(..., exact_below=0).fit(X)`, package defaults: e-means, auto nprobe, target recall 0.95 |
| `eumap_lloyd_e2e` | the same with `trainer="lloyd"` |
| `layout_exact` | `umap.UMAP(precomputed_knn=(exact I, exact D)).fit(X)`, the quality ceiling. The exact graph's cost is not included. |

- **Stage split:** `t_e2e` is the whole `fit`. `t_graph` is UMAP's `fuzzy_simplicial_set`, timed
  by wrapping the module global. `t_knn` is UMAP's own `nearest_neighbors` call (wrapped the same
  way) for `umap_e2e`, and `knn_info_["t_total"]` for the eUMAP arms. `t_layout` is
  `t_e2e - t_knn - t_graph`.
- **Graph recall:** the recall of the graph each UMAP fit actually used is recorded too, as
  `graph_recall`.

**Quality metrics:** these are computed on every embedding, untimed.

- **`np15`, neighbourhood preservation at 15:** the mean over all n points of the overlap between
  the 15 non-self 2-D neighbours and the 15 non-self exact high-dimensional neighbours, divided by
  15. The 2-D neighbours are exact, from a k-d tree (`scipy.spatial.cKDTree`) in float64.
- **`trustworthiness`:** `sklearn.manifold.trustworthiness(n_neighbors=15)` in the dataset's
  metric, on one fixed 10,000-row subsample per (dataset, n) (seed 12345). Every arm and seed uses
  the same subsample.
- **`knn_acc`:** 5-fold stratified (shuffle, seed 0), k = 15 kNN-classifier accuracy on the full
  embedding, for datasets with labels.

## Fairness rules

- **One node per dataset:** one job runs every arm sequentially, on one node, in one process.
- **3 seeds, rotated order:** seed i runs the arm list rotated by `floor(i * n_arms / n_seeds)`
  positions. The record keeps the order and the arm's position in it.
- **Thread parity:**
  - `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` all equal
    the allocation.
  - `faiss.omp_set_num_threads(affinity)` is set, and NN-descent and UMAP get `n_jobs=-1`.
  - Every record stores the affinity, Faiss, numba and BLAS thread counts and a `thread_parity` flag.
- **Numba:** the cache is node-local and per job (`NUMBA_CACHE_DIR` under `$TMPDIR`/`/tmp`).
- **Warm-up:** before any timer, every selected arm is run once on the first 5,000 rows of the same
  data, and so is the quality code. The warm-up time is recorded as `t_warmup`.
- **Data handling:**
  - Data is loaded as float32 C-contiguous and touched (a full sum) before timing. `t_load` and
    `t_download` are recorded, not compared.
  - UMAP gets copies of any graph it might mutate.
- **Memory:** peak resident memory is recorded per stage. A background thread samples `VmRSS` from
  `/proc/self/status` every 50 ms. The kernel high-water mark `VmHWM` is also reset at each stage
  start through `/proc/self/clear_refs`, where that is allowed. The larger value is reported.
- **Timers:** `time.perf_counter` wraps exactly the named stage. Recall, quality metrics and
  bookkeeping copies are outside every timed region.
- **Records:**
  - One JSON line per (dataset, n, arm, seed) is appended to `results/raw/<file>.jsonl` and
    fsync'ed. Each line carries a stable `run_id`, `<dataset>|n=<n>|k=15|arm=<arm>|seed=<seed>`.
  - A resubmitted job skips the run_ids already recorded as `ok`.
  - An arm that raises gets a `status: "error"` record and is retried on resubmission.
- **Ground truth:** a separate job computes it first. The benchmark job is submitted with
  `afterok` on that job.
- **Deep10M:** this dataset has no full exact graph (that would take about 30 h). Its recall is
  measured on 10,000 sampled query rows (seed 7) against an exact brute-force search over all rows,
  and it runs in `--knn-only` mode.

## Matched-recall comparison (as specified)

For each NN-descent arm `a` and seed `s`, let `r` be the recall of `a`. Take the IVF operating point
of the same seed with the smallest nprobe whose recall is >= `r`. Then compare the total IVF kNN
time (train + add + search at that nprobe) with the NN-descent time: `ratio = t_ivf / t_nnd`.

**Operational details, fixed now:**

1. **Cap at 0.999.** The sweep stops at recall 0.999, so the matching target is `min(r, 0.999)`.
   A recall difference above 0.999 counts as a tie, and the table flags such rows as `capped`.
2. **No match.** If no swept point reaches the target (the nprobe limit was hit first), the pair
   is `unmatched`. Its ratio is infinite, and IVF fails for that seed. An IVF arm that raised an
   error on that seed is unmatched too. For information only, the report also gives the IVF cost
   at the last swept nprobe divided by the NN-descent time. Search time grows with nprobe, so this
   is a lower bound on the ratio a longer sweep could give. The report says whether the verdict
   would change if each unmatched seed got this lower bound, that is, whether the nprobe cap of
   256 decided it.
3. **Summary.** The per-dataset value is the median ratio over seeds, reported with its min-max.
4. **Faster NN-descent arm.** This is the NN-descent arm with the lower median `t_knn` on that
   dataset.
5. **eUMAP's kNN stage.** This is the `ivf_emeans` arm, the trainer `EUMAP` uses by default.
6. **Grid overshoot (informational).** nprobe is swept in powers of two, so the matched point can
   be almost twice the nprobe that the target needs. Next to each ratio, the report also gives the
   ratio with the search time interpolated linearly in recall between the two swept nprobes that
   bracket the target. The verdicts use only the pre-registered ratio.

## Pre-registered reading of the results

(verbatim from the specification)

Evaluated on the datasets with n >= 5e5 (Covertype, SIFT1M, GloVe-100, synthetic 1e6, and GIST1M if run).

- "eUMAP is worth using" if, on at least 3 of 4 (or 4 of 5): (a) its kNN-stage time at matched
  recall is <= 0.8x the faster NN-descent arm, (b) `eumap_e2e` is <= 0.85x `umap_e2e`, and
  (c) every quality metric is within the seed range of `umap_e2e` or within 0.005 absolute.
- "e-means contributes" if `ivf_emeans` or `ivf_emeans2` trains in <= 0.7x the time of `ivf_lloyd`
  at equal or better matched-recall search cost, and `eumap_e2e` <= 0.95x `eumap_lloyd_e2e`.
  If the IVF arms meet (a)-(c) but e-means adds less than 5% end to end, the conclusion is
  "the Faiss IVF graph helps, e-means does not".
- "Not worth using" if IVF does not beat NN-descent at matched recall on any dataset with
  n <= 1e6, or if quality falls outside seed noise.
- The synthetic sweep reports the n at which IVF overtakes NN-descent, or "none up to 3e6".

### How `aggregate.py` evaluates each clause (fixed before results)

Medians and ranges are taken over seeds. "Seed range" means [min, max] over the seeds of the
reference arm.

- **Evaluation set:** `covtype`, `sift1m`, `glove100`, `synthetic` at n = 1e6, and `gist1m` if it
  has records. That is 4 datasets, or 5 with GIST. Other datasets and sizes get the same verdict
  lines, marked informational.
- **(a)** The median over seeds of `ratio(ivf_emeans, faster NN-descent arm)` is <= 0.8.
- **(b)** `median t_e2e(eumap_e2e) / median t_e2e(umap_e2e)` is <= 0.85. If `eumap_e2e` raised an
  error on any seed (with no later successful retry), (b) fails on that dataset.
- **(c)** For each of `np15`, `trustworthiness` and `knn_acc` (where there are labels), the median
  of `eumap_e2e` is >= min over seeds of `umap_e2e`, or >= the median of `umap_e2e` minus 0.005.
  All three metrics are higher-is-better, so an eUMAP value above the `umap_e2e` range passes:
  only being worse fails. If `eumap_e2e` returned a non-finite embedding, or raised an error, on
  any seed, (c) fails on that dataset. A broken embedding is outside seed noise, not missing
  data. The verdict line also prints each metric's difference relative to the `umap_e2e` median,
  for information. On data where a metric is small (NP@15 is about 0.02 on the synthetic data),
  the 0.005 absolute tolerance is a large relative one, and the line makes that visible.
- **"eUMAP is worth using":** (a), (b) and (c) all hold on at least 3 of the 4 evaluation
  datasets, or at least 4 of 5 when GIST1M ran. When fewer datasets have results, the line says
  how many were evaluated and that the verdict is incomplete.
- **"e-means contributes"** is evaluated per dataset. It holds when both of these hold:
  - For `ivf_emeans` or `ivf_emeans2`:
    - its median `t_train` is <= 0.7 x the median `t_train` of `ivf_lloyd`, and
    - its median search time at the matched recall (the target of (a)) is <= the maximum over seeds
      of `ivf_lloyd`'s search time at that target. Being inside Lloyd's seed range counts as
      "equal". An e-means arm that never reaches the target (unmatched) cannot count as equal or
      better.
  - The median `t_e2e` of `eumap_e2e` is <= 0.95 x that of `eumap_lloyd_e2e`.

  The overall line says on how many evaluation datasets it holds. It prints YES when it holds on
  at least as many evaluation datasets as "worth using" needs: 3 of 4, or 4 of 5 with GIST1M.
- **"The Faiss IVF graph helps, e-means does not"** holds on a dataset when both of these hold:
  - (a)-(c) hold either for the e-means pair (`ivf_emeans`, `eumap_e2e`) or for the Lloyd pair
    (`ivf_lloyd`, `eumap_lloyd_e2e`), and
  - e-means saves less than 5% end to end: `(median t_e2e(eumap_lloyd_e2e) - median t_e2e(eumap_e2e)) / median t_e2e(eumap_lloyd_e2e) < 0.05`.

  The overall line uses the same cross-dataset rule as "e-means contributes" and "worth using":
  YES when it holds on at least 3 of 4 evaluation datasets, or 4 of 5 with GIST1M. Per dataset,
  the two e-means clauses exclude each other: one needs a gain of at least 5%, the other a gain
  below 5%.
- **"Not worth using"** holds if either of these holds:
  - On no dataset with n <= 1e6 does any IVF arm have a median matched-recall ratio < 1 against
    the faster NN-descent arm. The datasets here are every dataset and size with records,
    including MNIST, Fashion-MNIST and the synthetic sizes up to 1e6.
  - (c) fails on any evaluation dataset.

  If this verdict and "worth using" both hold, the report prints both and says so. The conflict
  is not resolved silently.
- **Synthetic crossover:** for each synthetic n, IVF beats NN-descent when the median
  `ratio(ivf_emeans, faster NN-descent arm)` is < 1. The crossover is the smallest n from which
  every larger measured n also beats. Otherwise the line says "none up to <largest n measured>".
  The same line is also given for the best IVF arm at each n. Where a size is unmatched, the line
  also shows the lower bound from point 2 of the matched-recall rules. When those lower bounds
  would move the crossover, a note says that the nprobe cap decided it there.

## Prior

This comes from one probe run on 1M x 128 easy synthetic data with 36 threads. e-means saved about
15 s of quantizer training, against about 22 s of search and about 104 s of layout. Warm NN-descent
with the 0.5.x settings took 8.3 s, at recall 0.9999. A negative or marginal verdict is expected
unless harder data changes this.

## Change log

- 2026-11-11, before any real result existed (results/raw/ was empty; only smoke runs on
  subsets). This entry follows a review of the harness. It clarifies points the text left
  open. No threshold, arm, metric or matching rule changed.
  1. "e-means contributes" and "the Faiss IVF graph helps, e-means does not": the overall lines
     used opposite cross-dataset rules (all datasets vs. any one), and the text gave neither.
     Both now use the "worth using" threshold (3 of 4, or 4 of 5).
  2. (b)/(c): an `eumap_e2e` error or non-finite embedding counted as missing data, which favoured
     eUMAP. It now counts as a failure. In (a), an IVF error on a seed counts as unmatched.
  3. "Same hostname and job" became "same hostname": continuation parts are separate jobs by
     design, on the same exclusive node. Part 2 is now pinned to part 1's node, and aggregate.py
     uses one host per (dataset, n).
  4. Informational outputs added, which the verdicts do not use: the lower bound for unmatched
     pairs and whether the nprobe cap decided a verdict; the ratio with interpolated search
     time; the relative quality differences on the (c) line. The Deep10M figure no longer shows
     the 10k-query exact time as if it were a full exact graph.
  5. Cached ground truth now stores a fingerprint of its input data (shape, metric, generator
     parameters, float64 sum and a BLAKE2b hash of X) and is recomputed when that does not match.
     The sampled ground truth (Deep10M) gets the same direct distance rescoring as
     `eumap.exact_knn`. Its ids were already exact.
- 2026-11-11, still before any real result existed (results/raw/ was empty).
  6. `np15`: the exact 2-D neighbours now come from a k-d tree instead of Faiss brute force. Both
     are exact; brute force is O(n^2) and would have taken about 4.7 h of the synthetic 3e6 job.
     The metric's definition is unchanged.
  7. Submission order. The first submission runs every core dataset except synthetic 3e6 (about
     21 h of an exclusive node). Synthetic 3e6 is submitted afterwards only if, over the synthetic
     sizes 1e5, 3e5 and 1e6, the matched-recall ratio of IVF kNN time to the faster NN-descent arm
     falls with n and is below 2 at 1e6 (using the lower bound where a pair is unmatched), i.e.
     only if a crossover by 3e6 is plausible. Otherwise the sweep is reported as
     "none up to 1e6" and that limit is stated. The extended datasets (GIST1M, Deep10M) follow
     the same rule: they are run only if IVF is within a factor of 2 of NN-descent on SIFT1M or
     GloVe-100.
