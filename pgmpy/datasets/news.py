from __future__ import annotations

import io
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from pgmpy.base import DAG
from pgmpy.datasets._base import BaseSimulatedDataset


class NewsDataset(BaseSimulatedDataset):
    """Semi-synthetic News dataset for counterfactual inference benchmarking.

    Simulates a media consumer's reading experience of news articles under
    different viewing devices (desktop vs. mobile), following the benchmark
    introduced by Johansson et al. (2016) :cite:p:`johansson_2016`.

    Each unit is a news article represented as word counts from the NY Times
    corpus (300K articles, UCI Bag of Words). The intervention is the viewing
    device (desktop ``t=0`` or mobile ``t=1``). The outcome is the reader's
    simulated experience score.

    The data generation process:

    1. **Topic model**: LDA with :math:`K` topics is fit on the NY Times
       corpus. Each article's topic distribution
       :math:`z(x) \\in \\mathbb{R}^K` is its latent representation.

    2. **Centroids**: Two topic-space centroids are defined:

       - :math:`z_0^c` = average topic distribution of all documents
         (desktop preference)
       - :math:`z_1^c` = topic distribution of a randomly sampled document
         (mobile preference)

    3. **Response surfaces**: The noiseless potential outcomes are:

       .. math::

           \\mu_0(x_i) = C \\cdot z(x_i)^\\top z_0^c

           \\mu_1(x_i) = C \\cdot \\bigl(z(x_i)^\\top z_0^c
                         + z(x_i)^\\top z_1^c\\bigr)

       where :math:`C` is the scaling constant. The Individual Treatment
       Effect is:

       .. math::

           \\text{ITE}(x_i) = \\mu_1(x_i) - \\mu_0(x_i)
                            = C \\cdot z(x_i)^\\top z_1^c

    4. **Treatment assignment**: Assignment is biased toward the preferred
       device via a softmax model:

       .. math::

           p(t=1 \\mid x) = \\frac{\\exp(\\kappa \\cdot z(x)^\\top z_1^c)}
           {\\exp(\\kappa \\cdot z(x)^\\top z_0^c)
            + \\exp(\\kappa \\cdot z(x)^\\top z_1^c)}

       where :math:`\\kappa \\geq 0` controls assignment bias strength.
       :math:`\\kappa = 0` gives completely random assignment.

    5. **Noise**: Standard normal noise :math:`\\epsilon \\sim
       \\mathcal{N}(0, 1)` is added to the factual outcome.

    Parameters
    ----------
    n_samples : int, default 5000
        Number of news articles to sample per replication.
        Must not exceed the number of documents in the corpus (300,000).

    num_topics : int, default 50
        Number of LDA topics. The original paper uses 50 topics. For
        ``num_topics=50``, precomputed topic distributions are loaded
        instantly. For ``num_topics <= 200``, topics are derived from a
        precomputed K=200 model via agglomerative merging (seconds). For
        ``num_topics > 200``, LDA is refit from the raw corpus (minutes).

    top_n_words : int, default 100
        Number of top words per topic used to construct the vocabulary.
        With ``num_topics=50`` and ``top_n_words=100``, this produces
        the paper's standard 3,477-word vocabulary. Changing this value
        alters the number of covariate columns in the output.

    kappa : float, default 10.0
        Strength of treatment assignment bias (:math:`\\kappa`). Higher
        values produce stronger confounding. :math:`\\kappa = 0` gives
        random assignment. The original paper uses :math:`\\kappa = 10`.

    C : float, default 50.0
        Scaling constant for the response surface. The original paper
        uses :math:`C = 50`.

    noise : distribution object, optional
        Any object with a ``.sample(n_samples=...)`` or
        ``.rvs(size=...)`` method (e.g., ``scipy.stats`` or ``skpro``
        distributions). When ``None``, standard normal
        :math:`\\mathcal{N}(0, 1)` noise is used.

    seed : int, optional
        Random seed for reproducible centroid selection, document
        sampling, and noise generation.

    References
    ----------
    - :cite:p:`johansson_2016`
    - :cite:p:`shalit_2017`
    """

    _tags = {
        "name": "news",
        "has_ground_truth": True,
        "is_continuous": True,
    }

    def __init__(
        self,
        n_samples: int = 5000,
        num_topics: int = 50,
        top_n_words: int = 100,
        kappa: float = 10.0,
        C: float = 50.0,
        noise: Any = None,
        seed: int | None = None,
    ):
        if n_samples > 300_000:
            raise ValueError(f"n_samples must not exceed 300,000 (the corpus size), got {n_samples}.")

        self.n_samples = n_samples
        self.num_topics = num_topics
        self.top_n_words = top_n_words
        self.kappa = kappa
        self.C = C
        self.noise = noise
        self.seed = seed

        rng = np.random.default_rng(seed)
        self._rng = rng

        corpus_data = self._load_corpus()

        # Topic distributions
        # K=50 (paper default): load precomputed, instant.
        # K<=200: merge precomputed K=200 topics via agglomerative clustering, seconds.
        # K>200: full LDA refit from raw corpus, minutes (with warning).
        if num_topics == 50:
            self._topic_distributions = corpus_data["topic_distributions_k50"]
            lda_components = corpus_data["lda_components_k50"]
        elif num_topics <= 200:
            self._topic_distributions = self._merge_topics(
                corpus_data["topic_distributions_k200"],
                corpus_data["lda_components_k200"],
                num_topics,
            )
            lda_components = corpus_data["lda_components_k200"]
        else:
            warnings.warn(
                f"num_topics={num_topics} exceeds the precomputed maximum (200). "
                f"Refitting LDA on the full 300K-document corpus — this will take "
                f"several minutes. Set num_topics <= 200 for instant loading.",
                UserWarning,
                stacklevel=2,
            )
            self._topic_distributions, lda_components = self._fit_lda_and_transform(
                corpus_data["word_counts_full"], num_topics, seed
            )

        # --- Vocabulary selection ---
        # Default path (num_topics=50, top_n_words=100) uses precomputed 3,477-word vocab.
        # Any other combination recomputes from LDA components.
        if num_topics == 50 and top_n_words == 100:
            self._vocabulary = corpus_data["vocab_3477"]
            self._covariate_word_counts = corpus_data["word_counts_3477"]
        else:
            self._vocabulary, self._covariate_word_counts, _ = self._select_vocabulary(
                lda_components,
                top_n_words,
                corpus_data["word_counts_full"],
                corpus_data["full_vocab"],
            )
        self._n_covariates = len(self._vocabulary)

        # Sample centroids: z0c = corpus mean, z1c = one random document
        n_docs = self._topic_distributions.shape[0]
        centroid_idx = rng.integers(0, n_docs)
        self._z0c = self._topic_distributions.mean(axis=0)
        self._z1c = self._topic_distributions[centroid_idx]

        # Pre-compute response surfaces (paper formulas, applied to every document)
        z = self._topic_distributions
        self._mu0 = C * (z @ self._z0c)
        self._mu1 = C * (z @ (self._z0c + self._z1c))
        self._ite = self._mu1 - self._mu0

        self._propensity = self._compute_propensity(z, self._z0c, self._z1c, kappa)

    def load_dataframe(self, n_samples: int | None = None) -> pd.DataFrame:
        """Generate one replication of the News semi-synthetic dataset.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for this replication. When ``None``, uses
            the value from ``__init__``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``treatment``, ``y_factual``,
            ``y_cfactual``, ``mu0``, ``mu1``, ``ite``, and
            ``x1`` through ``x{n_covariates}`` (word count features).
        """
        n = self.n_samples if n_samples is None else n_samples

        doc_indices = self._rng.choice(self._topic_distributions.shape[0], size=n, replace=False)

        propensity = self._propensity[doc_indices]
        treatment = self._rng.binomial(1, propensity)

        mu0 = self._mu0[doc_indices]
        mu1 = self._mu1[doc_indices]
        ite = self._ite[doc_indices]

        if self.noise is None:
            noise_vals = self._rng.normal(0, 1, size=n)
        elif hasattr(self.noise, "sample"):
            noise_vals = np.asarray(self.noise.sample(n_samples=n)).flatten()
        elif hasattr(self.noise, "rvs"):
            noise_vals = np.asarray(self.noise.rvs(size=n)).flatten()
        else:
            raise TypeError(f"noise must have a .sample() or .rvs() method, got {type(self.noise).__name__}.")

        y0 = mu0 + noise_vals
        y1 = mu1 + noise_vals

        y_factual = np.where(treatment == 1, y1, y0)
        y_cfactual = np.where(treatment == 1, y0, y1)

        word_counts = self._get_word_counts(doc_indices)
        result = pd.DataFrame(
            word_counts,
            columns=[f"x{i}" for i in range(1, self._n_covariates + 1)],
        )
        result.insert(0, "treatment", treatment)
        result.insert(1, "y_factual", y_factual)
        result.insert(2, "y_cfactual", y_cfactual)
        result.insert(3, "mu0", mu0)
        result.insert(4, "mu1", mu1)
        result.insert(5, "ite", ite)

        return result

    def load_ground_truth(self, **kwargs) -> DAG:
        """Return the ground-truth causal DAG with roles.

        The News dataset has the following causal structure: all
        word-count features are pre-treatment confounders influencing
        both treatment assignment and the outcome. ``treatment`` is the
        binary intervention (desktop vs. mobile). ``y_factual`` is the
        observed outcome.

        Parameters
        ----------
        **kwargs
            Absorbed for call-signature compatibility with
            ``load_dataset()``. See ``BaseSimulatedDataset``.

        Returns
        -------
        DAG
            Causal DAG with roles set for pgmpy prediction estimators.
        """
        covariate_names = [f"x{i}" for i in range(1, self._n_covariates + 1)]
        edges = (
            [(cov, "treatment") for cov in covariate_names]
            + [(cov, "y_factual") for cov in covariate_names]
            + [("treatment", "y_factual")]
        )
        return DAG(
            ebunch=edges,
            roles={
                "exposures": "treatment",
                "outcomes": "y_factual",
                "adjustment": covariate_names,
            },
        )

    @staticmethod
    def _load_corpus() -> dict:
        """Load pre-computed NYT corpus with LDA topic distributions from HuggingFace.

        Returns
        -------
        dict
            Dictionary with keys: ``topic_distributions_k50``,
            ``topic_distributions_k200``, ``lda_components_k50``,
            ``lda_components_k200``, ``vocab_3477``,
            ``word_counts_3477``, ``word_counts_full``, ``full_vocab``.
        """
        raw = NewsDataset._get_raw_data("news/news_nytimes_lda.npz")
        data = np.load(io.BytesIO(raw), allow_pickle=True)

        from scipy import sparse

        word_counts_3477 = sparse.csr_matrix(
            (data["word_counts_3477_data"], data["word_counts_3477_indices"], data["word_counts_3477_indptr"]),
            shape=tuple(data["word_counts_3477_shape"]),
        )
        word_counts_full = sparse.csr_matrix(
            (data["word_counts_full_data"], data["word_counts_full_indices"], data["word_counts_full_indptr"]),
            shape=tuple(data["word_counts_full_shape"]),
        )

        return {
            "topic_distributions_k50": data["topic_distributions_k50"],
            "topic_distributions_k200": data["topic_distributions_k200"],
            "lda_components_k50": data["lda_components_k50"],
            "lda_components_k200": data["lda_components_k200"],
            "vocab_3477": data["vocab_3477"],
            "word_counts_3477": word_counts_3477,
            "word_counts_full": word_counts_full,
            "full_vocab": data["full_vocab"],
        }

    @staticmethod
    def _compute_propensity(z, z0c, z1c, kappa):
        """Compute treatment propensity scores for all documents.

        .. math::

            p(t=1 \\mid x) = \\frac{\\exp(\\kappa \\cdot z(x)^\\top z_1^c)}
            {\\exp(\\kappa \\cdot z(x)^\\top z_0^c)
             + \\exp(\\kappa \\cdot z(x)^\\top z_1^c)}

        Uses ``scipy.special.expit`` for numerical stability at large
        :math:`\\kappa`.

        Parameters
        ----------
        z : np.ndarray of shape (n_docs, num_topics)
            Topic distributions for all documents.
        z0c : np.ndarray of shape (num_topics,)
            Desktop centroid (corpus mean).
        z1c : np.ndarray of shape (num_topics,)
            Mobile centroid (random document).
        kappa : float
            Assignment bias strength.

        Returns
        -------
        np.ndarray of shape (n_docs,)
            Propensity scores p(t=1 | x) for each document.
        """
        return expit(kappa * ((z @ z1c) - (z @ z0c)))

    def _get_word_counts(self, doc_indices):
        """Retrieve bag-of-words features for sampled documents.

        These are the vocabulary-selected covariates exposed to the
        user — a different matrix from the one used to compute topic
        mixtures :math:`z(x)`.

        Parameters
        ----------
        doc_indices : np.ndarray
            Indices of documents to retrieve.

        Returns
        -------
        np.ndarray of shape (n, n_covariates)
            Word count matrix for the selected documents.
        """
        X = self._covariate_word_counts
        if hasattr(X, "toarray"):
            return X[doc_indices].toarray()
        return X[doc_indices]

    @staticmethod
    def _select_vocabulary(lda_components, top_n_words, word_counts_full, full_vocab):
        """Select vocabulary as the union of top_n_words per topic.

        Parameters
        ----------
        lda_components : np.ndarray of shape (num_topics, vocab_size)
            Topic-word distribution matrix from LDA.
        top_n_words : int
            Number of top words per topic to include.
        word_counts_full : sparse matrix or np.ndarray of shape
            (n_docs, vocab_size)
            Full-vocabulary word counts.
        full_vocab : np.ndarray of shape (vocab_size,)
            Full vocabulary word strings.

        Returns
        -------
        vocab_selected : np.ndarray of str
            Selected vocabulary words.
        word_counts_selected : sparse matrix or np.ndarray
            Word counts sliced to the selected vocabulary columns.
        vocab_indices : np.ndarray of int
            Indices into the full vocabulary.
        """
        top_words_per_topic = np.argsort(lda_components, axis=1)[:, -top_n_words:]
        vocab_indices = np.unique(top_words_per_topic.flatten())
        vocab_selected = full_vocab[vocab_indices]
        word_counts_selected = word_counts_full[:, vocab_indices]
        return vocab_selected, word_counts_selected, vocab_indices

    @staticmethod
    def _merge_topics(topic_distributions_k200, lda_components_k200, target_k):
        """Derive a target_k-topic representation by merging K=200 topics.

        Uses agglomerative clustering on cosine similarity of the K=200
        topic-word distributions to identify which topics to merge,
        then sums document-topic weights within each cluster and
        re-normalizes.

        Parameters
        ----------
        topic_distributions_k200 : np.ndarray of shape (n_docs, 200)
            Per-document topic proportions from the K=200 LDA fit.
        lda_components_k200 : np.ndarray of shape (200, vocab_size)
            Topic-word distributions from the K=200 LDA fit.
        target_k : int
            Desired number of topics (must be < 200).

        Returns
        -------
        np.ndarray of shape (n_docs, target_k)
            Merged topic distributions.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize

        normed = normalize(lda_components_k200, norm="l2")
        clustering = AgglomerativeClustering(n_clusters=target_k, metric="cosine", linkage="average")
        labels = clustering.fit_predict(normed)

        merged = np.zeros((topic_distributions_k200.shape[0], target_k))
        for new_k in range(target_k):
            old_topics = np.where(labels == new_k)[0]
            merged[:, new_k] = topic_distributions_k200[:, old_topics].sum(axis=1)

        row_sums = merged.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        merged /= row_sums
        return merged

    @staticmethod
    def _fit_lda_and_transform(word_counts_full, num_topics, seed):
        """Refit LDA when the user asks for num_topics > 200.

        Only reached when a user explicitly requests a topic count
        exceeding the precomputed maximum. Takes minutes on the full
        300K-document corpus.

        Parameters
        ----------
        word_counts_full : sparse matrix of shape (300000, ~102660)
            Full-vocabulary word counts for the entire corpus.
        num_topics : int
            Number of LDA topics to fit.
        seed : int or None
            Passed to LDA's ``random_state`` for reproducibility.

        Returns
        -------
        topic_distributions : np.ndarray of shape (300000, num_topics)
            Topic distributions z(x) for every document.
        lda_components : np.ndarray of shape (num_topics, ~102660)
            Fitted topic-word distributions.
        """
        from sklearn.decomposition import LatentDirichletAllocation

        lda = LatentDirichletAllocation(n_components=num_topics, learning_method="batch", random_state=seed)
        topic_distributions = lda.fit_transform(word_counts_full)
        return topic_distributions, lda.components_
