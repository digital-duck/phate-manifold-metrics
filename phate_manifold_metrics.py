""" PHATE MANIFOLD METRICS

Concepts:
- Manifold Similarity (MS): Language-specific geometric structure preservation
- Relational Affinity (RA): Universal relational alignment across languages/models

Usage:
    python phate_manifold_metrics.py --help
    python phate_manifold_metrics.py --models labse,gemma,fasttext --knn 3 --t 10

    
Author: Digital Duck

"""

import numpy as np
import pandas as pd
import phate
import requests
from scipy import sparse
from typing import List, Tuple, Dict, Any, Union
import warnings
import time
import os
import psutil
import click
from pathlib import Path
from sentence_transformers import SentenceTransformer
import fasttext
import hashlib
import gensim.downloader as api
import hashlib
import pickle
from datetime import datetime
import shutil

# Suppress warnings for clean CLI output
warnings.filterwarnings("ignore", message="Building a kNNGraph on data of shape")
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# System Utilities
# ============================================================================

def get_ram_usage():
    """Get current RAM usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def check_ollama_availability():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, [m["name"] for m in models]
        return False, []
    except Exception as e:
        return False, str(e)


# ============================================================================
# PHATE Diffusion Operator Extraction
# ============================================================================

def extract_phate_diffusion_operator(embeddings: np.ndarray, knn: int = 5, t: Union[int, str] = 'auto'):
    """
    Fits PHATE and extracts the diffusion operator (P^t).

    The diffusion operator captures how probability mass flows across the manifold,
    revealing intrinsic geometric structure beyond Euclidean distances.
    """
    # Break symmetry/singularities with low-level noise (critical for small datasets)
    jitter = np.random.normal(0, 1e-9, embeddings.shape)
    stable_embeddings = embeddings + jitter

    phate_op = phate.PHATE(
        knn=knn,
        t=t,
        n_components=2,
        random_state=42,
        n_jobs=-1,
        verbose=0  # Suppress PHATE output
    )
    phate_op.fit_transform(stable_embeddings)

    # Extract transition matrix
    P = getattr(phate_op.graph, 'diff_op', getattr(phate_op.graph, 'P', None))
    if P is None:
        raise AttributeError("Transition matrix not found in PHATE graph")

    t_res = phate_op.t
    return (P ** t_res) if t_res > 1 else P, phate_op


# ============================================================================
# DUAL PHATE METRIC CLASS
# ============================================================================

class PhateMetrics:
    """
    Combines Manifold Similarity (MS) and Relational Affinity (RA)

    MS: Measures similarity of diffusion distances (language-specific structure)
    RA: Measures directional alignment of relational vectors (universal relations)
    """

    def __init__(self, knn: int = 2, t: int = 5):
        self.knn = knn
        self.t = t
        self.phate_op = None
        self.diff_potential = None

    def fit(self, embeddings: np.ndarray):
        """Fit PHATE and extract diffusion operator"""
        P_t, phate_op = extract_phate_diffusion_operator(embeddings, knn=self.knn, t=self.t)
        self.phate_op = phate_op

        # Extract or compute diffusion potential
        if hasattr(self.phate_op.graph, 'diff_potential') and self.phate_op.graph.diff_potential is not None:
            self.diff_potential = self.phate_op.graph.diff_potential
        else:
            # Compute from transition matrix
            P_t_dense = P_t.toarray() if sparse.issparse(P_t) else P_t
            self.diff_potential = -np.log(P_t_dense + 1e-7)

        return self

    def compute_manifold_similarity(self, pairs: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        METRIC 1: Manifold Similarity (MS)

        Measures the similarity of diffusion distances on the manifold.
        Uses Coefficient of Variation (CV) of diffusion distances.

        MS = 1 / (1 + CV)  where CV = std(distances) / mean(distances)

        Returns:
            ms_score: [0, 1], higher = more similar/consistent manifold structure
            cv: Coefficient of variation (lower = better)
            mean_dist: Average diffusion distance
            std_dist: Standard deviation of distances
        """
        diff_dists = [
            np.linalg.norm(self.diff_potential[a] - self.diff_potential[b])
            for a, b in pairs
        ]
        diff_dists = np.array(diff_dists)

        mean_dist = np.mean(diff_dists)
        std_dist = np.std(diff_dists)
        cv = std_dist / mean_dist if mean_dist > 0 else 0
        ms_score = 1.0 / (1.0 + cv)

        return {
            'ms_score': ms_score,
            'cv': cv,
            'mean_dist': mean_dist,
            'std_dist': std_dist
        }

    def compute_relational_affinity(self, pairs: List[Tuple[int, int]], embeddings: np.ndarray) -> Dict[str, float]:
        """
        METRIC 2: Relational Affinity (RA) - YOUR ORIGINAL METRIC

        Measures directional alignment of relational vectors.
        Uses pairwise cosine similarity of normalized difference vectors.

        rel_vec_i = emb(word2_i) - emb(word1_i)
        RA = mean(cosine_similarity(rel_vec_i, rel_vec_j)) for all pairs i,j

        Returns:
            ra_score: [-1, 1], higher = stronger relational alignment
            std_ra: Standard deviation of pairwise RA scores
            min_ra: Minimum pairwise RA
            max_ra: Maximum pairwise RA
        """
        # Compute relational vectors
        rel_vectors = [embeddings[b] - embeddings[a] for a, b in pairs]

        # L2 normalize (critical for cosine similarity)
        rel_vectors = [v / (np.linalg.norm(v) + 1e-9) for v in rel_vectors]

        # Compute pairwise cosine similarities
        ra_scores = []
        for i in range(len(rel_vectors)):
            for j in range(i + 1, len(rel_vectors)):
                cosine_sim = np.dot(rel_vectors[i], rel_vectors[j])
                ra_scores.append(cosine_sim)

        if not ra_scores:
            return {'ra_score': 0.0, 'std_ra': 0.0, 'min_ra': 0.0, 'max_ra': 0.0}

        return {
            'ra_score': float(np.mean(ra_scores)),
            'std_ra': float(np.std(ra_scores)),
            'min_ra': float(np.min(ra_scores)),
            'max_ra': float(np.max(ra_scores))
        }

    def compute_all_metrics(self, pairs: List[Tuple[int, int]], embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute both MS and RA metrics"""
        ms_metrics = self.compute_manifold_similarity(pairs)
        ra_metrics = self.compute_relational_affinity(pairs, embeddings)

        return {**ms_metrics, **ra_metrics}


# ============================================================================
# MODEL LOADING FUNCTIONS (Fixed Implementations)
# ============================================================================

def get_ollama_embeddings_fixed(words: List[str], model_name: str = "snowflake-arctic-embed2") -> np.ndarray:
    """
    FIXED: Uses proven Semanscope implementation

    Improvements:
    - Session reuse for performance
    - Proper error reporting (not silent failures)
    - Dimension validation
    """
    session = requests.Session()
    embeddings = []
    failed_words = []

    url = "http://localhost:11434/api/embeddings"

    for word in words:
        try:
            response = session.post(
                url,
                json={"model": model_name, "prompt": word},
                timeout=10  # Increased from 5s
            )

            if response.status_code == 200:
                embedding = response.json().get("embedding")
                if embedding:
                    emb_array = np.array(embedding)
                    # Validate embedding
                    if np.isnan(emb_array).any():
                        failed_words.append(f"{word} (NaN)")
                    elif np.allclose(emb_array, 0.0):
                        failed_words.append(f"{word} (Zero)")
                    else:
                        embeddings.append(emb_array)
                else:
                    failed_words.append(f"{word} (No embedding)")
            else:
                failed_words.append(f"{word} (HTTP {response.status_code})")

        except Exception as e:
            failed_words.append(f"{word} ({type(e).__name__})")

    if failed_words:
        click.secho(f"  ⚠️  Failed: {', '.join(failed_words[:3])}" +
                   (f" + {len(failed_words)-3} more" if len(failed_words) > 3 else ""),
                   fg='yellow')

    if not embeddings:
        raise ValueError(f"All {len(words)} words failed to embed")

    return np.array(embeddings)


def load_fasttext_from_extracted(words: List[str], lang: str = 'en', vocab_dir: Path = None) -> np.ndarray:
    """
    Load FastText embeddings from pre-extracted vocabulary files

    Much faster and smaller than full models:
    - Extracted file: ~120MB (top 100k words)
    - Original model: 6.8GB
    - Instant loading: <1s

    Use extract_fasttext_full_vocab.py to create these files.
    """
    if vocab_dir is None:
        vocab_dir = Path(__file__).parent

    # Try different vocabulary sizes
    vocab_files = [
        vocab_dir / f'fasttext_{lang}_top200k.npz',
        vocab_dir / f'fasttext_{lang}_top100k.npz',
        vocab_dir / f'fasttext_{lang}_top50k.npz',
    ]

    vocab_file = None
    for vf in vocab_files:
        if vf.exists():
            vocab_file = vf
            break

    if vocab_file is None:
        raise FileNotFoundError(
            f"No extracted FastText vocabulary found for {lang}. "
            f"Run: python extract_fasttext_full_vocab.py"
        )

    click.echo(f"  ├─ Loading FastText-{lang.upper()} from extracted vocab ({vocab_file.name})... ", nl=False)
    start = time.time()

    # Load vocabulary
    data = np.load(vocab_file, allow_pickle=True)
    vocab_words = data['words']
    vocab_embeddings = data['embeddings']

    # Create word → index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}

    # Extract embeddings for requested words
    embeddings = []
    oov_words = []

    for word in words:
        if word in word_to_idx:
            idx = word_to_idx[word]
            embeddings.append(vocab_embeddings[idx])
        else:
            oov_words.append(word)
            # Use zero vector for OOV
            embeddings.append(np.zeros(300))

    embeddings = np.array(embeddings)

    click.secho(f"Done ({time.time()-start:.3f}s) ⚡", fg='green')

    if oov_words:
        click.secho(f"  ⚠️  OOV: {', '.join(oov_words[:5])}" +
                   (f" + {len(oov_words)-5} more" if len(oov_words) > 5 else ""), fg='yellow')

    return embeddings


def load_fasttext_multilingual(words: List[str], lang: str = 'en') -> np.ndarray:
    """
    Load multilingual FastText embeddings (Pre-BERT era baseline)

    Uses Facebook's Common Crawl + Wikipedia trained models (2018-2019)
    - English: cc.en.300.bin (~6.8GB)
    - Chinese: cc.zh.300.bin (~7.2GB)

    Download from: https://fasttext.cc/docs/en/crawl-vectors.html
    Place in: ~/.fasttext/models/
    """


    # Map language codes to model files
    model_files = {
        'en': 'cc.en.300.bin',
        'zh': 'cc.zh.300.bin',
        'es': 'cc.es.300.bin',
        'fr': 'cc.fr.300.bin',
        'de': 'cc.de.300.bin',
    }

    if lang not in model_files:
        raise ValueError(f"Language '{lang}' not supported. Available: {list(model_files.keys())}")

    # Model storage location
    model_dir = Path.home() / '.fasttext' / 'models'
    model_path = model_dir / model_files[lang]

    # Check if model exists
    if not model_path.exists():
        click.secho(f"  ⚠️  FastText model not found: {model_path}", fg='red', bold=True)
        click.secho(f"  ℹ️  Download from: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{model_files[lang]}.gz", fg='cyan')
        click.secho(f"  ℹ️  Extract to: {model_dir}/", fg='cyan')
        click.secho(f"  ℹ️  Quick setup:", fg='cyan')
        click.secho(f"      mkdir -p {model_dir}", fg='white')
        click.secho(f"      cd {model_dir}", fg='white')
        click.secho(f"      wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{model_files[lang]}.gz", fg='white')
        click.secho(f"      gunzip {model_files[lang]}.gz", fg='white')
        raise FileNotFoundError(f"FastText model not found: {model_path}")

    # Cache setup - centralized embedding cache
    cache_dir = Path.home() / 'projects' / 'embedding_cache' / 'fasttext'
    cache_dir.mkdir(parents=True, exist_ok=True)


    vocab_key = hashlib.md5(f"{lang}|{'|'.join(sorted(words))}".encode()).hexdigest()
    cache_file = cache_dir / f'fasttext_{lang}_{vocab_key}.npz'

    # Try loading from cache
    if cache_file.exists():
        click.echo(f"  ├─ Loading FastText-{lang.upper()} (from cache)... ", nl=False)
        start = time.time()

        cached = np.load(cache_file)
        embeddings = cached['embeddings']
        oov_words = cached['oov_words'].tolist() if 'oov_words' in cached else []

        click.secho(f"Done ({time.time()-start:.3f}s) ⚡", fg='green')

        if oov_words:
            click.secho(f"  ⚠️  OOV: {', '.join(oov_words[:5])}" +
                       (f" + {len(oov_words)-5} more" if len(oov_words) > 5 else ""), fg='yellow')

        return embeddings

    # Cache miss - load model
    click.echo(f"  ├─ Loading FastText-{lang.upper()} model... ", nl=False)
    start = time.time()

    ft_model = fasttext.load_model(str(model_path))

    click.secho(f"Done ({time.time()-start:.1f}s)", fg='green')
    click.echo("  ├─ Extracting embeddings... ", nl=False)
    extract_start = time.time()

    # Extract embeddings
    embeddings = []
    oov_words = []

    for word in words:
        emb = ft_model.get_word_vector(word)  # FastText always returns a vector (uses subwords)
        embeddings.append(emb)

        # Check if word is truly in vocabulary (not just subword-generated)
        if word not in ft_model.words:
            oov_words.append(word)

    embeddings = np.array(embeddings)

    click.secho(f"Done ({time.time()-extract_start:.1f}s)", fg='green')

    # Save to cache
    click.echo("  ├─ Caching for future runs... ", nl=False)
    np.savez_compressed(cache_file, embeddings=embeddings, oov_words=np.array(oov_words))
    click.secho("Done ✓", fg='green')

    if oov_words:
        click.secho(f"  ⚠️  OOV (subword): {', '.join(oov_words[:5])}" +
                   (f" + {len(oov_words)-5} more" if len(oov_words) > 5 else ""), fg='yellow')

    return embeddings


def load_fasttext_fixed(words: List[str]) -> np.ndarray:
    """
    FIXED: Proper FastText loading with smart caching

    OPTIMIZATION: Cache extracted embeddings for instant subsequent loads
    - First run: ~180s (full model load + extraction)
    - Subsequent runs: <0.5s (load from cache)

    Strategy:
    1. Check cache for this vocabulary
    2. If cache hit: load instantly from .npy file
    3. If cache miss: load FastText, extract, save to cache
    4. Educational value: Shows optimization techniques
    """


    # Create cache directory - centralized embedding cache
    cache_dir = Path.home() / 'projects' / 'embedding_cache' / 'fasttext'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create unique cache key from vocabulary (sorted for consistency)
    vocab_key = hashlib.md5('|'.join(sorted(words)).encode()).hexdigest()
    cache_file = cache_dir / f'fasttext_{vocab_key}.npz'

    # Try loading from cache
    if cache_file.exists():
        click.echo("  ├─ Loading FastText (from cache)... ", nl=False)
        start = time.time()

        cached = np.load(cache_file)
        embeddings = cached['embeddings']
        oov_words = cached['oov_words'].tolist() if 'oov_words' in cached else []

        click.secho(f"Done ({time.time()-start:.3f}s) ⚡", fg='green')

        if oov_words:
            click.secho(f"  ⚠️  OOV: {', '.join(oov_words)}", fg='yellow')

        return embeddings

    # Cache miss - load full model
    click.echo("  ├─ Loading FastText model (first run, will cache)... ", nl=False)
    start = time.time()

    ft_model = api.load('fasttext-wiki-news-subwords-300')

    click.secho(f"Done ({time.time()-start:.1f}s)", fg='green')
    click.echo("  ├─ Extracting embeddings... ", nl=False)
    extract_start = time.time()

    # Extract embeddings for our vocabulary only
    embeddings = []
    oov_words = []

    for word in words:
        try:
            # FastText handles OOV via subword embeddings
            emb = ft_model[word]
            embeddings.append(emb)
        except KeyError:
            oov_words.append(word)
            # Use zero vector for truly OOV words (rare with FastText)
            embeddings.append(np.zeros(300))

    embeddings = np.array(embeddings)

    click.secho(f"Done ({time.time()-extract_start:.1f}s)", fg='green')

    # Save to cache for next time
    click.echo("  ├─ Caching for future runs... ", nl=False)
    np.savez_compressed(cache_file, embeddings=embeddings, oov_words=np.array(oov_words))
    click.secho("Done ✓", fg='green')

    if oov_words:
        click.secho(f"  ⚠️  OOV: {', '.join(oov_words)}", fg='yellow')

        # Check if ALL words are OOV (likely wrong language model)
        if len(oov_words) == len(words):
            click.secho(f"  ⚠️  WARNING: ALL words are OOV!", fg='red', bold=True)
            click.secho(f"  ℹ️  Note: fasttext-wiki-news-subwords-300 is English-only", fg='cyan')
            click.secho(f"  ℹ️  Chinese words will return zero vectors (RA will be 0)", fg='cyan')
            click.secho(f"  ℹ️  For multilingual use, see: https://fasttext.cc/docs/en/crawl-vectors.html", fg='cyan')

    return embeddings


def load_labse_embeddings(words: List[str]) -> np.ndarray:
    """Load LaBSE embeddings (BERT era)"""

    click.echo("  ├─ Loading LaBSE model... ", nl=False)
    start = time.time()

    model = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')

    click.secho(f"Done ({time.time()-start:.1f}s)", fg='green')

    embeddings = model.encode(words, show_progress_bar=False)
    return embeddings


# ============================================================================
# MAIN CLI APPLICATION
# ============================================================================

@click.command()
@click.option('--models', default=None,
              help='Models to test. Use "list" to show all, or comma-separated: "labse,gemma,fasttext"')
@click.option('--datasets', default=None,
              help='Datasets to test. Use "list" to show all, or comma-separated: "animal-gender,kinship"')
@click.option('--knn', default=5,
              help='k-Nearest Neighbors for PHATE (default: 5)')
@click.option('--t', default=5,
              help='Diffusion time for PHATE (default: 5)')
@click.option('--output', default=None,
              help='Save results to CSV file (optional). For multiple datasets, files will be named ms_ra-<date-time-string>.csv')
@click.option('--clear-cache', is_flag=True,
              help='Clear FastText embedding cache before running')
def calculate_ms_ra_metrics(models, datasets, knn, t, output, clear_cache):
    """
    PHATE MANIFOLD Metric Analysis

    Evaluates Manifold Similarity (MS) and Relational Affinity (RA)
    across FastText (pre-BERT), LaBSE (BERT), and Gemma (LLM) embeddings.

    Example usage:
        python phate_manifold_metrics.py                    # All combinations (with confirmation)
        python phate_manifold_metrics.py --models list      # List available models
        python phate_manifold_metrics.py --datasets list    # List available datasets
        python phate_manifold_metrics.py --models labse,fasttext
        python phate_manifold_metrics.py --datasets animal-gender,kinship,temporal
        python phate_manifold_metrics.py --models labse --datasets sequential-days --output results.csv
        python phate_manifold_metrics.py --clear-cache      # Clear FastText cache
    """
    # Import datasets first (for listing)
    try:
        from test_datasets import ALL_DATASETS
    except ImportError:
        click.secho("✗ test_datasets.py not found!", fg='red', bold=True)
        click.echo("Please ensure test_datasets.py is in the same directory.")
        raise SystemExit(1)

    # Define available models
    AVAILABLE_MODELS = {
        'fasttext': 'FastText Multilingual 300D (Pre-BERT era)',
        'labse': 'LaBSE 768D (BERT era)',
        'gemma': 'Gemma/Ollama (LLM era)'
    }

    # Handle model listing
    if models and models.lower() == "list":
        click.secho("╔══════════════════════════════════════════════════════════════════════╗", fg='bright_blue')
        click.secho("║              Available Models for PHATE Analysis                    ║", fg='bright_blue', bold=True)
        click.secho("╚══════════════════════════════════════════════════════════════════════╝", fg='bright_blue')
        click.echo("")

        for model_id, description in AVAILABLE_MODELS.items():
            click.secho(f"▶ {model_id}", fg='yellow', bold=True)
            click.echo(f"  {description}")
            click.echo("")

        click.secho("Usage Examples:", fg='cyan', bold=True)
        click.echo("  # Test single model")
        click.echo("  python phate_manifold_metrics.py --models labse")
        click.echo("")
        click.echo("  # Test multiple models")
        click.echo("  python phate_manifold_metrics.py --models labse,gemma")
        click.echo("")
        click.echo("  # Test all models")
        click.echo("  python phate_manifold_metrics.py --models fasttext,labse,gemma")
        click.echo("")

        raise SystemExit(0)

    # Handle dataset listing
    if datasets and datasets.lower() == "list":
        click.secho("╔══════════════════════════════════════════════════════════════════════╗", fg='bright_blue')
        click.secho("║         Available Datasets for Dual-Metric PHATE Analysis          ║", fg='bright_blue', bold=True)
        click.secho("╚══════════════════════════════════════════════════════════════════════╝", fg='bright_blue')
        click.echo("")

        for name, dataset_info in ALL_DATASETS.items():
            if isinstance(dataset_info, dict) and 'description' in dataset_info:
                click.secho(f"▶ {name}", fg='yellow', bold=True)
                click.echo(f"  Name: {dataset_info.get('name', name)}")
                click.echo(f"  Description: {dataset_info.get('description', 'N/A')}")

                # Show pair count
                if 'en' in dataset_info:
                    click.echo(f"  Pairs: {len(dataset_info['en'])}")

                # Show expected RA
                if 'expected_ra' in dataset_info:
                    click.echo(f"  Expected RA:")
                    click.echo(f"    EN: {dataset_info['expected_ra'].get('en', 'N/A')}")
                    click.echo(f"    ZH: {dataset_info['expected_ra'].get('zh', 'N/A')}")
                click.echo("")
            else:
                click.secho(f"▶ {name}", fg='yellow')
                click.echo(f"  (Legacy format)")
                click.echo("")

        click.secho("Usage Examples:", fg='cyan', bold=True)
        click.echo("  # Test single dataset")
        click.echo("  python phate_manifold_metrics.py --datasets animal-gender")
        click.echo("")
        click.echo("  # Test multiple datasets")
        click.echo("  python phate_manifold_metrics.py --datasets animal-gender,kinship,temporal")
        click.echo("")
        click.echo("  # Test all compositional datasets")
        click.echo("  python phate_manifold_metrics.py --datasets sequential-days,action-agent")
        click.echo("")

        raise SystemExit(0)

    # Handle "no args" mode: ALL COMBINATIONS (with confirmation)
    if models is None and datasets is None:
        # Generate timestamp for output filename

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"ms_ra_{timestamp}.csv"

        # All models and all datasets
        all_models = list(AVAILABLE_MODELS.keys())
        all_datasets = list(ALL_DATASETS.keys())

        click.secho("╔══════════════════════════════════════════════════════════════════════╗", fg='bright_blue')
        click.secho("║         COMPREHENSIVE DUAL-METRIC PHATE ANALYSIS                    ║", fg='bright_blue', bold=True)
        click.secho("╚══════════════════════════════════════════════════════════════════════╝", fg='bright_blue')
        click.echo("")
        click.echo("⚠️  No models or datasets specified - will test ALL COMBINATIONS")
        click.echo("")

        click.secho("📊 Test Plan:", fg='cyan', bold=True)
        click.echo(f"  • Models ({len(all_models)}): {', '.join(all_models)}")
        click.echo(f"  • Datasets ({len(all_datasets)}): {', '.join(all_datasets)}")
        click.echo(f"  • Total combinations: {len(all_models)} × {len(all_datasets)} = {len(all_models) * len(all_datasets)}")
        click.echo(f"  • Output file: {default_output}")
        click.echo("")

        click.secho("⏱️  Estimated Time:", fg='yellow', bold=True)
        click.echo("  • First run: ~45-60 minutes (with FastText model loading)")
        click.echo("  • Subsequent runs: ~15-20 minutes (with cache)")
        click.echo("")

        # Prompt user
        if not click.confirm("Proceed with comprehensive analysis?", default=False):
            click.secho("❌ Aborted by user", fg='red')
            raise SystemExit(0)

        click.echo("")

        # Use all combinations
        selected_models = all_models
        dataset_list = all_datasets
        output = output or default_output

    else:
        # Parse user-provided models and datasets
        if models is None:
            # Default to all models if not specified
            selected_models = list(AVAILABLE_MODELS.keys())
        else:
            selected_models = [m.strip().lower() for m in models.split(',')]

        if datasets is None:
            # Default to all datasets if not specified
            dataset_list = list(ALL_DATASETS.keys())
        else:
            dataset_list = [d.strip() for d in datasets.split(',')]

    # Clear FastText cache if requested
    if clear_cache:
        cache_dir = Path.home() / 'projects' / 'embedding_cache' / 'fasttext'
        if cache_dir.exists():
            
            shutil.rmtree(cache_dir)
            click.secho(f"✓ Cleared FastText cache: {cache_dir}", fg='yellow')
        else:
            click.secho("✓ No FastText cache to clear", fg='yellow')

    # Validate all datasets exist
    for dataset in dataset_list:
        if dataset not in ALL_DATASETS:
            click.secho(f"✗ Dataset '{dataset}' not found!", fg='red', bold=True)
            click.echo(f"Available datasets: {', '.join(ALL_DATASETS.keys())}")
            click.echo("")
            click.echo("Run: python phate_manifold_metrics.py --datasets list")
            raise SystemExit(1)

    # Validate all models
    for model in selected_models:
        if model not in AVAILABLE_MODELS:
            click.secho(f"✗ Model '{model}' not found!", fg='red', bold=True)
            click.echo(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
            click.echo("")
            click.echo("Run: python phate_manifold_metrics.py --models list")
            raise SystemExit(1)

    # Master results collection (for all-combinations mode)
    all_results = []

    # Process each dataset
    for dataset_idx, dataset in enumerate(dataset_list):
        dataset_info = ALL_DATASETS[dataset]

        # Extract word pairs based on format
        if isinstance(dataset_info, dict) and 'en' in dataset_info:
            word_pairs_en = dataset_info['en']
            word_pairs_zh = dataset_info['zh']
            dataset_name = dataset_info.get('name', dataset)
            dataset_desc = dataset_info.get('description', '')
        else:
            # Legacy format
            word_pairs_en = dataset_info
            word_pairs_zh = dataset_info
            dataset_name = dataset
            dataset_desc = ''

        words_en = [w for pair in word_pairs_en for w in pair]
        words_zh = [w for pair in word_pairs_zh for w in pair]
        indices = [(i*2, i*2 + 1) for i in range(len(word_pairs_en))]

        # Determine output file for this dataset
        if len(dataset_list) > 1:
            # Multiple datasets: auto-name
            dataset_output = f"{dataset}_results.csv" if not output else f"{Path(output).stem}_{dataset}.csv"
        else:
            # Single dataset: use provided output or None
            dataset_output = output

        # Header (clear screen only for first dataset)
        if dataset_idx == 0:
            click.clear()

        # Dataset separator for multiple datasets
        if len(dataset_list) > 1 and dataset_idx > 0:
            click.echo("\n")
            click.secho("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", fg='bright_blue')
            click.echo("\n")

        click.secho("╔══════════════════════════════════════════════════════════════════════╗", fg='bright_blue')
        click.secho("║         PHATE MANIFOLD METRICS: DUAL-METRIC ANALYSIS                ║", fg='bright_blue', bold=True)
        click.secho("╚══════════════════════════════════════════════════════════════════════╝", fg='bright_blue')

        if len(dataset_list) > 1:
            click.echo(f"  [Progress] Dataset {dataset_idx + 1}/{len(dataset_list)}")

        click.echo(f"  [RAM] Usage: {get_ram_usage():.2f}GB")
        click.echo(f"  [PHATE] knn={knn}, t={t}")
        click.echo(f"  [Dataset] {dataset_name}: {len(word_pairs_en)} pairs × 2 languages")
        if dataset_desc:
            click.echo(f"  [Description] {dataset_desc}")
        click.echo("")

        results = []

        # ========================================================================
        # ERA 1: FASTTEXT (Pre-BERT Static Embeddings)
        # ========================================================================
        if 'fasttext' in selected_models or 'all' in selected_models:
            click.secho("▶ ERA 1: Static Manifold (FastText Multilingual 300D)", fg='yellow', bold=True)

            # Try multilingual FastText first, fall back to gensim if not available
            use_multilingual = True

            try:
                for lang, words in [("EN", words_en), ("ZH", words_zh)]:
                    click.echo(f"  ├─ Auditing {lang}... ", nl=False)

                    # Try loading FastText (extracted vocab > full model > ERROR)
                    lang_code = 'en' if lang == 'EN' else 'zh'

                    try:
                        # Option 1: Extracted vocabulary (fast, small ~120MB)
                        embs = load_fasttext_from_extracted(words, lang=lang_code)
                    except FileNotFoundError:
                        try:
                            # Option 2: Full multilingual model (slow, large ~7GB)
                            embs = load_fasttext_multilingual(words, lang=lang_code)
                        except (ImportError, FileNotFoundError) as e:
                            # HARD REQUIREMENT: Must have bilingual support
                            click.secho("", nl=True)  # New line
                            click.secho(f"✗ FastText {lang} model not available!", fg='red', bold=True)
                            click.echo("")
                            click.secho("For proper 3-era bilingual study, you need:", fg='yellow')
                            click.echo("")
                            click.secho("  Option 1: Extract vocabulary (RECOMMENDED)", fg='cyan')
                            click.echo("    pip install fasttext")
                            click.echo("    ./setup_fasttext.sh  # Download 14GB")
                            click.echo("    python extract_fasttext_full_vocab.py --top-n 100000")
                            click.echo("    rm -rf ~/.fasttext/models/  # Delete 14GB, keep 240MB")
                            click.echo("")
                            click.secho("  Option 2: Keep full models (14GB)", fg='cyan')
                            click.echo("    pip install fasttext")
                            click.echo("    ./setup_fasttext.sh")
                            click.echo("")
                            click.secho("  FastText is REQUIRED for Pre-BERT era baseline.", fg='yellow', bold=True)
                            click.secho("  No fallback to English-only gensim model.", fg='yellow')
                            click.echo("")
                            raise SystemExit(1)

                    metrics = PhateMetrics(knn=knn, t=t).fit(embs).compute_all_metrics(indices, embs)

                    results.append({
                        "Dataset": dataset,
                        "Era": "FastText",
                        "Lang": lang,
                        "MS": metrics['ms_score'],
                        "RA": metrics['ra_score'],
                        "CV": metrics['cv'],
                        "RA_std": metrics['std_ra']
                    })

                    click.secho("OK", fg='green')

            except Exception as e:
                click.secho(f"FAILED: {type(e).__name__}: {str(e)}", fg='red')

            click.echo("")

        # ========================================================================
        # ERA 2: LABSE (BERT-era Contextual Embeddings)
        # ========================================================================
        if 'labse' in selected_models or 'all' in selected_models:
            click.secho("▶ ERA 2: BERT Manifold (LaBSE 768D)", fg='yellow', bold=True)
            try:
                for lang, words in [("EN", words_en), ("ZH", words_zh)]:
                    click.echo(f"  ├─ Auditing {lang}... ", nl=False)

                    embs = load_labse_embeddings(words)
                    metrics = PhateMetrics(knn=knn, t=t).fit(embs).compute_all_metrics(indices, embs)

                    results.append({
                        "Dataset": dataset,
                        "Era": "LaBSE",
                        "Lang": lang,
                        "MS": metrics['ms_score'],
                        "RA": metrics['ra_score'],
                        "CV": metrics['cv'],
                        "RA_std": metrics['std_ra']
                    })

                    click.secho("OK", fg='green')

            except Exception as e:
                click.secho(f"FAILED: {type(e).__name__}: {str(e)}", fg='red')

            click.echo("")

        # ========================================================================
        # ERA 3: GEMMA (LLM-era Embeddings via Ollama)
        # ========================================================================
        if 'gemma' in selected_models or 'all' in selected_models:
            click.secho("▶ ERA 3: LLM Manifold (Gemma via Ollama)", fg='yellow', bold=True)

            # Check Ollama availability
            is_available, info = check_ollama_availability()

            if not is_available:
                click.secho(f"  ✗ Ollama unavailable: {info}", fg='red')
                click.secho("  ℹ️  Start Ollama: ollama serve", fg='cyan')
            else:
                available_models = info
                click.secho(f"  ✓ Ollama running | Models: {', '.join(available_models[:3])}", fg='green')

                # Try multiple model names
                model_candidates = ["snowflake-arctic-embed2", "snowflake-arctic-embed", "gemma:2b"]
                model_to_use = None

                for candidate in model_candidates:
                    if any(candidate in m for m in available_models):
                        model_to_use = candidate
                        break

                if not model_to_use:
                    click.secho(f"  ✗ No suitable model found. Available: {available_models}", fg='red')
                else:
                    click.secho(f"  ✓ Using model: {model_to_use}", fg='green')

                    try:
                        for lang, words in [("EN", words_en), ("ZH", words_zh)]:
                            click.echo(f"  ├─ Auditing {lang}... ", nl=False)

                            embs = get_ollama_embeddings_fixed(words, model_name=model_to_use)
                            metrics = PhateMetrics(knn=knn, t=t).fit(embs).compute_all_metrics(indices, embs)

                            results.append({
                                "Dataset": dataset,
                                "Era": "Gemma/Ollama",
                                "Lang": lang,
                                "MS": metrics['ms_score'],
                                "RA": metrics['ra_score'],
                                "CV": metrics['cv'],
                                "RA_std": metrics['std_ra']
                            })

                            click.secho("OK", fg='green')

                    except Exception as e:
                        click.secho(f"FAILED: {type(e).__name__}: {str(e)}", fg='red')

            click.echo("")

        # ========================================================================
        # RESULTS TABLE
        # ========================================================================
        if results:
            df = pd.DataFrame(results)

            click.secho("┌────────────────────────────────────────────────────────────────────────┐", fg='bright_blue')
            click.secho("│                       DUAL-METRIC AUDIT REPORT                         │", fg='bright_blue', bold=True)
            click.secho("├────────────────────────────────────────────────────────────────────────┤", fg='bright_blue')
            click.secho("│  MS = Manifold Similarity (↑ better) | RA = Relational Affinity       │", fg='cyan')
            click.secho("├────────────────────────────────────────────────────────────────────────┤", fg='bright_blue')

            # Table header
            header = f"{'Model':<15} | {'Lang':<4} | {'MS':<8} | {'RA':<8} | {'CV':<8} | {'RA_std':<8}"
            click.echo(f"│ {header} │")
            click.echo("├────────────────┼──────┼──────────┼──────────┼──────────┼──────────┤")

            # Table rows with color coding
            for _, row in df.iterrows():
                ms_color = 'green' if row['MS'] > 0.7 else 'yellow' if row['MS'] > 0.4 else 'red'
                ra_color = 'green' if row['RA'] > 0.5 else 'yellow' if row['RA'] > 0.2 else 'red'

                line = f"{row['Era']:<15} | {row['Lang']:<4} | "
                click.echo(f"│ {line}", nl=False)
                click.secho(f"{row['MS']:<8.3f}", fg=ms_color, nl=False)
                click.echo(" | ", nl=False)
                click.secho(f"{row['RA']:<8.3f}", fg=ra_color, nl=False)
                click.echo(f" | {row['CV']:<8.3f} | {row['RA_std']:<8.3f} │")

            click.secho("└────────────────────────────────────────────────────────────────────────┘", fg='bright_blue')

            # Key insights
            click.echo("\n📊 Key Insights:")

            # Compare EN vs ZH for each model
            for era in df['Era'].unique():
                era_data = df[df['Era'] == era]
                if len(era_data) == 2:
                    en_row = era_data[era_data['Lang'] == 'EN'].iloc[0]
                    zh_row = era_data[era_data['Lang'] == 'ZH'].iloc[0]

                    ra_gap = (zh_row['RA'] - en_row['RA']) / en_row['RA'] * 100 if en_row['RA'] > 0 else 0

                    if abs(ra_gap) > 100:  # Significant morphological anisotropy
                        click.secho(f"  • {era}: Morphological Anisotropy = {ra_gap:+.0f}% (ZH vs EN)", fg='yellow', bold=True)
                        click.echo(f"    → Chinese compositional advantage: RA(ZH)={zh_row['RA']:.3f} vs RA(EN)={en_row['RA']:.3f}")

            # Collect results for master CSV
            all_results.extend(results)

            # Save to per-dataset CSV if requested and not in comprehensive mode
            if dataset_output and len(dataset_list) > 1:
                df.to_csv(dataset_output, index=False)
                click.secho(f"\n💾 Results saved to: {dataset_output}", fg='green')
        else:
            click.secho("❌ No results collected. Check model availability.", fg='red', bold=True)

    # Save master CSV with all results
    if all_results:
        if len(dataset_list) > 1 or (models is None and datasets is None):
            # Comprehensive mode or multiple datasets: save master CSV
            master_df = pd.DataFrame(all_results)

            # Reorder columns for clarity
            column_order = ["Dataset", "Era", "Lang", "MS", "RA", "CV", "RA_std"]
            master_df = master_df[column_order]

            if output:
                master_df.to_csv(output, index=False)
                click.echo("")
                click.secho("╔══════════════════════════════════════════════════════════════════════╗", fg='bright_green')
                click.secho("║                    Master Results Saved                              ║", fg='bright_green', bold=True)
                click.secho("╚══════════════════════════════════════════════════════════════════════╝", fg='bright_green')
                click.echo(f"  💾 All results: {output}")
                click.echo(f"  📊 Total rows: {len(master_df)}")
                click.echo(f"  📁 Datasets: {len(dataset_list)}")
                click.echo(f"  🎯 Models: {len(selected_models)}")
                click.echo("")
        elif output:
            # Single dataset: save to specified output
            master_df = pd.DataFrame(all_results)
            column_order = ["Dataset", "Era", "Lang", "MS", "RA", "CV", "RA_std"]
            master_df = master_df[column_order]
            master_df.to_csv(output, index=False)
            click.secho(f"\n💾 Results saved to: {output}", fg='green')


if __name__ == "__main__":
    calculate_ms_ra_metrics()
