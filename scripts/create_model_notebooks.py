"""Generate notebooks 04, 05, 06, 07 for Phase 4 Model Development."""

import json
from pathlib import Path

OUT_DIR = Path("notebooks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def create_nb(filename: str, title: str, description: str, code_cells: list):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n", f"## E-Commerce Recommendation System\n\n", f"**Objective:** {description}"]
        }
    ]
    for code in code_cells:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")]
        })
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {OUT_DIR / filename}")


# 04_popularity_model.ipynb
code_04_1 = """import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('..').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.popularity import PopularityRecommender

pop_model = PopularityRecommender()
pop_model.load_from_parquet(ROOT / 'data/processed/popularity_features.parquet')
print(f'Popularity baseline loaded with {len(pop_model.sorted_items):,} items.')"""

code_04_2 = """recs = pop_model.recommend(visitor_id=12345, top_n=10)
print('Top 10 Global Popular Recommendations:')
for rank, (item_id, score) in enumerate(recs, 1):
    print(f'  Rank {rank:2d}: Item {item_id:<10d} (Score: {score:.1f})')"""

create_nb(
    "04_popularity_model.ipynb",
    "04 - Popularity-Based Baseline Recommendation Model",
    "Implement and test the global popularity baseline with weighted interaction confidence.",
    [code_04_1, code_04_2]
)

# 05_collaborative_filtering.ipynb
code_05_1 = """import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path('..').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.collaborative_filtering import ALSRecommender
from src.features.build_features import build_sparse_interaction_matrix

train_user_item = pd.read_parquet(ROOT / 'data/processed/train.parquet')
print(f'Train events: {len(train_user_item):,}')"""

code_05_2 = """# Build interaction matrix and train ALS
from src.features.build_features import aggregate_user_item_interactions
agg_train = aggregate_user_item_interactions(train_user_item)
mat, mappings = build_sparse_interaction_matrix(agg_train)

als = ALSRecommender(factors=64, regularization=0.1, iterations=20, random_state=42)
als.fit(mat, mappings)"""

code_05_3 = """# Test recommendation for warm user
sample_visitor = list(mappings['visitorid_to_idx'].keys())[10]
recs = als.recommend(sample_visitor, top_n=10)
print(f'Personalized ALS recommendations for Visitor {sample_visitor}:')
for r, (iid, score) in enumerate(recs, 1):
    print(f'  Rank {r:2d}: Item {iid} (Predicted score: {score:.4f})')"""

create_nb(
    "05_collaborative_filtering.ipynb",
    "05 - Implicit Alternating Least Squares (ALS) Collaborative Filtering",
    "Train and evaluate matrix factorization using Implicit ALS on sparse user-item interaction matrix.",
    [code_05_1, code_05_2, code_05_3]
)

# 06_content_based_model.ipynb
code_06_1 = """import sys
from pathlib import Path
import pandas as pd

ROOT = Path('..').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.content_based import ContentBasedRecommender

# Load sample of item properties for content modeling
p1 = pd.read_csv(ROOT / 'dataset/item_properties_part1.csv', nrows=200000)
user_item = pd.read_parquet(ROOT / 'data/processed/user_item_interactions.parquet')

content_model = ContentBasedRecommender(max_features=5000, min_df=2)
content_model.fit(p1, user_item)"""

code_06_2 = """sample_user = user_item['visitorid'].iloc[5]
recs = content_model.recommend(sample_user, top_n=10)
print(f'Content-based attribute similarity recommendations for Visitor {sample_user}:')
for r, (iid, score) in enumerate(recs, 1):
    print(f'  Rank {r:2d}: Item {iid} (Cosine similarity: {score:.4f})')"""

create_nb(
    "06_content_based_model.ipynb",
    "06 - Attribute-Based Content Recommendation Model",
    "Vectorize anonymized item properties with TF-IDF and compute sparse cosine similarity recommendations.",
    [code_06_1, code_06_2]
)

# 07_hybrid_recommendation.ipynb
code_07_1 = """import sys
from pathlib import Path

ROOT = Path('..').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.popularity import PopularityRecommender
from src.recommendation.hybrid import HybridRecommender

pop = PopularityRecommender()
pop.load_from_parquet(ROOT / 'data/processed/popularity_features.parquet')

hybrid = HybridRecommender(
    popularity_model=pop,
    collaborative_weight=0.5,
    content_weight=0.3,
    popularity_weight=0.2
)

recs, source = hybrid.recommend(visitor_id=999999999, top_n=10)
print(f'Cold-start user test: Source={source}')
for r, (iid, score) in enumerate(recs, 1):
    print(f'  Rank {r:2d}: Item {iid} (Score: {score:.4f})')"""

create_nb(
    "07_hybrid_recommendation.ipynb",
    "07 - Hybrid Recommendation Engine & Rank Fusion",
    "Combine collaborative, content-based, and popularity scores with min-max normalization and cold-start fallback.",
    [code_07_1]
)
