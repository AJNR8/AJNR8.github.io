"""
main.py — Web-based allergy-aware recipe recommender + Logistic Regression ranker
 
Project structure:
PROJECT/
├── code/
│   ├── main.py
│   └── templates/
│       ├── index.html
│       └── results.html
└── datasets/
    ├── FOOD-DATA-GROUP1.csv ... FOOD-DATA-GROUP5.csv
    ├── recipes.csv
    └── food_ingredients_and_allergens.csv
 
Run from project root:
  python .\\code\\main.py
Then open:
  http://127.0.0.1:5000
"""
 
import os
import re
from typing import Any, Dict, List, Optional, Set
 
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
 
app = Flask(__name__)
 
#--------
# Paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "..", "datasets")
 
FOOD_FILES = [
    "FOOD-DATA-GROUP1.csv",
    "FOOD-DATA-GROUP2.csv",
    "FOOD-DATA-GROUP3.csv",
    "FOOD-DATA-GROUP4.csv",
    "FOOD-DATA-GROUP5.csv",
]
RECIPES_FILE = "recipes.csv"
ING_ALLER_FILE = "food_ingredients_and_allergens.csv"
MODEL_PATH = os.path.join(SCRIPT_DIR, "ranker.joblib")
 
 
#-----------------------
# Allergens + categories

STANDARD_ALLERGENS = {
    "dairy", "eggs", "gluten", "peanuts", "tree_nuts", "soy", "fish", "shellfish", "sesame"
}
 
CATEGORY_SYNONYMS = {
    "pasta": [
        "pasta", "spaghetti", "penne", "macaroni", "fettuccine", "linguine",
        "lasagna", "ravioli", "tortellini", "tagliatelle", "fusilli",
        "noodle", "noodles", "orzo", "gnocchi", "vermicelli"
    ],
    "soup": ["soup", "broth", "stew", "chowder", "bisque", "ramen", "pho"],
    "rice": ["rice", "risotto", "pilaf", "biryani", "fried rice", "paella"],
    "chicken": ["chicken", "breast", "thigh", "drumstick", "wing", "tenderloin"],
    "salad": ["salad", "slaw", "greens", "bowl"],
    "curry": ["curry", "masala", "korma", "tikka", "vindaloo"],
    "breakfast": ["breakfast", "pancake", "waffle", "omelet", "omelette", "muffin", "granola", "oatmeal", "porridge", "toast"],
    "dessert": ["dessert", "cake", "cookie", "brownie", "pie", "pudding", "ice cream", "cheesecake"],
}
 
 
#----------------
# Text utilities

def norm_text(x: Any) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.lower().strip()
    x = re.sub(r"[^a-z0-9,\s/&\-\(\)]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x
 
 
def split_list_field(x: Any) -> List[str]:
    s = norm_text(x)
    if not s:
        return []
    parts = re.split(r"[,/;&]+", s)
    return [p.strip() for p in parts if p.strip()]
 
 
def norm_allergen_label(a: Any) -> str:
    a = norm_text(a)
    mapping = {
        "milk": "dairy", "dairy": "dairy", "lactose": "dairy", "casein": "dairy", "whey": "dairy",
        "butter": "dairy", "cheese": "dairy", "cream": "dairy", "yoghurt": "dairy", "yogurt": "dairy",
        "egg": "eggs", "eggs": "eggs",
        "wheat": "gluten", "gluten": "gluten", "barley": "gluten", "rye": "gluten",
        "semolina": "gluten", "spelt": "gluten", "flour": "gluten",
        "peanut": "peanuts", "peanuts": "peanuts", "groundnut": "peanuts",
        "tree nuts": "tree_nuts", "tree_nuts": "tree_nuts",
        "almond": "tree_nuts", "cashew": "tree_nuts", "walnut": "tree_nuts",
        "hazelnut": "tree_nuts", "pistachio": "tree_nuts", "pecan": "tree_nuts", "macadamia": "tree_nuts",
        "soy": "soy", "soya": "soy", "tofu": "soy",
        "fish": "fish", "salmon": "fish", "tuna": "fish", "cod": "fish",
        "shellfish": "shellfish", "shrimp": "shellfish", "prawn": "shellfish", "crab": "shellfish", "lobster": "shellfish",
        "sesame": "sesame", "tahini": "sesame",
    }
    return mapping.get(a, a.replace(" ", "_"))
 
 
#----------------
# Load datasets

def load_food_nutrients(base_path: str, food_files: List[str]) -> pd.DataFrame:
    """
    Load and merge the five food nutrient CSVs.
    Normalises the food name and de-duplicates by food name.
    """
    paths = [os.path.join(base_path, f) for f in food_files]
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    for col in ["Unnamed: 0", "Unnamed: 0.1"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    if "food" in df.columns:
        df["food_norm"] = df["food"].apply(norm_text)
        df = df.drop_duplicates(subset=["food_norm"], keep="first").reset_index(drop=True)
    return df
 
 
def load_recipes(base_path: str, recipes_file: str) -> pd.DataFrame:
    path = os.path.join(base_path, recipes_file)
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)
    df["recipe_name_norm"] = df["recipe_name"].apply(norm_text)
    df["ingredients_norm"] = df["ingredients"].apply(norm_text)
    return df
 
 
#-----------------------------------
# Ingredient/allergen knowledge base

def load_and_clean_ingredients_allergens(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
 
    product_col = None
    for candidate in ["Food Product", "Food", "Product", "Food_Product", "food_product"]:
        if candidate in df.columns:
            product_col = candidate
            break
    if product_col is None:
        product_col = df.columns[0]
 
    allergen_col = None
    for candidate in ["Allergens", "Allergen", "allergens", "allergen"]:
        if candidate in df.columns:
            allergen_col = candidate
            break
 
    ingredient_cols = [c for c in df.columns if c.lower() in {
        "main ingredient", "main_ingredient", "sweetener", "fat/oil", "fat_oil", "seasoning", "ingredients"
    }]
 
    df["product_norm"] = df[product_col].apply(norm_text)
 
    for c in ingredient_cols:
        df[c + "_list"] = df[c].apply(split_list_field)
 
    if allergen_col:
        df["allergens_list_raw"] = df[allergen_col].apply(split_list_field)
        df["allergens_list"] = df["allergens_list_raw"].apply(
            lambda xs: sorted({norm_allergen_label(x) for x in xs if x})
        )
    else:
        df["allergens_list"] = [[] for _ in range(len(df))]
 
    agg_dict = {"allergens_list": lambda lists: sorted(set(a for sub in lists for a in sub))}
    for c in ingredient_cols:
        agg_dict[c + "_list"] = lambda lists: sorted(set(i for sub in lists for i in sub))
 
    return df.groupby("product_norm", as_index=False).agg(agg_dict)
 
 
def build_allergen_knowledge(cleaned_df: pd.DataFrame) -> Dict[str, Set[str]]:
    allergen_to_terms: Dict[str, Set[str]] = {a: set() for a in STANDARD_ALLERGENS}
    ingredient_list_cols = [c for c in cleaned_df.columns if c.endswith("_list") and c != "allergens_list"]
 
    for _, row in cleaned_df.iterrows():
        allergens = set(row.get("allergens_list", []) or [])
        allergens = {a for a in allergens if a in STANDARD_ALLERGENS}
 
        ingredient_terms: Set[str] = set()
        for c in ingredient_list_cols:
            ingredient_terms.update(row.get(c, []) or [])
 
        for term in ingredient_terms:
            mapped = norm_allergen_label(term)
            if mapped in STANDARD_ALLERGENS:
                allergen_to_terms[mapped].add(norm_text(term))
 
        for a in allergens:
            allergen_to_terms[a].add(a)
 
    extras = {
        "dairy": {"milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "whey", "casein", "lactose"},
        "gluten": {"wheat", "flour", "bread", "pasta", "semolina", "barley", "rye", "spelt"},
        "eggs": {"egg", "eggs"},
        "peanuts": {"peanut", "peanuts", "groundnut"},
        "tree_nuts": {"almond", "cashew", "walnut", "hazelnut", "pistachio", "pecan", "macadamia"},
        "soy": {"soy", "soya", "tofu", "soy sauce"},
        "fish": {"fish", "salmon", "tuna", "cod"},
        "shellfish": {"shellfish", "shrimp", "prawn", "crab", "lobster"},
        "sesame": {"sesame", "tahini"},
    }
    for a, terms in extras.items():
        allergen_to_terms[a].update({norm_text(t) for t in terms})
 
    return allergen_to_terms
 
 
#-------------------------
# Allergy detection/filter

def detect_allergens_in_text(text_norm: str, allergen_to_terms: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    hits: Dict[str, Set[str]] = {}
    for allergen, terms in allergen_to_terms.items():
        for term in terms:
            if term and term in text_norm:
                hits.setdefault(allergen, set()).add(term)
    return {a: sorted(list(ts)) for a, ts in hits.items()}
 
 
def apply_allergy_filter(
    recipes: pd.DataFrame,
    allergen_to_terms: Dict[str, Set[str]],
    allergies: Set[str],
) -> pd.DataFrame:
    df = recipes.copy()
    df["detected_allergens"] = df["ingredients_norm"].apply(
        lambda t: detect_allergens_in_text(t, allergen_to_terms)
    )
    df["blocked"] = df["detected_allergens"].apply(lambda d: any(a in d for a in allergies))
 
    def reason(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {a: d[a] for a in allergies if a in d}
 
    df["block_reason"] = df["detected_allergens"].apply(reason)
    return df
 
 
#--------------------
# Deduping + search

def dedupe_recipes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "url" in out.columns:
        out["url_norm"] = out["url"].fillna("").astype(str).str.strip()
        out = out.sort_values(by=["url_norm"]).drop_duplicates(subset=["url_norm"], keep="first")
        out.drop(columns=["url_norm"], inplace=True)
    else:
        out = out.drop_duplicates(subset=["recipe_name_norm"], keep="first")
    return out
 
 
def _term_pattern(term: str) -> str:
    term = norm_text(term)
    if not term:
        return ""
    words = term.split()
    if len(words) == 1:
        return r"\b" + re.escape(words[0]) + r"\b"
    return r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"
 
 
def search_recipes(df: pd.DataFrame, query_norm: str) -> pd.DataFrame:
    if not query_norm:
        return df.copy()
 
    haystack = df["recipe_name_norm"].fillna("") + " " + df["ingredients_norm"].fillna("")
 
    if query_norm in CATEGORY_SYNONYMS:
        term_patterns = [_term_pattern(t) for t in CATEGORY_SYNONYMS[query_norm]]
        term_patterns = [p for p in term_patterns if p]
        pattern = r"(?:%s)" % "|".join(term_patterns)
        return df[haystack.str.contains(pattern, na=False, regex=True)].copy()
 
    keywords = [k for k in query_norm.split() if k]
    if not keywords:
        return df.copy()
 
    mask = pd.Series(True, index=df.index)
    for k in keywords:
        mask = mask & haystack.str.contains(_term_pattern(k), na=False, regex=True)
    return df[mask].copy()
 
 
#---------------------------------
# Nutrition parsing for recipe CSV

def parse_recipe_nutrition(nutrition_text: Any) -> Dict[str, float]:
    t = norm_text(nutrition_text)

    def find_g(label: str) -> Optional[float]:
        m = re.search(label + r"\s+(\d+(?:\.\d+)?)g", t)
        return float(m.group(1)) if m else None

    def find_mg(label: str) -> Optional[float]:
        m = re.search(label + r"\s+(\d+(?:\.\d+)?)mg", t)
        return float(m.group(1)) if m else None

    out: Dict[str, float] = {}

    fat   = find_g(r"total fat")
    carbs = find_g(r"total carbohydrate")
    prot  = find_g(r"protein")
    sod   = find_mg(r"sodium")

    if fat is not None and carbs is not None and prot is not None:
        out["calories"] = round(fat * 9 + carbs * 4 + prot * 4, 1)

    if prot is not None:
        out["protein_g"] = prot

    if sod is not None:
        out["sodium_mg"] = sod

    return out
 
 
def ensure_recipe_nutrition_columns(recipes: pd.DataFrame) -> pd.DataFrame:
    df = recipes.copy()
    if "nutrition" not in df.columns:
        df["calories"] = np.nan
        df["protein_g"] = np.nan
        df["sodium_mg"] = np.nan
        return df
 
    parsed = df["nutrition"].apply(parse_recipe_nutrition)
    df["calories"] = parsed.apply(lambda d: d.get("calories", np.nan))
    df["protein_g"] = parsed.apply(lambda d: d.get("protein_g", np.nan))
    df["sodium_mg"] = parsed.apply(lambda d: d.get("sodium_mg", np.nan))
    return df
 
 
#-----------------------------------
# Food nutrient database integration

# Columns from the food CSVs to surface as nutrient highlights
NUTRIENT_DISPLAY_COLS = [
    "Protein", "Fat", "Carbohydrates", "Dietary Fiber",
    "Sodium", "Calcium", "Iron", "Potassium",
    "Vitamin C", "Vitamin A", "Vitamin D",
    "Nutrition Density",
]
 
 
def enrich_recipes_with_food_nutrients(
    recipes: pd.DataFrame,
    food_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each recipe, find the longest food-database entry whose name appears
    as a whole word/phrase in the recipe's ingredient list.  Attach those
    nutrient values as extra columns (prefixed 'fn_') so they can drive both
    the ML ranker features and the explanation trace.
    """
    available_cols = [c for c in NUTRIENT_DISPLAY_COLS if c in food_df.columns]
 
    # Build a fast lookup dict: food_norm -> Series
    food_lookup: Dict[str, pd.Series] = {
        row["food_norm"]: row for _, row in food_df.iterrows() if row["food_norm"]
    }

     # Blacklist of minor ingredients to skip during matching
    MINOR_INGREDIENTS = {
        "salt", "pepper", "black pepper", "white pepper", "oil", "olive oil",
        "vegetable oil", "butter", "sugar", "brown sugar", "white sugar",
        "flour", "water", "garlic", "onion", "vinegar", "red wine vinegar",
        "white vinegar", "balsamic vinegar", "apple cider vinegar",
        "lemon juice", "lime juice", "soy sauce", "hot sauce",
        "worcestershire sauce", "cayenne pepper", "paprika", "cumin",
        "oregano", "thyme", "basil", "parsley", "cilantro", "coriander",
        "bay leaf", "cinnamon", "nutmeg", "ginger", "turmeric", "chili",
        "chilli", "red pepper", "red pepper flakes", "mustard", "honey",
        "maple syrup", "cornstarch", "baking soda", "baking powder",
        "vanilla", "vanilla extract", "cream of tartar", "yeast",
        "cooking spray", "nonstick spray",
    }
    # Sort descending by name length so longer/more-specific names win
    sorted_foods = sorted(food_lookup.keys(), key=len, reverse=True)
 
    def _enrich(ingredients_norm: str) -> Dict[str, Any]:
        for fname in sorted_foods:
            if fname in MINOR_INGREDIENTS:
                continue
            pattern = r"\b" + re.escape(fname) + r"\b"
            if re.search(pattern, ingredients_norm):
                row = food_lookup[fname]
                result = {f"fn_{c}": row.get(c, np.nan) for c in available_cols}
                result["fn_matched_food"] = row.get("food", fname)
                return result
        return {f"fn_{c}": np.nan for c in available_cols} | {"fn_matched_food": None}
 
    enriched = recipes["ingredients_norm"].apply(_enrich)
    enriched_df = pd.DataFrame(list(enriched), index=recipes.index)
    return pd.concat([recipes, enriched_df], axis=1)
 
 
#----------------------------
# Explanation trace builder

def build_explanation(row: pd.Series, profile: Dict[str, Any]) -> List[str]:
    """
    Returns a human-readable list of reasons explaining why this recipe was
    recommended, covering: allergen safety, query match, calorie/protein/sodium
    targets, likes/dislikes, food-nutrient-db data, and user rating.
    """
    reasons: List[str] = []
 
    # - Allergen safety (always shown first)
    reasons.append("✅ Passes allergen safety check — no blocked ingredients detected.")
 
    # - Search query match 
    query = profile.get("query", "")
    if query:
        name = norm_text(row.get("recipe_name", ""))
        ingredients = row.get("ingredients_norm", "")
        if query in CATEGORY_SYNONYMS:
            matched_syn = [s for s in CATEGORY_SYNONYMS[query] if s in name or s in ingredients]
            if matched_syn:
                reasons.append(
                    f"🔍 Matches your search '{query}' (found: {', '.join(matched_syn[:3])})."
                )
        else:
            if query in name or query in ingredients:
                reasons.append(f"🔍 Contains your search term '{query}'.")
 
    # - Calorie target
    cal_target = profile.get("calorie_target")
    calories = row.get("calories")
    if cal_target and pd.notna(calories):
        diff = abs(float(calories) - cal_target)
        pct = diff / cal_target * 100
        if pct <= 15:
            reasons.append(
                f"🎯 Calories ({int(calories)} kcal) are very close to your target of {cal_target} kcal."
            )
        elif pct <= 30:
            reasons.append(
                f"⚖️ Calories ({int(calories)} kcal) are within 30 % of your target of {cal_target} kcal."
            )
 
    # -Protein target
    min_protein = profile.get("min_protein_g")
    protein = row.get("protein_g")
    if min_protein and pd.notna(protein):
        if float(protein) >= min_protein:
            reasons.append(f"💪 Protein ({protein:.1f} g) meets your minimum of {min_protein} g.")
        else:
            reasons.append(f"⚠️ Protein ({protein:.1f} g) is below your target of {min_protein} g.")
 
    # -Sodium limit
    max_sodium = profile.get("max_sodium_mg")
    sodium = row.get("sodium_mg")
    if max_sodium and pd.notna(sodium):
        if float(sodium) <= max_sodium:
            reasons.append(f"🧂 Sodium ({int(sodium)} mg) is within your limit of {max_sodium} mg.")
        else:
            reasons.append(f"⚠️ Sodium ({int(sodium)} mg) exceeds your limit of {max_sodium} mg.")
 
    # -Likes
    likes = profile.get("likes") or set()
    text = row.get("recipe_name_norm", "") + " " + row.get("ingredients_norm", "")
    matched_likes = [t for t in likes if t and t in text]
    if matched_likes:
        reasons.append(f"❤️ Contains ingredients/style you like: {', '.join(matched_likes)}.")
 
    # -Dislikes
    dislikes = profile.get("dislikes") or set()
    matched_dislikes = [t for t in dislikes if t and t in text]
    if matched_dislikes:
        reasons.append(f"👎 Note: contains items you dislike: {', '.join(matched_dislikes)}.")
 
    # -Food nutrient database match
    matched_food = row.get("fn_matched_food")
    if matched_food:
        nd = row.get("fn_Nutrition Density")
        fn_protein = row.get("fn_Protein")
        fn_fiber = row.get("fn_Dietary Fiber")
        fn_vitc = row.get("fn_Vitamin C")
 
        nutrient_notes = []
        if pd.notna(nd):
            nutrient_notes.append(f"nutrition density {nd:.1f}")
        if pd.notna(fn_protein):
            nutrient_notes.append(f"{fn_protein:.1f} g protein/100 g")
        if pd.notna(fn_fiber):
            nutrient_notes.append(f"{fn_fiber:.1f} g fibre/100 g")
        if pd.notna(fn_vitc):
            nutrient_notes.append(f"vitamin C {fn_vitc:.1f} mg/100 g")
 
        if nutrient_notes:
            reasons.append(
                f"📊 Key ingredient '{matched_food}' (nutrient DB): {'; '.join(nutrient_notes)}."
            )
        else:
            reasons.append(f"📊 Key ingredient '{matched_food}' found in nutrient database.")
 
    # - User rating
    rating = row.get("rating")
    if pd.notna(rating):
        try:
            r = float(rating)
            if r >= 4.5:
                reasons.append(f"⭐ Highly rated by users ({r:.1f}/5).")
            elif r >= 4.0:
                reasons.append(f"⭐ Well rated by users ({r:.1f}/5).")
        except (ValueError, TypeError):
            pass
 
    return reasons
 
 
#----------------------------
# Logistic Regression ranker

def build_features(row: pd.Series, profile: Dict[str, Any]) -> List[float]:
    # Recipe-level nutrition (from recipe CSV)
    calories = float(row["calories"]) if pd.notna(row.get("calories")) else 0.0
    protein  = float(row["protein_g"]) if pd.notna(row.get("protein_g")) else 0.0
    sodium   = float(row["sodium_mg"]) if pd.notna(row.get("sodium_mg")) else 0.0
 
    # Food-database nutrient features 
    fn_protein = float(row["fn_Protein"])          if pd.notna(row.get("fn_Protein"))          else 0.0
    fn_fiber   = float(row["fn_Dietary Fiber"])    if pd.notna(row.get("fn_Dietary Fiber"))    else 0.0
    fn_nd      = float(row["fn_Nutrition Density"]) if pd.notna(row.get("fn_Nutrition Density")) else 0.0
    fn_sodium  = float(row["fn_Sodium"])           if pd.notna(row.get("fn_Sodium"))           else 0.0
    has_food_match = 1.0 if row.get("fn_matched_food") else 0.0
 
    cal_target  = profile.get("calorie_target")  or 0
    min_protein = profile.get("min_protein_g")   or 0
    max_sodium  = profile.get("max_sodium_mg")   or 0
 
    calorie_diff    = abs(calories - cal_target) if cal_target else 0.0
    is_high_protein = 1.0 if (min_protein and protein >= min_protein) else 0.0
    sodium_over     = max(0.0, sodium - max_sodium) if max_sodium else 0.0
 
    likes    = profile.get("likes",    set()) or set()
    dislikes = profile.get("dislikes", set()) or set()
    text = row.get("recipe_name_norm", "") + " " + row.get("ingredients_norm", "")
 
    matches_preferences = 1.0 if any(t and t in text for t in likes)    else 0.0
    contains_disliked   = 1.0 if any(t and t in text for t in dislikes) else 0.0
 
    return [
        calorie_diff,
        protein,
        sodium,
        is_high_protein,
        sodium_over,
        matches_preferences,
        contains_disliked,
        # food-nutrient-db features
        fn_protein,
        fn_fiber,
        fn_nd,
        fn_sodium,
        has_food_match,
    ]
 
 
def train_ranker(
    recipes_for_training: pd.DataFrame,
    training_profile: Dict[str, Any],
    save_path: Optional[str] = None,
) -> LogisticRegression:
    df = recipes_for_training.copy()
    df = df[df["rating"].notna()].copy()
    df["label"] = df["rating"].apply(lambda r: 1 if float(r) >= 4.5 else 0)
 
    X = np.array([build_features(r, training_profile) for _, r in df.iterrows()], dtype=float)
    y = df["label"].to_numpy(dtype=int)
 
    if len(df) >= 50 and len(set(y)) == 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test,  y_test  = X, y
 
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
 
    if len(set(y_test)) == 2 and len(y_test) >= 10:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"[ML] Ranker trained. Test ROC-AUC: {auc:.3f}")
    else:
        print("[ML] Ranker trained (not enough data for ROC-AUC).")
 
    if save_path:
        joblib.dump(model, save_path)
        print(f"[ML] Saved model to: {save_path}")
 
    return model
 
 
def load_or_train_ranker(
    training_df: pd.DataFrame,
    training_profile: Dict[str, Any],
    model_path: str,
) -> LogisticRegression:
    def current_feature_count() -> int:
        return len(build_features(training_df.iloc[0], training_profile))
 
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            expected = getattr(model, "n_features_in_", None)
            got = current_feature_count()
            if expected is not None and expected != got:
                raise ValueError(f"Feature mismatch: model expects {expected}, code provides {got}")
            print("[ML] Loaded existing ranker:", model_path)
            return model
        except Exception as e:
            print(f"[ML] Saved model not compatible ({e}). Retraining...")
 
    print("[ML] Training a new ranker...")
    return train_ranker(training_df, training_profile, save_path=model_path)
 
 
def rank_with_model(
    candidates: pd.DataFrame,
    model: LogisticRegression,
    profile: Dict[str, Any],
) -> pd.DataFrame:
    df = candidates.copy()
    if len(df) == 0:
        return df
 
    X = np.array([build_features(r, profile) for _, r in df.iterrows()], dtype=float)
    df["ml_score"] = model.predict_proba(X)[:, 1]
 
    if "rating" in df.columns:
        df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        df["final_score"] = 0.7 * df["ml_score"] + 0.3 * (df["rating_num"] / 5.0)
    else:
        df["final_score"] = df["ml_score"]
 
    return df.sort_values("final_score", ascending=False)
 
 
#----------------------
# App initialisation

FOOD_NUTRIENTS = load_food_nutrients(BASE_PATH, FOOD_FILES)
print(f"[DATA] Loaded {len(FOOD_NUTRIENTS)} food nutrient entries.")
 
RECIPES = load_recipes(BASE_PATH, RECIPES_FILE)
RECIPES = ensure_recipe_nutrition_columns(RECIPES)
RECIPES = dedupe_recipes(RECIPES)
 
# Enrich every recipe with food-database nutrient columns at startup
RECIPES = enrich_recipes_with_food_nutrients(RECIPES, FOOD_NUTRIENTS)
matched_count = RECIPES["fn_matched_food"].notna().sum()
print(f"[DATA] Food-nutrient enrichment: {matched_count}/{len(RECIPES)} recipes matched a food entry.")

cal_count = RECIPES['calories'].notna().sum()
print(f"[DATA] Recipes with calorie data: {cal_count}/{len(RECIPES)}")
 
ING_ALL_PATH = os.path.join(BASE_PATH, ING_ALLER_FILE)
ING_ALL_CLEANED = load_and_clean_ingredients_allergens(ING_ALL_PATH)
ALLERGEN_TO_TERMS = build_allergen_knowledge(ING_ALL_CLEANED)
 
TRAIN_PROFILE: Dict[str, Any] = {
    "calorie_target": 600,
    "min_protein_g": 20,
    "max_sodium_mg": 800,
    "likes": set(),
    "dislikes": set(),
}
 
# it will auto-retrain
MODEL = load_or_train_ranker(RECIPES, TRAIN_PROFILE, MODEL_PATH)
 
 
#--------------
# Flask routes

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        allergens=sorted(STANDARD_ALLERGENS),
        categories=list(CATEGORY_SYNONYMS.keys()),
    )
 
 
@app.route("/results", methods=["POST"])
def results():
    selected_allergens = request.form.getlist("allergens")
    allergies = {
        norm_allergen_label(a)
        for a in selected_allergens
        if norm_allergen_label(a) in STANDARD_ALLERGENS
    }
 
    query = norm_text(request.form.get("query", ""))
 
    calorie_target = request.form.get("calorie_target", "").strip()
    min_protein_g  = request.form.get("min_protein_g",  "").strip()
    max_sodium_mg  = request.form.get("max_sodium_mg",  "").strip()
    likes_raw      = request.form.get("likes",    "").strip()
    dislikes_raw   = request.form.get("dislikes", "").strip()
 
    def parse_prefs(raw: str) -> Set[str]:
        """Split comma- or space-separated preference terms into a set."""
        if not raw:
            return set()
        parts = re.split(r"[,\s]+", norm_text(raw))
        return {p for p in parts if p}
 
    profile: Dict[str, Any] = {
        "query": query,
        "allergies": allergies,
        "calorie_target": int(calorie_target) if calorie_target.isdigit() else None,
        "min_protein_g":  int(min_protein_g)  if min_protein_g.isdigit()  else None,
        "max_sodium_mg":  int(max_sodium_mg)  if max_sodium_mg.isdigit()  else None,
        "likes":    parse_prefs(likes_raw),
        "dislikes": parse_prefs(dislikes_raw),
    }
 
    annotated = apply_allergy_filter(RECIPES, ALLERGEN_TO_TERMS, allergies)
    safe = dedupe_recipes(annotated[~annotated["blocked"]].copy())
 
    matched = search_recipes(safe, query) if query else safe
    ranked  = rank_with_model(matched, MODEL, profile)
    top     = ranked.head(20).copy()
 
    results_list = []
    for _, r in top.iterrows():
        # Collect non-null nutrient highlights from the food database
        nutrient_highlights: Dict[str, Any] = {}
        for col in NUTRIENT_DISPLAY_COLS:
            val = r.get(f"fn_{col}")
            if pd.notna(val):
                nutrient_highlights[col] = round(float(val), 2)
 
        results_list.append({
            "recipe_name":        r.get("recipe_name", ""),
            "url":                r.get("url", ""),
            "rating":             r.get("rating", ""),
            "score":              float(r["final_score"]) if pd.notna(r.get("final_score")) else None,
            "calories":           int(r["calories"])      if pd.notna(r.get("calories"))    else None,
            "protein_g":          float(r["protein_g"])   if pd.notna(r.get("protein_g"))   else None,
            "sodium_mg":          int(r["sodium_mg"])     if pd.notna(r.get("sodium_mg"))   else None,
            "matched_food":       r.get("fn_matched_food"),
            "nutrient_highlights": nutrient_highlights,
            "explanation":        build_explanation(r, profile),
        })
 
    return render_template(
        "results.html",
        query=query,
        allergies=sorted(allergies),
        total=len(RECIPES),
        safe_count=len(safe),
        results=results_list,
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)