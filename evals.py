
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer



def compute_top_cosine_similarity(pred_string, reference_strings):
    """
    Compute the highest cosine similarity between an input string embedding and a list of string embeddings.
    ex) input_string = "Add 1 cup of rice"
        list_of_strings = ["Add 1 cup of rice", "Add 1 cup of pasta", "Add 1 cup of chicken"]
        output = 0.95
        best_match = "Add 1 cup of rice"
        best_match_index = 0

    Args:
        string: the string that we want to find the best match for (string)
        list_of_strings: List of strings (list of strings)
    
    Returns:
        A tuple containing:
        - best_score: The highest cosine similarity score
        - best_idx: The index of the best matching string
        - best_string: The text of the best matching string
    """
    # Ensure the step_embedding is 2D (batch size 1)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    pred_sentence_embedding = embedder.encode([pred_string])

    reference_sentence_embeddings = embedder.encode(reference_strings)
    
    if len(pred_sentence_embedding.shape) == 1:
        pred_sentence_embedding = pred_sentence_embedding.reshape(1, -1)
    
    best_score = -1  # Initialize with impossible score
    best_idx = -1
    
    # Compute similarity with each golden step
    for i, reference_embedding in enumerate(reference_sentence_embeddings):
        # Ensure golden embedding is 2D
        if len(reference_embedding.shape) == 1:
            reference_embedding = reference_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(pred_sentence_embedding, reference_embedding)[0][0]
        
        # Update if this is the best score so far
        if similarity > best_score:
            best_score = similarity
            best_idx = i
    
    return best_score, best_idx, reference_strings[best_idx]


def compute_bleu_score(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)


def compute_rouge_scores(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)

def compute_best_item_bleu(pred_item, reference_items):
    best_score = 0
    for ref_item in reference_items:
        score = compute_bleu_score(ref_item, pred_item) 
        if score > best_score:
            best_score = score
    return best_score

def compute_ingredient_bleu_score(pred_ingredients, golden_ingredients):
    scores = []
    for pred in pred_ingredients:
        best = compute_best_item_bleu(pred, golden_ingredients)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0


def compute_best_item_rouge(pred_item, reference_items):
    """
    For a single predicted ingredient, compute ROUGE scores against each reference and return the best match.
    Here we take the best average of ROUGE-1 and ROUGE-L f-measures.
    """
    best_avg = 0
    best_rouge1 = 0
    best_rougeL = 0
    for ref_item in reference_items:
        scores = compute_rouge_scores(ref_item, pred_item)
        avg_score = (scores['rouge1'].fmeasure + scores['rougeL'].fmeasure) / 2
        if avg_score > best_avg:
            best_avg = avg_score
            best_rouge1 = scores['rouge1'].fmeasure
            best_rougeL = scores['rougeL'].fmeasure
    return best_rouge1, best_rougeL

def compute_ingredient_rouge_score(pred_ingredients, reference_ingredients):
    """
    Compute the average best-match ROUGE scores (both rouge1 and rougeL) for all predicted ingredients.
    """
    rouge1_scores = []
    rougeL_scores = []
    for pred in pred_ingredients:
        r1, rL = compute_best_item_rouge(pred, reference_ingredients)
        rouge1_scores.append(r1)
        rougeL_scores.append(rL)
    avg_r1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    return avg_r1, avg_rL

def compute_evals(pred_recipe, golden_recipe):
    """
    Compute evaluation metrics for the predicted recipe against the golden recipe.
    - For instructions (steps), we compare the entire concatenated string.
    - For ingredients (unordered), we use a per-item best-match approach.
    
    Metrics computed:
      * Cosine Similarity (per-item best match)
      * BLEU Score (steps: whole string; ingredients: per-item average)
      * ROUGE Scores (steps: per-item; ingredients: per-item best match averaged)
    """
    # Assume these fields are lists of strings.
    pred_steps_list = pred_recipe['steps']
    golden_steps_list = golden_recipe['instruction_steps']
    pred_ingredients_list = pred_recipe['ingredients']
    golden_ingredients_list = golden_recipe['parsed_ingredients']

    # Concatenate entire instructions for full-string comparisons.
    pred_steps_string = " ".join(pred_steps_list)
    golden_steps_string = " ".join(golden_steps_list)
    # For ingredients, we'll use the list directly for per-item matching.

    # --- Cosine Similarity ---
    cosine_scores = {"steps": [], "ingredients": []}
    for step in pred_steps_list:
        score, _, _ = compute_top_cosine_similarity(step, golden_steps_list)
        cosine_scores["steps"].append(score)
    for ingredient in pred_ingredients_list:
        score, _, _ = compute_top_cosine_similarity(ingredient, golden_ingredients_list)
        cosine_scores["ingredients"].append(score)

    # --- BLEU Scores ---
    bleu_scores = {"steps": None, "ingredients": None}
    # For steps, use the whole concatenated string.
    step_bleu_list = []
    for step in pred_steps_list:
        score = compute_bleu_score(golden_steps_string, step)
        step_bleu_list.append(score)
    bleu_steps = sum(step_bleu_list) / len(step_bleu_list) if step_bleu_list else 0
    # For ingredients, use per-item best-match BLEU.
    bleu_ingredients = compute_ingredient_bleu_score(pred_ingredients_list, golden_ingredients_list)
    bleu_scores["steps"] = bleu_steps
    bleu_scores["ingredients"] = bleu_ingredients

    # --- ROUGE Scores ---
    rouge_scores = {"steps": {"rouge1": [], "rougeL": []}, "ingredients": {"rouge1": None, "rougeL": None}}
    # For steps, compute ROUGE per step against the whole instructions.
    for step in pred_steps_list:
        scores = compute_rouge_scores(golden_steps_string, step)
        rouge_scores["steps"]["rouge1"].append(scores['rouge1'].fmeasure)
        rouge_scores["steps"]["rougeL"].append(scores['rougeL'].fmeasure)
    rouge_steps_r1 = sum(rouge_scores["steps"]["rouge1"]) / len(rouge_scores["steps"]["rouge1"]) if rouge_scores["steps"]["rouge1"] else 0
    rouge_steps_rL = sum(rouge_scores["steps"]["rougeL"]) / len(rouge_scores["steps"]["rougeL"]) if rouge_scores["steps"]["rougeL"] else 0

    # For ingredients, use per-item best-match ROUGE.
    rouge_ing_r1, rouge_ing_rL = compute_ingredient_rouge_score(pred_ingredients_list, golden_ingredients_list)

    # --- Aggregate and Return ---
    def average(lst):
        return sum(lst) / len(lst) if lst else 0

    aggregated = {
        'cosine_similarity': {
            'steps': average(cosine_scores["steps"]),
            'ingredients': average(cosine_scores["ingredients"])
        },
        'bleu_score': bleu_scores,
        'rouge_scores': {
            'steps': {
                'rouge1': rouge_steps_r1,
                'rougeL': rouge_steps_rL,
            },
            'ingredients': {
                'rouge1': rouge_ing_r1,
                'rougeL': rouge_ing_rL,
            }
        }
    }
    return aggregated