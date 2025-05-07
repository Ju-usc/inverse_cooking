


def check_format(text: str) -> float:
    """
    Strictly validate that the model’s text is in the form

        <think> … </think>
        <recipe>
          <title>…</title>
          <ingredients>
            <ingredient> … </ingredient>+
          </ingredients>
          <instructions>
            <step> … </step>+
          </instructions>
        </recipe>

    If – and only if – **all** structural constraints are met, return 1.0;
    otherwise return 0.0.  (GRPO expects a deterministic binary reward.)
    """

    try:

        text = completion.strip()

        # ──────────────────────────────────────────────────────────────
        # Regex design notes
        # ──────────────────────────────────────────────────────────────
        #  • ^ … $         → anchor entire string (no junk before/after)
        #  • (?:(?!<tag>).)*?  → tempered greedy token: consumes anything
        #                       *except* the opening of that <tag> again,
        #                       so we guarantee “exactly one” occurrence.
        #  • [^<]+          → at least one character inside the leaf tags
        #  • DOTALL flag    → "." matches newlines so pretty printing works
        #  • IGNORECASE     → tags are case-insensitive (conservative choice)
        #
        # We only insist on *one* <ingredient> and *one* <step> to keep the
        # check simple; the generator is free to add more – the look-ahead
        # guards allow multiple as long as the outer structure is intact.
        # ──────────────────────────────────────────────────────────────
        recipe_regex = r"""
            ^<think>                             # single think section …
                (?:(?!<think>).)*?               #   … no nested <think>
            </think>\s*                          # end </think> (trim ws)
            <recipe>                             # single recipe section
                (?:(?!<recipe>).)*?              #   … no nested <recipe>

                <title>[^<]+</title>             # required title text

                (?:(?!<recipe>).)*?              # anything until ingredients
                <ingredients>                    # open ingredients
                    (?:(?!</ingredients>).)*?    #   stuff but not </ingredients>
                    <ingredient>[^<]+</ingredient> # ≥1 ingredient
                    (?:(?!</ingredients>).)*?    #   (possibly more)
                </ingredients>                   # close ingredients

                (?:(?!<recipe>).)*?              # anything until instructions
                <instructions>                   # open instructions
                    (?:(?!</instructions>).)*?   #   stuff but not </instructions>
                    <step>[^<]+</step>           # ≥1 step
                    (?:(?!</instructions>).)*?   #   (possibly more)
                </instructions>                  # close instructions

                (?:(?!<recipe>).)*?              # anything else but no new <recipe>
            </recipe>$                           # close recipe – end of string
        """

        # Compile once for speed and readability
        pattern = re.compile(recipe_regex, re.DOTALL | re.IGNORECASE | re.VERBOSE)
        return 1.0 if pattern.search(text) else 0.0

    except Exception:        # any parsing or regex failure ⇒ 0 reward
        return 0.0


def _avg_best_cosine(
    pred_items: List[str],
    golden_items: List[str],
    golden_embeddings: List[np.ndarray],
) -> float:
    """
    For every *predicted* item find its best cosine similarity against the
    *golden* items and average the scores.

    Returns 0 when either side is empty.
    """
    if not pred_items or not golden_items:
        return 0.0

    scores: List[float] = []

    # (Pre-encode golden once → ndarray list)
    golden_emb_arr = [np.asarray(e).reshape(1, -1) for e in golden_embeddings]

    for item in pred_items:
        try:
            pred_emb = embedder.encode([item]).reshape(1, -1)
        except Exception:
            # encoding failed – skip this item
            continue

        # find best similarity to any golden item
        best = max(
            cosine_similarity(pred_emb, gold_emb)[0][0] for gold_emb in golden_emb_arr
        )
        scores.append(best)

    return float(np.mean(scores)) if scores else 0.0



def _extract_recipe_xml(text: str) -> str | None:
    """Return the *first* <recipe>…</recipe> block or None."""
    _xml_pat = re.compile(r"<recipe>[\s\S]*?</recipe>", re.IGNORECASE)  # first recipe only
    m = _xml_pat.search(text)
    return m.group(0) if m else None



# ──────────────────────────────────────────────────────────────────────────────
# GRPO-ready reward callables
# ──────────────────────────────────────────────────────────────────────────────
def cosine_ingredients_reward(completions: List[List[dict]], **kwargs) -> List[float]:
    """
    Soft reward in [0,1] based on average best-match cosine similarity of the
    *ingredient* lines.  Expects the dataset batch to supply

        kwargs["parsed_ingredients"]      # List[List[str]]
        kwargs["ingredients_embeddings"]  # List[List[np.ndarray]]

    """
    gold_ing_list   = kwargs["parsed_ingredients"]      # batch-aligned
    gold_ing_embeds = kwargs["ingredients_embeddings"]  # batch-aligned

    rewards: List[float] = []

    for i, comp in enumerate(completions):
        text = comp[0]["content"]

        # -------- parse predicted recipe ----------
        xml_block = _extract_recipe_xml(text)
        if xml_block is None:
            rewards.append(0.0)
            continue

        parsed = parse_recipe_xml(xml_block)
        if parsed is None:
            rewards.append(0.0)
            continue

        pred_ingredients = parsed["ingredients"]

        # -------- compute reward ----------
        score = _avg_best_cosine(
            pred_ingredients,
            gold_ing_list[i],
            gold_ing_embeds[i],
        )

        # Map cosine range [-1,1] → [0,1]  (MiniLM usually >=0, but be safe)
        rewards.append(max(0.0, score))

    return rewards



def cosine_steps_reward(completions: List[List[dict]], **kwargs) -> List[float]:
    """
    Same idea as above, but for instruction *steps*.

    Requires batch kwargs:
        kwargs["instruction_steps"]
        kwargs["instructions_embeddings"]
    """
    gold_steps_list   = kwargs["instruction_steps"]
    gold_steps_embeds = kwargs["instructions_embeddings"]

    rewards: List[float] = []

    for i, comp in enumerate(completions):
        text = comp[0]["content"]

        xml_block = _extract_recipe_xml(text)
        if xml_block is None:
            rewards.append(0.0)
            continue

        parsed = parse_recipe_xml(xml_block)
        if parsed is None:
            rewards.append(0.0)
            continue

        pred_steps = parsed["steps"]

        score = _avg_best_cosine(
            pred_steps,
            gold_steps_list[i],
            gold_steps_embeds[i],
        )

        rewards.append(max(0.0, score))

    return rewards


