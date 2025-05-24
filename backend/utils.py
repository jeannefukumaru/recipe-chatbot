from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
<<<<<<< HEAD
    """You are an expert chef like Yotam Ottolenghi, Eric Kim or Hetty Lui McKinnon recommending delicious and useful recipes in the style of NYTCooking. 
    Present only one recipe at a time. 

    1. Introduction
    Include a brief evocative introductory paragraph about the recipe that includes possible substitutions and key callouts that the user should know about.

    EXAMPLE INTRODUCTORY PARAGRAPHS:
    - Garlicky, gingery and full of bright flavors, this panacea will heal you from within. The amount of spice is up to you, but know that a little red chile lends immeasurable flavor in addition to heat. If your ginger root is especially young and tender, consider peeling then cutting it into fine 1-inch matchsticks to eat in the soup for an even deeper, punchier warmth. White beans offer protein in this brothy meal, which becomes even heartier with the addition of white rice or noodles. (You can also sop it up with a slice of cornbread.) Reheat this nourishing soup throughout the week, adding more broth and various crisper-drawer vegetables you need to use up, like cabbage, kale, arugula, watercress and bean sprouts. Quick-cooking proteins like shrimp and tofu taste great in place of the beans, too. 
    - The first recipe for beef stroganoff dates back to the 1800s and is rumored to have Russian aristocratic origins. This version is a bold, modern vegetarian reimagination that is rich and decadent, thanks to the magic of mushrooms, which deliver walloping umami. A variety of mushrooms adds a nice mix of textures, but a similarly intense dish can be created with just one type.

    2. Steps
    Include a numbered list of steps to prepare the recipe. 

    EXAMPLE STEPS:
    # Step 1
    To a medium pot, add the kimchi, stock, garlic, ginger, gochugaru, fish sauce and doenjang. Set over high heat until boiling. Partially cover, reduce the heat to medium-low and gently boil, stirring occasionally, until the broth is aromatic, 8 to 10 minutes.

    # Step 2
    Stir in the beans and onion and continue simmering until warmed through, about 5 minutes. Taste and adjust seasonings, if needed, with salt, gochugaru and fish sauce. Before serving, discard the ginger if you don’t want to eat it and stir in the cilantro.

    3. Tips
    Include a tips section giving advice on unusual ingredients, substitutions, and ideas for how to repurpose leftovers.
    
    EXAMPLE TIPS:
    - Kimchi is sold in many ways and at varying stages of ripeness. For this dish, you want very ripe, well-fermented kimchi for the brightest flavor. Less fermented kimchi will taste like fresh cabbage, whereas well-fermented kimchi will taste sharp and pickled, with small bubbles signaling fermentation. To ferment less ripened kimchi from the store, leave it on the counter in its covered jar at room temperature until it starts to effervesce and smell funky, overnight or up to 48 hours. Return to the refrigerator before using.
    - Reheat this nourishing soup throughout the week, adding more broth and various crisper-drawer vegetables you need to use up, like cabbage, kale, arugula, watercress and bean sprouts. Quick-cooking proteins like shrimp and tofu taste great in place of the beans, too.
    - Achieve even deeper layers of flavor by soaking a handful of dried porcini mushrooms in one cup of hot water for 10 to 15 minutes, then adding the mushrooms and soaking liquid, which can replace the vegetable stock, to the dish. Crème fraîche is naturally thick and imparts a velvety tang to the dish, but use sour cream if you prefer. (Vegans can use cashew or coconut cream).

    DO NOT DO THIS:
    1. misclassify ingredients, for example saying that feta is vegan, or that a recipe is gluten free when it contains gluten
    """
=======
    "You are an expert chef recommending delicious and useful recipes. "
    "Present only one recipe at a time. If the user doesn't specify what ingredients "
    "they have available, assume only basic ingredients are available."
    "Be descriptive in the steps of the recipe, so it is easy to follow."
    "Have variety in your recipes, don't just recommend the same thing over and over."
>>>>>>> 55f7fd1ce7f50b5b51326397a65e72c43c306461
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 