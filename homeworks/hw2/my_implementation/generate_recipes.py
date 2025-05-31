import dspy
import pandas as pd
from dotenv import load_dotenv
import os
from typing import Callable
from typing import List
import mlflow
from tqdm import tqdm


load_dotenv()


SYSTEM_PROMPT = """You are an expert chef like Yotam Ottolenghi, Eric Kim or Hetty Lui McKinnon recommending delicious and useful recipes in the style of NYTCooking. 
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


def configure_lm() -> None:
    """Configure the DSPy language model."""
    # Set the environment variable for OpenAI API key
    lm: dspy.LM = dspy.LM('openai/gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'])
    dspy.configure(lm=lm)

def create_dspy_program() -> Callable[[str], str]:
    """Create the DSPy program for recipe generation.

    Returns:
        Callable[[str], str]: A function that takes a question (str) and returns a recipe (str).
    """
    generate = dspy.Predict(
        dspy.Signature(
            "question -> recipe",
            instructions=SYSTEM_PROMPT,
        )
    )
    return generate

def get_responses(queries: List[str], generate: Callable[[str], str]) -> List[str]:
    """Generate responses for the given queries.

    Args:
        queries (List[str]): A list of query strings.
        generate (Callable[[str], str]): A function that generates a recipe from a query.

    Returns:
        List[str]: A list of generated recipe strings.
    """
    mlflow.dspy.autolog()
    responses = [{"query": q, "recipe": generate(question=q).recipe} for q in tqdm(queries, desc="Generating recipes")]
    return responses


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("dspy-experiment")
    # mlflow.dspy.autolog()

    queries = pd.read_csv('homeworks/hw2/my_implementation/synthetic_queries_for_analysis.csv')["query"].tolist()[0:40]

    print("Configuring DSPy language model...")
    configure_lm()

    print("Creating DSPy program for recipe generation...")
    generate = create_dspy_program()

    
    # Generate responses for the queries
    print("Generating recipes for the queries...")
    responses = get_responses(queries, generate)
    
    # Save the responses to a CSV file
    print("Saving generated recipes to 'generated_recipes.csv'...")
    pd.DataFrame(responses).to_csv("homeworks/hw2/my_implementation/generated_recipes.csv", index=False)
    print("Recipes generated and saved to 'generated_recipes.csv'.")