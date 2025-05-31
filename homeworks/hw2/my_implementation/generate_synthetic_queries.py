from pydantic import BaseModel

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

load_dotenv()


class DimensionTuple(BaseModel):
    occasion: str
    author_style: str
    ingredients: List[str]
    cooking_method: str

class QueryWithDimensions(BaseModel):
    id: str
    query: str
    dimension_tuple: DimensionTuple
    is_realistic_and_kept: int = 1
    notes_for_filtering: str = ""

class DimensionTuplesList(BaseModel):
    tuples: List[DimensionTuple]

class QueriesList(BaseModel):
    queries: List[str]

MODEL_NAME = "gpt-4o-mini"
NUM_TUPLES_TO_GENERATE = 20
NUM_QUERIES_PER_TUPLE = 5
OUTPUT_CSV_PATH = Path(__file__).parent / "synthetic_queries_for_analysis.csv"
MAX_WORKERS = 5  # Number of parallel LLM calls

def call_llm(messages: List[Dict[str, str]], response_format: Any) -> Any:
    """Make a single LLM call with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=MODEL_NAME,
                messages=messages,
                response_format=response_format
            )
            return response_format(**json.loads(response.choices[0].message.content))
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Wait before retry

def generate_dimension_tuples():
    """Create a dimension tuple object from the given parameters. The DimensionTuple class 
    will then be used to generate synthetic queries for recipe generation."""

    PROMPT = """You are helping to generate synthetic queries for recipe generation. Recipes are meant to reflect 
    the style of various NYTimes Cooking authors, and the queries should be realistic and diverse.

    You will be given a set of parameters that describe the query, and you will return a
    DimensionTuple object that contains the occasion, author style, ingredients, and cooking method.

    Important: Aim for an even distribution across all dimensions. For example:
    - Don't focus too heavily on quick recipes
    - Don't over-represent any particular cuisine
    - Vary the query styles naturally
    - Try to use interesting flavor and ingredient combinations suitable for the occasion

    Here are the parameters you can use to create the DimensionTuple object:
    [RECIPE OCCASION] 
    - weeknight
    - brunch
    - summer picnic
    - dinner party
    - breakfast
    - dessert

    [AUTHOR STYLE]
    - Hetty McKinnon
    - Yotam Ottolenghi
    - Eric Kim

    [COOKING METHOD]
    - roasting
    - grilling
    - sautéing
    - slow cooking
    - no bake
    - no cook

    [INGREDIENTS]
    - Vegetables: sweet potato, Eggplant, Cabbage, Asparagus, Tomato, Zuchini, Spinach, kimchi
    - Grains: Rice, Quinoa, Barley, Oats, Pasta
    - Proteins: Chicken, Tofu, Lentils, Beans, Fish
    - Sweets: Chocolate, Ice Cream, Cake, Cookies, Brownies

    Here are some examples of parameters and their corresponding DimensionTuple objects:
    - summer vibes, make-ahead salad
    {{class DimensionTuple(BaseModel):
    occasion: summer picnic
    author_style: Hetty McKinnon
    ingredients: tomato, zuchini, spinach
    cooking_method: no cook}}

    - summer picnic, vegetarian, no bake
    {{class DimensionTuple(BaseModel):
    occasion: summer picnic
    author_style: Yotam Ottolenghi
    ingredients: eggplant, barley, fish
    cooking_method: str}}

    - comforting stew, carb-heavy, slow cooking
    {{class DimensionTuple(BaseModel):
    occasion: dinner party
    author_style: Eric Kim
    ingredients: kimchi
    cooking_method: slow cooking}}

    - Korean-American fusion, weeknight dinner, grilling
    {{class DimensionTuple(BaseModel):
    occasion: weeknight
    author_style: Eric Kim
    ingredients: Lentils, gochujang, cabbage
    cooking_method: grilling}}

    - Yotam Ottolenghi, minimalist ingredients, brunch
    {{class DimensionTuple(BaseModel):
    occasion: brunch
    author_style: Yotam Ottolenghi
    ingredients: oats, chocolate
    cooking_method: no bake}}

    - Middle Eastern-inspired dessert for dinner party
    {{class DimensionTuple(BaseModel):
    occasion: dinner party
    author_style: Yotam Ottolenghi
    ingredients: fish, quinoa, zuchini
    cooking_method: sautéing}}

    Generate {NUM_TUPLES_TO_GENERATE} unique dimension tuples following these patterns. Remember to maintain balanced diversity across all dimensions.
    """
    messages = [{"role": "user", "content": PROMPT}]

    try:
        print("Generating dimension tuples in parallel...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit five generation tasks using a loop
            futures = []
            for _ in range(5):
                futures.append(executor.submit(call_llm, messages, DimensionTuplesList))
            
            # Wait for all to complete and collect results
            responses = []
            for future in futures:
                responses.append(future.result())
        
        # Combine tuples and remove duplicates
        all_tuples = []
        for response in responses:
            all_tuples.extend(response.tuples)
        unique_tuples = []
        seen = set()
        
        for tup in all_tuples:
            # Convert tuple to a comparable string representation
            tuple_str = tup.model_dump_json()
            if tuple_str not in seen:
                seen.add(tuple_str)
                unique_tuples.append(tup)
        
        print(f"Generated {len(all_tuples)} total tuples, {len(unique_tuples)} unique")
        return unique_tuples
    except Exception as e:
        print(f"Error generating dimension tuples: {e}")
        return []
    
def generate_queries_for_tuple(dimension_tuple: DimensionTuple) -> List[str]:
    """Generate natural language queries for a given dimension tuple."""
    prompt = f"""Generate {NUM_QUERIES_PER_TUPLE} different natural language queries for a recipe chatbot based on these characteristics:
{dimension_tuple.model_dump_json(indent=2)}

The queries should:
1. Sound like real users asking for recipe help
2. Naturally incorporate all the dimension values
3. Vary in style and detail level
4. Be realistic and practical
5. Include natural variations in typing style, such as:
   - Some queries in all lowercase
   - Some with random capitalization
   - Some with common typos
   - Some with missing punctuation
   - Some with extra spaces or missing spaces
   - Some with emojis or text speak

Here are examples of realistic query variations for a weeknight easy recipe:

Proper formatting:
- "Need a simple dinner that's ready in 20 minutes"
- "What's an easy plant-based recipe I can make quickly?"

All lowercase:
- "feeling like cooking something quick and easy"
- "spinach and tofu for dinner"

Random caps:
- "NEED a Quick Vegan DINNER recipe"
- "what's an EASY plant based recipe i can make"

Common typos:
- "summer picnic ideas, no cokking"
- "summer zuchinni salad recipe"

With emojis/text speak:
- "need lunch! 🥗"
- "chocolate dessert ideas plz 🍫"

Generate {NUM_QUERIES_PER_TUPLE} unique queries that match the given dimensions, varying the text style naturally."""

    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_llm(messages, QueriesList)
        return response.queries
    except Exception as e:
        print(f"Error generating queries for tuple: {e}")
        return []

def generate_queries_parallel(dimension_tuples: List[DimensionTuple]) -> List[QueryWithDimensions]:
    """Generate queries in parallel for all dimension tuples."""
    all_queries = []
    query_id = 1
    
    print(f"Generating {NUM_QUERIES_PER_TUPLE} queries each for {len(dimension_tuples)} dimension tuples...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all query generation tasks
        future_to_tuple = {
            executor.submit(generate_queries_for_tuple, dim_tuple): i 
            for i, dim_tuple in enumerate(dimension_tuples)
        }
        
        # Process completed generations as they finish
        with tqdm(total=len(dimension_tuples), desc="Generating Queries") as pbar:
            for future in as_completed(future_to_tuple):
                tuple_idx = future_to_tuple[future]
                try:
                    queries = future.result()
                    if queries:
                        for query in queries:
                            all_queries.append(QueryWithDimensions(
                                id=f"SYN{query_id:03d}",
                                query=query,
                                dimension_tuple=dimension_tuples[tuple_idx]
                            ))
                            query_id += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Tuple {tuple_idx + 1} generated an exception: {e}")
                    pbar.update(1)
    
    return all_queries

def save_queries_to_csv(queries: List[QueryWithDimensions]):
    """Save generated queries to CSV using pandas."""
    if not queries:
        print("No queries to save.")
        return

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'id': q.id,
            'query': q.query,
            'dimension_tuple_json': q.dimension_tuple.model_dump_json(),
            'is_realistic_and_kept': q.is_realistic_and_kept,
            'notes_for_filtering': q.notes_for_filtering
        }
        for q in queries
    ])
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(queries)} queries to {OUTPUT_CSV_PATH}")

def main():
    """Main function to generate and save queries."""
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    start_time = time.time()
    
    # Step 1: Generate dimension tuples
    print("Step 1: Generating dimension tuples...")
    dimension_tuples = generate_dimension_tuples()
    if not dimension_tuples:
        print("Failed to generate dimension tuples. Exiting.")
        return
    print(f"Generated {len(dimension_tuples)} dimension tuples.")
    
    # Step 2: Generate queries for each tuple
    print("\nStep 2: Generating natural language queries...")
    queries = generate_queries_parallel(dimension_tuples)
    
    if queries:
        save_queries_to_csv(queries)
        elapsed_time = time.time() - start_time
        print(f"\nQuery generation completed successfully in {elapsed_time:.2f} seconds.")
        print(f"Generated {len(queries)} queries from {len(dimension_tuples)} dimension tuples.")
    else:
        print("Failed to generate any queries.")
    
if __name__ == "__main__":
    main()