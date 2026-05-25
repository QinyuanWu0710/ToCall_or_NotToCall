"""
Entity extraction and verification pipeline using Wolfram Language Entity Types and GPT-4o.
- Reads messages from CSV
- Extracts named entities
- Classifies using Wolfram Entity Types
- Verifies ONLY high-level category classifications using web-grounded reasoning (separate step)
- Saves one JSON per message
"""
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =========================
# ARGUMENTS
# =========================
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="gpt-5-nano-2025-08-07")
argparser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
argparser.add_argument(
    "--input_csv",
    type=str,
    default="/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_conversations.csv",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    default="/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_entities",
)
argparser.add_argument(
    "--skip_verification",
    action="store_true",
    help="Skip verification step (extraction only)"
)
args = argparser.parse_args()

MODEL_NAME = args.model_name
INPUT_CSV = args.input_csv
OUTPUT_DIR = args.output_dir
SKIP_VERIFICATION = args.skip_verification

os.makedirs(OUTPUT_DIR, exist_ok=True)
client = OpenAI(api_key=args.api_key)

# =========================
# JSON SCHEMAS
# =========================
EXTRACTION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_text": {"type": "string"},
                            "entity_type": {"type": "string"},
                            "entity_type_category": {"type": "string"}
                        },
                        "required": ["entity_text", "entity_type", "entity_type_category"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
}

VERIFICATION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_verification",
        "schema": {
            "type": "object",
            "properties": {
                "verified": {"type": "boolean"},
                "correct_entity_type_classification": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["verified", "correct_entity_type_classification", "reason"]
        }
    }
}

# =========================
# PROMPTS
# =========================

PROMPT_ENTITY_EXTRACTION = """
You are an expert entity annotator. Your task:

1. Extract all named entities from the given text.
2. Classify each entity using Wolfram Language Entity Types listed below.

VALID WOLFRAM ENTITY TYPES:

### Geographic Entities
Country, AdministrativeDivision, City, Neighborhood, MetropolitanArea, ZIPCode, USCongressionalDistrict, DistrictCourt, Ocean, Island, UnderseaFeature, Reef, Beach, Lake, Mountain, Volcano, River, Glacier, Waterfall, EarthImpact, Desert, Forest, GeographicRegion, Airport, Park, AmusementPark, AmusementParkRide, Stadium, Bridge, Canal, Tunnel, Dam, Mine, Cave, OilField, Building, Castle, Cemetery, HistoricalSite, PreservationStatus, ReserveLand, Shipwreck, University, SchoolDistrict, PublicSchool, PrivateSchool, Museum, LibrarySystem, LibraryBranch, WeatherStation, AstronomicalObservatory, ParticleAccelerator, NuclearReactor, NuclearTestSite, NuclearExplosion, TimeZone

### Astronomical Entities
Planet, PlanetaryMoon, MinorPlanet, Comet, SolarSystemFeature, MeteorShower, Exoplanet, Star, Galaxy, StarCluster, Nebula, Supernova, Pulsar, AstronomicalRadioSource, Constellation

### Space-Related
Satellite, Rocket, DeepSpaceProbe, MannedSpaceMission

### Weather & Earth Science
WeatherStation, TropicalStorm, Cloud, AtmosphericLayer, Earthquake, GeologicalLayer, GeologicalPeriod, Mineral, FamousGem, TidalConstituent, TideStation

### Transportation-Related
Aircraft, Airline, Airport, Ship

### Engineering & Structures
FrequencyAllocation, BroadcastStation, MeasurementDevice, Building, Bridge, Tunnel, Dam, Mine

### Dynamical Systems
SystemModel

### Culture & Entertainment
Language, Religion, Mythology, Movie, MusicAct, MusicAlbum, MusicAlbumRelease, MusicWork, MusicWorkRecording, BroadcastStation, BroadcastStationClassification, Book, Artwork, Periodical, FictionalCharacter, Museum, LibraryBranch, LibrarySystem

### Activities & Hobbies
SportObject, SportMatch, MusicalInstrument, BoardGame, PopularCurve, YogaPose, YogaPosition, YogaSequence, YogaProp, PilatesExercise, Pokemon, Digimon

### Finance-related
Financial, Company, CurrencyDenomination

### Food & Nutrition
Food, FoodType, BasicFoodGroup, USDAFoodGroup, FoodTypeGroup, FoodAlcoholLabel, FoodCaffeineLabel, FoodCalorieLabel, FoodFiberLabel, FoodFatLabel, FoodIronLabel, FoodSodiumLabel, FoodSugarLabel, FoodBoneContent, FoodSkinContent, FoodSeedContent, FoodCrustType, FoodFatType, FoodGeometryType, FoodPeelingType, FoodProcessingType, FoodServingType, FoodStorageType, FoodSugarType, FoodBeefGrade, FoodMeatCut, FoodMeatQuality, FoodPattyCount, FoodBrandName, FoodSubBrandName, FoodManufacturer, FoodAge, FoodComposition, FoodConcentration, FoodCulture, FoodDataSource, FoodFlavor, FoodIntendedUse, FoodLocation, FoodMoistureLevel, FoodNutritionalSupplement, FoodNutritionalSupplementNotAdded, FoodPackaging, FoodPart, FoodPreparation, FoodSeafoodVariety, FoodSize, FoodState, FoodTexture, FoodTrimmingLevel, FoodVariety, FoodVegetablePart

### People & Personal Attributes
Person, PersonTitle, GivenName, Surname, Gender, Emotion

### History-Related
HistoricalCountry, HistoricalEvent, HistoricalPeriod, HistoricalSite, Shipwreck, MilitaryConflict

### Linguistic Entities
Language, Word, GrammaticalUnit, WritingScript, Alphabet, Character, Concept, WritingDirection, WritingScriptBaseline, WritingScriptType

### Physical Sciences
Chemical, Element, Isotope, Particle, Mineral, Laser, CrystalFamily, CrystalSystem, CrystallographicSpaceGroup, PhysicalSystem, PhysicalConstant, FamousPhysicsProblem, FamousChemistryProblem, Color, ColorSet, LightColor, MeasurementDevice

### Life Sciences
Gene, SNP, Protein

### Medical Entities
AnatomicalStructure, AnimalAnatomicalStructure, Neuron, Disease, MedicalTest, Protein, AnatomicalFunctionalConcept, AnatomicalTemporalConcept, CognitiveTask, ICDNine, ICDTen

### Organism Types
Plant, Species, Dinosaur, DogBreed, CatBreed

### Mathematical & Computational Entities
Polyhedron, Solid, Lamina, Surface, SpaceCurve, PlaneCurve, Lattice, LatticeSystem, PeriodicTiling, NonperiodicTiling, Graph, Knot, FiniteGroup, MathematicalFunction, IntegerSequence, ContinuedFraction, FunctionSpace, TopologicalSpaceType, FamousMathProblem, FamousMathGame, ComputationalComplexityClass, MathWorld, ContinuedFractionResult, ContinuedFractionSource, FunctionalAnalysisSource, GeometricScene

### Computing-Related Entities
FileFormat, DisplayFormat, NotableComputer, InternetDomain, IPAddress, NetworkService, TopLevelDomain, ProgrammingLanguage, WolframLanguageSymbol

### Visual Entities
Icon, Color, ColorSet, LightColor

For each entity return:
- entity_text: exact text span from the input text
- entity_type: one of the Wolfram entity types listed above
- entity_type_category: high-level category (Geographic Entities, Astronomical Entities, Space-Related, Weather & Earth Science, Transportation-Related, Engineering & Structures, Dynamical Systems, Culture & Entertainment, Activities & Hobbies, Finance-related, Food & Nutrition, People & Personal Attributes, History-Related, Linguistic Entities, Physical Sciences, Life Sciences, Medical Entities, Organism Types, Mathematical & Computational Entities, Computing-Related Entities, Visual Entities)

Rules:
- Do NOT hallucinate entities that are not in the text.
- If an entity type is ambiguous, choose the most specific applicable Wolfram type.
- Extract only entities explicitly mentioned in the text.
- If unsure about the entity type, use the closest matching type from the list above.
- Use exact entity type names as listed (case-sensitive).

Return valid JSON only.
"""

PROMPT_ENTITY_VERIFICATION = """
You are an entity verification expert. Your task is to verify whether the HIGH-LEVEL CATEGORY classification is correct using real-world knowledge.

Original Text:  {original_text}
Entity: {entity_text}
Classified Category: {entity_type_category}

VALID HIGH-LEVEL CATEGORIES:
Geographic Entities, Astronomical Entities, Space-Related, Weather & Earth Science, Transportation-Related, Engineering & Structures, Dynamical Systems, Culture & Entertainment, Activities & Hobbies, Finance-related, Food & Nutrition, People & Personal Attributes, History-Related, Linguistic Entities, Physical Sciences, Life Sciences, Medical Entities, Organism Types, Mathematical & Computational Entities, Computing-Related Entities, Visual Entities


Verify the following:
1. Is "{entity_text}" a real entity (not fictional in a way that doesn't exist, not hallucinated)?
2. Is the HIGH-LEVEL CATEGORY "{entity_type_category}" correct? (Ignore the specific sub-type)

Return:
- verified: true if the entity is real AND the high-level category is correct, false otherwise
- reason: brief explanation focusing on whether the category is appropriate (e.g., "Correct: Paris is indeed a Geographic Entity", "Incorrect: Python is a Computing-Related Entity, not a Transportation-Related entity", "False entity: No such place exists")

Note: You are NOT verifying the specific entity_type (like City vs Country), only the broader category. Be strict about the category classification and whether the entity exists.
"""

# =========================
# LLM CALLS
# =========================
def extract_entities(text: str):
    """Extract entities without verification"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format=EXTRACTION_JSON_SCHEMA,
        messages=[
            {"role": "system", "content": "You are a precise entity extraction system."},
            {"role": "user", "content": PROMPT_ENTITY_EXTRACTION + "\n\nTEXT:\n" + text}
        ],
        temperature=0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decoding failed: {e}")
        return {"entities": []}

def verify_and_correct_entity_category(original_text: str, entity_text: str, entity_type_category: str):
    """Verify the high-level category and return possibly corrected category"""
    
    prompt = f"""
You are an entity verification expert. Your task is to verify whether the HIGH-LEVEL CATEGORY classification is correct using real-world knowledge.

Original Text: {original_text}
Entity: {entity_text}
Classified Category: {entity_type_category}

VALID HIGH-LEVEL CATEGORIES:
Geographic Entities, Astronomical Entities, Space-Related, Weather & Earth Science, Transportation-Related, Engineering & Structures, Dynamical Systems, Culture & Entertainment, Activities & Hobbies, Finance-related, Food & Nutrition, People & Personal Attributes, History-Related, Linguistic Entities, Physical Sciences, Life Sciences, Medical Entities, Organism Types, Mathematical & Computational Entities, Computing-Related Entities, Visual Entities

Tasks:
1. Verify if "{entity_text}" is a real entity (not fictional or hallucinated).
2. Verify if the HIGH-LEVEL CATEGORY "{entity_type_category}" is correct.
3. If the category is incorrect, provide the correct high-level category from the valid list above.

Return JSON strictly in this format:
{{
    "verified": true/false,  # true if real entity AND category correct
    "reason": "explanation of verification",
    "correct_entity_type_classification": "Correct category if different, otherwise same as input"
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format=VERIFICATION_JSON_SCHEMA,
        messages=[
            {"role": "system", "content": "You are an entity verification expert with access to real-world knowledge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        verification = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        verification = {
            "verified": False,
            "reason": "Failed to parse LLM response",
            "correct_entity_type_classification": None
        }

    # Update entity_type_category if LLM suggested a correction
    if not verification.get("verified", False) and verification.get("correct_entity_type_classification"):
        entity_type_category = verification["correct_entity_type_classification"]

    return verification, entity_type_category


# =========================
# MAIN PIPELINE
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)
    required_cols = {"user_id", "conversation_id","message_id", "message_role","message_content"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Filter out rows with missing or empty message_content
    df = df[~df["message_content"].isna()]           # remove NaNs
    df = df[df["message_content"].str.strip() != ""] # remove empty strings

    # Only keep the user query and the assistant response, remove the system message and tool calling message
    df = df[df['message_role'].isin(['user', 'assistant'])]

    # now only keep the first 100 users:
    # df = df[df['user_id']<101]

    print(f'Number of sequences need to annotate: {len(df)}')

    # Now iterate over filtered messages
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing messages"):
        user_id = row["user_id"]
        conversation_id = row["conversation_id"]
        message_id = row["message_id"]
        message_content = str(row["message_content"])
        
        output_path = os.path.join(OUTPUT_DIR, f"user-{user_id}_msg-{message_id}.json")
        if os.path.exists(output_path):
            continue
        try:
            # Step 1: Extract entities
            entity_result = extract_entities(message_content)
            entities = entity_result["entities"]

            # Step 2: Verify & correct each entity's CATEGORY (if not skipped)
            if not SKIP_VERIFICATION:
                for entity in entities:
                    verification, corrected_category = verify_and_correct_entity_category(
                        message_content,
                        entity["entity_text"],
                        entity["entity_type_category"]
                    )
                    # entity["entity_type_category"] = corrected_category
                    entity["verification"] = verification
            else:
                for entity in entities:
                    entity["verification"] = {
                        "verified": None,
                        "reason": "Verification skipped"
                    }

            # Save result
            output_data = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "message_content": message_content,
                "entities": entities
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\n[ERROR] user_id={user_id}, message_id={message_id}: {e}")
            continue


if __name__ == "__main__":
    main()