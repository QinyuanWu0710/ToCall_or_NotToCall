#!/usr/bin/env python3
"""
Entity Hallucination Testing with FastMCP and Agent Loop

Main entry point for running entity hallucination tests.
This script provides a CLI interface to the testing framework.

Example usage:
    # Without web search
    python main.py --data data.csv --model-name meta-llama/Llama-3.1-8B
    
    # With web search, let the model automatically decide
    python main.py --data data.csv --model-name meta-llama/Llama-3.1-8B --web-search
    
    # With web search (using Google/Serper)
    python main.py --data data.csv --model-name meta-llama/Llama-3.1-8B --web-search --search-provider google
"""

import argparse
import os
from tester import EntityHallucinationTester
from config import AgentConfig


def main():
    parser = argparse.ArgumentParser(
        description="Test language models for entity hallucination with optional web search via FastMCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV/JSON file containing data"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="vLLM model name (e.g., meta-llama/Llama-3.1-8B)"
    )
    
    # Web search arguments
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search via MCP (requires BRAVE_API_KEY or SERPAPI_API_KEY)"
    )

    parser.add_argument(
        "--force-search",
        action="store_true",
        help="Enable force search to skip the decision loop but make the model call for a web search everytime."
    )

    parser.add_argument(
        "--search-provider",
        type=str,
        choices=["brave", "google"],
        default="google", # do not change now!
        help="Search provider (default: google)"
    )
    parser.add_argument(
        "--search-api-key",
        type=str,
        default=None,
        help="API key for search provider (overrides environment variable)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Maximum tokens to generate (default: {AgentConfig.DEFAULT_MAX_TOKENS}, auto-adjusted for some models)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=AgentConfig.DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {AgentConfig.DEFAULT_TEMPERATURE})"
    )
    
    # Model configuration
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism (default: 1)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=AgentConfig.DEFAULT_OUTPUT_DIR,
        help=f"Directory to save results (default: {AgentConfig.DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=AgentConfig.SAVE_EVERY,
        help=f"Save checkpoint every N data (default: {AgentConfig.SAVE_EVERY})"
    )
    parser.add_argument(
        "--keep-in-memory",
        action="store_true",
        help="Keep results in memory (useful for small datasets)"
    )
    
    parser.add_argument(
        "--tool-description",
        type=str,
        default=AgentConfig.DEFAULT_TOOL_DESCRIPTION_KEY,
        choices=list(AgentConfig.TOOL_DESCRIPTIONS.keys()),
        help=(
            f"Tool description variant to use for the web_search tool "
            f"(default: {AgentConfig.DEFAULT_TOOL_DESCRIPTION_KEY}). "
            f"Available: {', '.join(AgentConfig.TOOL_DESCRIPTIONS.keys())}"
        )
    )

    # Task / scoring arguments
    parser.add_argument(
        "--task",
        type=str,
        default="entity",
        choices=["entity", "bfcl"],
        help=(
            "Scoring task: 'entity' uses GPT claim-extraction + web verification; "
            "'bfcl' uses exact-match against ground-truth function calls (default: entity)"
        ),
    )
    parser.add_argument(
        "--bfcl-gt-file",
        type=str,
        default=None,
        help=(
            "Path to BFCL ground-truth JSONL file "
            "(required when --task bfcl; each line: {\"id\": ..., \"ground_truth\": ...})"
        ),
    )
    parser.add_argument(
        "--extraction-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for claim extraction (entity task, default: gpt-4o)",
    )
    parser.add_argument(
        "--verification-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for claim verification (entity task, default: gpt-4o)",
    )

    # Testing arguments
    parser.add_argument(
        "--skip-scorer",
        action="store_true",
        help="Skip the scoring step and only run inference"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of data to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate web search setup
    if args.web_search:
        if args.search_provider == "brave" and not (args.search_api_key or os.getenv("BRAVE_API_KEY")):
            parser.error("--web-search with --search-provider brave requires BRAVE_API_KEY environment variable or --search-api-key")
        elif args.search_provider == "google" and not (args.search_api_key or os.getenv("SERPAPI_API_KEY")):
            parser.error("--web-search with --search-provider google requires SERPAPI_API_KEY environment variable or --search-api-key")
    
    # Initialize configuration
    config = AgentConfig()

    # Apply selected tool description variant (must happen before any tool list is built)
    AgentConfig.set_tool_description_key(args.tool_description)
    print(f"Tool description variant: {args.tool_description}")
    print(f"  → {AgentConfig.TOOL_DESCRIPTIONS[args.tool_description]}")

    # Nest results under a per-variant subdirectory for clean separation
    output_dir = os.path.join(args.output_dir, args.tool_description)
    
    # Determine max_tokens
    if args.max_tokens is None:
        max_tokens = config.get_max_tokens_for_model(args.model_name)
        print(f"Using max_tokens={max_tokens} for model {args.model_name}")
    else:
        max_tokens = args.max_tokens
    
    # Print configuration
    print("=" * 80)
    print("Entity Hallucination Testing Configuration")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Entities file: {args.data}")
    print(f"Web search: {'Enabled' if args.web_search else 'Disabled'}")
    if args.web_search:
        print(f"Search provider: {args.search_provider}")
        print(f"Tool description: [{args.tool_description}] {AgentConfig.TOOL_DESCRIPTIONS[args.tool_description]}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Output directory: {output_dir}")
    print(f"Chat template: Loaded from model tokenizer")
    print(f"Task / scorer: {args.task}{' (skipped)' if args.skip_scorer else ''}")
    if args.task == "bfcl" and args.bfcl_gt_file:
        print(f"BFCL GT file : {args.bfcl_gt_file}")
    if args.limit:
        print(f"Entity limit: {args.limit}")
    print("=" * 80)
    print()
    
    # Build scorer_kwargs based on task
    scorer_kwargs: dict = {}
    if args.task == "entity":
        scorer_kwargs = {
            "extraction_model": args.extraction_model,
            "verification_model": args.verification_model,
        }
    elif args.task == "bfcl":
        if args.bfcl_gt_file:
            scorer_kwargs = {"ground_truth_file": args.bfcl_gt_file}
        else:
            print(
                "Warning: --task bfcl without --bfcl-gt-file — "
                "ground truth must be supplied per-sample via the data file."
            )

    # Initialize tester
    tester = EntityHallucinationTester(
        output_dir=output_dir,
        config=config
    )

    # Run test
    try:
        results = tester.run_test(
            data_file=args.data,
            model_name=args.model_name,
            max_tokens=max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_web_search=args.web_search,
            force_search=args.force_search,
            search_provider=args.search_provider,
            api_key=args.search_api_key,
            save_every=args.save_every,
            keep_in_memory=args.keep_in_memory,
            limit=args.limit,
            task=args.task,
            scorer_kwargs=scorer_kwargs,
            skip_scorer=args.skip_scorer,
        )

        
        if args.keep_in_memory:
            print(f"\nGenerated {len(results)} responses")
        
        print("\n" + "=" * 80)
        print("Testing completed successfully!")
        print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        print("Partial results have been saved")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())