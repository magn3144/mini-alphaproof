#!/usr/bin/env python3
"""
Interactive manual testing script for the Lean 4 environment.
Navigate the proof environment tactic by tactic as a human.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.environment import Lean4Environment


def print_divider():
    """Print a visual divider."""
    print("\n" + "="*70 + "\n")


def print_help():
    """Print help message."""
    print("Commands:")
    print("  <tactic>        - Apply a tactic (e.g., 'intro n', 'norm_num')")
    print("  state           - Show current proof state")
    print("  stats           - Show proof statistics")
    print("  mode [format]   - Switch display mode: human (default), llm, or json")
    print("  help            - Show this help message")
    print("  quit            - Exit the interactive session")


def main():
    """Run the interactive testing session."""
    print("="*70)
    print("Lean 4 Interactive Proof Environment")
    print("="*70)

    # Get theorem from user or use default
    print("\nEnter a theorem to prove (or press Enter for default example):")
    print("Example: theorem ex : 1 + 1 = 2 := by sorry")

    theorem_input = input("\nTheorem: ").strip()

    if not theorem_input:
        theorem = "theorem ex : 1 + 1 = 2 := by sorry"
        print(f"Using default: {theorem}")
    else:
        theorem = theorem_input

    print_divider()

    # Initialize environment
    try:
        print("Initializing Lean environment...")
        env = Lean4Environment(
            theorem_statement=theorem,
            verbose=False
        )
        print("Environment initialized successfully!")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return 1

    print_divider()

    # Display mode setting
    display_mode = "human"  # Options: "human", "llm", "json"
    print(f"Display mode: {display_mode}")
    print("\nInitial proof state:")
    print(env.render(mode=display_mode))
    print_divider()

    print_help()
    print_divider()

    # Main interaction loop
    try:
        while True:
            # Check if proof is complete
            if env.is_complete():
                print("\n🎉 PROOF COMPLETE! 🎉")
                print(f"Completed in {env.steps_taken} steps")
                print("\nRestart the script to try another theorem.")
                break

            # Get user input
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            print()  # Empty line for readability

            # Handle special commands
            if command in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            elif command == "help":
                print_help()

            elif command == "state":
                print(f"Display mode: {display_mode}")
                print(env.render(mode=display_mode))

            elif command == "mode":
                if args and args.lower() in ["human", "llm", "json"]:
                    display_mode = args.lower()
                    print(f"✓ Display mode set to: {display_mode}")
                    print("\nCurrent proof state:")
                    print(env.render(mode=display_mode))
                elif args:
                    print(f"✗ Invalid mode: {args}")
                    print("Valid modes: human, llm, json")
                else:
                    print(f"Current display mode: {display_mode}")
                    print("Available modes: human, llm, json")
                    print("\nExamples:")
                    print("  mode llm   - Switch to LLM-friendly format")
                    print("  mode human - Switch to human-readable format")
                    print("  mode json  - Switch to JSON format")

            elif command == "stats":
                stats = env.get_stats()
                print("Proof Statistics:")
                print(f"  Steps taken: {stats['steps_taken']}")
                print(f"  Goals remaining: {stats['num_goals']}")
                print(f"  Proof complete: {stats['proof_complete']}")

            else:
                # Treat as a tactic
                tactic = user_input
                result = env.step(tactic)

                if result.success:
                    print(f"✓ Tactic '{tactic}' succeeded")
                    print_divider()
                    print(env.render(mode=display_mode))
                else:
                    print(f"✗ Tactic '{tactic}' failed")
                    if result.error_message:
                        print(f"Error: {result.error_message}")

                print_divider()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        env.close()
        print("Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
