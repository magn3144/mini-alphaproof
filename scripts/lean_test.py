from leantree import LeanProject, LeanTactic

project = LeanProject("lean_project")

with project.environment() as env:
    env.send_command("import Mathlib")

    branch = env.proof_from_sorry(
        "example (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := by sorry"
    )

    left, right = branch.apply_tactic(LeanTactic("constructor"))

    print(left.state)
    print(right.state)

    assert left.apply_tactic(LeanTactic("assumption")) == []
    assert right.apply_tactic(LeanTactic("assumption")) == []