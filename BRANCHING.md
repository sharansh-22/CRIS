# CRIS — Branching Strategy

## Core Rules
- main is always clean and always works
- Layer code lives on feature branches
- Input data lives on main in data/
- Generated outputs live inside each layer folder
- Layer outputs only appear on main after merge

## Branch Naming
  feature/bssc          Layer 3 BSSC
  feature/mmad          Layer 2 MMAD
  feature/signal        Layer 1 Signal Harvester
  feature/credit        Layer 4 Credit Risk
  feature/convergence   Convergence Engine
  feature/dashboard     Dashboard

## Test Hierarchy
  CRIS_TESTS.md              CRIS-level stress tests only
  CRIS_TEST_RESULTS.md       CRIS-level stress results only
    ↑ fed by
  layer*/tests/*_TESTS.md         component-level tests
  layer*/tests/*_TEST_RESULTS.md  component-level results

## Documentation on main
  docs/*_ADR.md    Architecture Decision Record per layer
                   Posted before branch work starts
                   Updated throughout development
                   Contains every hypothesis and change
  docs/*_FLOW.md   Diagram and theory per layer
                   Posted before branch work starts
                   Updated as design evolves

## Merge Requirements
  All component files complete
  All component tests passing
  BSSC_TESTS.md and BSSC_TEST_RESULTS.md updated
  CRIS_TESTS.md updated with layer stress test
  CRIS_TEST_RESULTS.md updated with results
  docs/ ADR and FLOW updated to final state

## Pull Request Process
  1. Push final commits to feature branch
  2. Open Pull Request: feature/x → main
  3. Title: feat(layer): [name] complete
  4. Review diff
  5. Merge to main
  6. Delete feature branch
  7. Update CRIS_TREE.md on main

## What main History Will Look Like
  feat(bssc): Layer 3 BSSC complete
  feat(mmad): Layer 2 MMAD complete
  feat(signal): Layer 1 Signal Harvester complete
  feat(credit): Layer 4 Credit Risk complete
  feat(convergence): Convergence Engine complete
  feat(dashboard): Dashboard — repository goes public
