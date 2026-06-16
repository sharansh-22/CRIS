.PHONY: help setup clean run_full_cris run_experiments run_certifications audit

help:
	@echo "CRIS Institutional Setup & Execution System"
	@echo "==========================================="
	@echo "make setup              - Create conda environment and install dependencies"
	@echo "make run_full_cris      - Execute the end-to-end CRIS pipeline (Macro, Credit, Validation)"
	@echo "make run_portfolio      - Execute the Portfolio Intelligence System pipeline"
	@echo "make run_experiments    - Run Governance Lab Experiments 01-09"
	@echo "make run_certifications - Run Institutional Audits and Stress Certification (IVSC)"
	@echo "make clean              - Remove all outputs, logs, and artifacts"
	@echo "make audit              - Run system integrity import audit"

setup:
	conda env create -f environment.yml

run_full_cris:
	conda run -n CRIS python orchestration/run_full_cris.py

run_portfolio:
	conda run -n CRIS python orchestration/portfolio/run_portfolio_intelligence.py

run_experiments:
	conda run -n CRIS python validation/governance_lab/experiments/01_sensitivity_sweep.py
	conda run -n CRIS python validation/governance_lab/experiments/02_recovery_velocity.py
	conda run -n CRIS python validation/governance_lab/experiments/03_source_dependent.py
	conda run -n CRIS python validation/governance_lab/experiments/04_temporal_cohesion.py
	conda run -n CRIS python validation/governance_lab/experiments/04_5_utility_sensitivity.py
	conda run -n CRIS python validation/governance_lab/experiments/05_unified_synthesis.py
	conda run -n CRIS python validation/governance_lab/experiments/06_governance_explainability.py
	conda run -n CRIS python validation/governance_lab/experiments/07_governance_replay.py
	conda run -n CRIS python validation/governance_lab/experiments/08_operational_realism.py
	conda run -n CRIS python validation/governance_lab/experiments/09_governance_elasticity.py

run_certifications:
	conda run -n CRIS python validation/governance_lab/experiments/phase2_robustness_audit.py
	conda run -n CRIS python validation/governance_lab/experiments/institutional_impact_audit.py
	conda run -n CRIS python validation/governance_lab/experiments/10_stress_certification.py

clean:
	rm -rf outputs/credit_risk/*
	rm -rf outputs/macro/*
	rm -rf validation/governance_lab/artifacts/*
	rm -rf validation/governance_lab/reports/*
	
audit:
	conda run -n CRIS python import_audit.py
