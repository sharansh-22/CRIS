"""
Governance Lab Engine — Institutional Experimentation Framework for CRIS.
Provides a controlled, walk-forward environment for policy-sensitivity analysis.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger('CRIS.governance_lab')

class GovernanceLabEngine:
    def __init__(self, data_path: Path, macro_path: Path, config_path: Path):
        self.data_path = data_path
        self.macro_path = macro_path
        self.config_path = config_path
        
        # Load Baseline State
        self.full_df = pd.read_parquet(data_path)
        self.full_df['issue_d'] = pd.to_datetime(self.full_df['issue_d'])
        self.full_df['issue_month'] = self.full_df['issue_d'].dt.strftime('%Y-%m-01')
        
        self.macro_df = pd.read_csv(macro_path)
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)['overlay_config']
            
        # Ensure data alignment
        self.merged_df = self.full_df.merge(self.macro_df, on='issue_month', how='left')
        self.merged_df = self.merged_df.dropna(subset=['macro_stress_score'])
        
        # Calculate borrower-centric PDs if missing
        if 'pd_borrower' not in self.merged_df.columns:
            import joblib
            model_path = Path(__file__).resolve().parent.parent.parent.parent / "systems/credit_risk/models/saved_models/lightgbm.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            logger.info("Calculating borrower PDs using saved model...")
            model = joblib.load(model_path)
            model_features = model.feature_name_
            
            X = self.merged_df.copy()
            X.columns = [c.replace(' ', '_') for c in X.columns]
            X = X[model_features]
            self.merged_df['pd_borrower'] = model.predict_proba(X)[:, 1]
            logger.info("PD calculation complete.")

    def run_baseline_credit_simulation(self, approval_threshold: float = 0.20) -> pd.DataFrame:
        df = self.merged_df.copy()
        df['pd_macro'] = df['pd_borrower']
        df['gov_state'] = "NORMAL"
        df['approved'] = (df['pd_macro'] < approval_threshold).astype(int)
        return df

    def run_cris_v1_simulation(self, beta: float = 0.4) -> pd.DataFrame:
        df = self.merged_df.copy()
        def apply_conditioning(row):
            score = row['macro_stress_score']
            pd_b = np.clip(row['pd_borrower'], 1e-6, 1 - 1e-6); logit_b = np.log(pd_b / (1 - pd_b))
            shift = min(beta * max(0, score - 0.25), 0.35)
            return 1 / (1 + np.exp(-(logit_b + shift)))
        df['pd_macro'] = df.apply(apply_conditioning, axis=1)
        df['gov_state'] = df['macro_stress_score'].apply(lambda s: "DEFENSIVE" if s > 0.45 else ("CAUTIOUS" if s > 0.20 else "NORMAL"))
        df['approved'] = (df['pd_macro'] < 0.20).astype(int)
        return df

    def run_unified_modular_simulation(self, 
                                       source_betas: Dict[str, float],
                                       velocity_betas: Dict[str, float],
                                       recovery_velocities: Dict[str, float],
                                       hysteresis_params: Dict[str, float]) -> pd.DataFrame:
        df = self.merged_df.copy()
        months_df = self.macro_df.sort_values('issue_month').copy()
        source_map = {'liquidity': 'liquidity_disruption', 'structural': 'structural_fragility', 'macro': 'uncertainty_pressure', 'volatility': 'trajectory_fragility'}
        for source, col in source_map.items():
            if col in months_df.columns:
                months_df[f'{source}_vel'] = months_df[col].diff().fillna(0)

        months = sorted(df['issue_month'].unique()); monthly_states = {}; current_state = "NORMAL"
        for month in months:
            m_data = months_df[months_df['issue_month'] == month]
            if m_data.empty: continue
            score = m_data['macro_stress_score'].iloc[0]
            if current_state == "NORMAL":
                if score > hysteresis_params.get('entry', 0.45): current_state = "DEFENSIVE"
                elif score > 0.20: current_state = "CAUTIOUS"
            elif current_state == "CAUTIOUS":
                if score > hysteresis_params.get('entry', 0.45): current_state = "DEFENSIVE"
                elif score < hysteresis_params.get('exit', 0.15): current_state = "NORMAL"
            elif current_state == "DEFENSIVE":
                if score < hysteresis_params.get('exit_defensive', 0.35): current_state = "CAUTIOUS"
            monthly_states[month] = current_state
        df['gov_state'] = df['issue_month'].map(monthly_states)

        def apply_unified(row):
            pd_b = np.clip(row['pd_borrower'], 1e-6, 1 - 1e-6); logit_b = np.log(pd_b / (1 - pd_b))
            total_shift = 0.0; stab = row.get('stabilization_strength', 0.5)
            for source, col in source_map.items():
                if col not in row: continue
                val = row[col]; vel = row.get(f'{source}_vel', 0)
                eff_beta = (source_betas.get(source, 0.4) + (velocity_betas.get(source, 0.0) * max(0, vel))) * (1.0 - 0.4 * np.clip(stab * recovery_velocities.get(source, 1.0), 0, 1))
                total_shift += eff_beta * max(0, val - 0.20)
            return 1 / (1 + np.exp(-(logit_b + min(total_shift, 0.50))))

        vel_cols = [f'{s}_vel' for s in source_map.keys() if f'{s}_vel' in months_df.columns]
        df = df.merge(months_df[['issue_month'] + vel_cols], on='issue_month', how='left')
        df['pd_macro'] = df.apply(apply_unified, axis=1)
        df['approved'] = (df['pd_macro'] < 0.20).astype(int)
        return df

    def run_elastic_governance_simulation(self, 
                                          source_betas: Dict[str, float],
                                          velocity_betas: Dict[str, float],
                                          recovery_velocities: Dict[str, float],
                                          elasticity_k: float = 15.0,
                                          dampening_factor: float = 0.3,
                                          adversarial_noise: float = 0.0,
                                          **kwargs) -> pd.DataFrame:
        df = self.merged_df.copy()
        months_df = self.macro_df.sort_values('issue_month').copy()
        source_map = {'liquidity': 'liquidity_disruption', 'structural': 'structural_fragility', 'macro': 'uncertainty_pressure', 'volatility': 'trajectory_fragility'}
        if adversarial_noise > 0:
            for col in source_map.values():
                if col in months_df.columns: months_df[col] = np.clip(months_df[col] + np.random.normal(0, adversarial_noise, len(months_df)), 0, 1)
        for source, col in source_map.items():
            if col in months_df.columns: months_df[f'{source}_vel'] = months_df[col].diff().fillna(0)
        def elastic_response(val, threshold=0.20):
            gate = 1 / (1 + np.exp(-elasticity_k * (val - threshold)))
            return gate * (val - threshold)
        months = sorted(months_df['issue_month'].unique()); monthly_shifts = {}; last_shift = 0.0
        for m in months:
            m_data = months_df[months_df['issue_month'] == m].iloc[0]
            total_raw_shift = 0.0; stab = m_data.get('stabilization_strength', 0.5)
            for source, col in source_map.items():
                if col not in m_data: continue
                val = m_data[col]; vel = m_data.get(f'{source}_vel', 0)
                eff_beta = (source_betas.get(source, 0.4) + (velocity_betas.get(source, 0.0) * max(0, vel))) * (1.0 - 0.4 * np.clip(stab * recovery_velocities.get(source, 1.0), 0, 1))
                total_raw_shift += eff_beta * elastic_response(val)
            smooth_shift = (1.0 - dampening_factor) * total_raw_shift + dampening_factor * last_shift
            monthly_shifts[m] = smooth_shift; last_shift = smooth_shift
        df['gov_shift'] = df['issue_month'].map(monthly_shifts)
        df['gov_state'] = df['gov_shift'].apply(lambda s: "DEFENSIVE" if s > 0.15 else ("CAUTIOUS" if s > 0.05 else "NORMAL"))
        def apply_elastic(row):
            pd_b = np.clip(row['pd_borrower'], 1e-6, 1 - 1e-6); logit_b = np.log(pd_b / (1 - pd_b))
            shift = min(row['gov_shift'], 0.50)
            return 1 / (1 + np.exp(-(logit_b + shift)))
        df['pd_macro'] = df.apply(apply_elastic, axis=1); df['approved'] = (df['pd_macro'] < 0.20).astype(int)
        return df

    def run_stress_certification(self, 
                                 source_betas: Dict[str, float],
                                 velocity_betas: Dict[str, float],
                                 recovery_velocities: Dict[str, float],
                                 scenario_type: str = "BASE",
                                 **kwargs) -> pd.DataFrame:
        """
        IVSC: Institutional Validation & Stress Certification.
        """
        # Create a deep copy of the current state
        macro_copy = self.macro_df.copy()
        
        if scenario_type == "CONTAGION_CASCADE":
            macro_copy.loc[macro_copy['issue_month'] >= '2008-01-01', 'liquidity_disruption'] *= 1.5
            macro_copy.loc[macro_copy['issue_month'] >= '2008-06-01', 'structural_fragility'] *= 1.5
        elif scenario_type == "FALSE_STABILIZATION":
            macro_copy['stabilization_strength'] *= 0.2
        elif scenario_type == "ADVERSARIAL_NOISE":
            for col in ['liquidity_disruption', 'structural_fragility', 'uncertainty_pressure', 'trajectory_fragility']:
                macro_copy[col] = np.clip(macro_copy[col] + np.random.normal(0, 0.15, len(macro_copy)), 0, 1)
        
        # Use a temporary engine to run the simulation with modified macro data
        # We must ensure pd_borrower is available
        df_stress = self.merged_df.drop(columns=[c for c in macro_copy.columns if c != 'issue_month']).merge(macro_copy, on='issue_month', how='left')
        
        # Create a temporary instance to use its methods
        temp_engine = GovernanceLabEngine(self.data_path, self.macro_path, self.config_path)
        temp_engine.merged_df = df_stress
        temp_engine.macro_df = macro_copy
        
        return temp_engine.run_elastic_governance_simulation(source_betas, velocity_betas, recovery_velocities)

    def calculate_experiment_metrics(self, df: pd.DataFrame, default_penalty: float = 10.0) -> Dict[str, Any]:
        y_true = df['target']; approved = df['approved']; total_approved = approved.sum()
        interest_gain = (df[(approved == 1) & (y_true == 0)].shape[0]) * 0.10
        default_loss = (df[(approved == 1) & (y_true == 1)].shape[0]) * (0.10 * default_penalty)
        net_utility = interest_gain - default_loss
        gtv = df.groupby('issue_month')['gov_shift'].first().diff().abs().mean() if 'gov_shift' in df.columns else 0.05
        return {
            "approval_rate": approved.mean() if len(df) > 0 else 0,
            "default_rate": y_true[approved == 1].mean() if total_approved > 0 else 0,
            "net_utility": net_utility,
            "gtv": gtv,
            "sample_size": len(df),
            "false_negatives_count": int(((approved == 1) & (y_true == 1)).sum()),
            "opportunity_loss_count": int(((approved == 0) & (y_true == 0)).sum()),
            "capital_efficiency": float(interest_gain / max(default_loss, 1.0))
        }

    def segment_by_regime(self, df: pd.DataFrame, regime_name: str) -> pd.DataFrame:
        if regime_name == "FAST_LIQUIDITY": return df[(df['issue_month'] >= '2007-06-01') & (df['issue_month'] <= '2009-06-01')]
        elif regime_name == "SLOW_STRUCTURAL": return df[(df['issue_month'] >= '2014-01-01') & (df['issue_month'] <= '2016-12-01')]
        elif regime_name == "INFLATIONARY_STRESS": return df[(df['issue_month'] >= '2017-01-01') & (df['issue_month'] <= '2018-12-01')]
        elif regime_name == "POLICY_DISTORTED": return df[(df['issue_month'] >= '2010-01-01') & (df['issue_month'] <= '2013-12-01')]
        elif regime_name == "VOL_WITHOUT_FRAGILITY": return df[(df['trajectory_fragility'] > 0.40) & (df['macro_stress_score'] < 0.25)]
        elif regime_name == "EXOGENOUS_SHOCK": return df[df['issue_month'].isin(['2011-08-01', '2011-09-01', '2011-10-01', '2016-06-01', '2016-07-01'])]
        else: return df
