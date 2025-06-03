# mindx/orchestration/model_selector.py
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from mindx.utils.config import Config
from mindx.utils.logging_config import get_logger
# Assuming ModelCapability and TaskType are defined in multimodel_agent
# and correctly imported if this were a separate file in the same package.
# For this combined output, they will be available from the MMA definition.
from .multimodel_agent import TaskType, ModelCapability # Relative import within package

logger = get_logger(__name__)

class ModelSelector:
    """
    Handles the selection of appropriate LLMs for different task types.

    This class encapsulates the logic for selecting models based on their
    statically defined capabilities, dynamically updated runtime statistics 
    (success rate, latency), cost, and task-specific contextual requirements.
    It aims to provide a flexible and extensible approach to model selection.
    """

    DEFAULT_SELECTION_WEIGHTS: Dict[str, float] = {
        "capability_match": 2.0,    # How well model's declared skills fit task_type
        "success_rate": 1.5,        # Historical success rate of the model
        "latency_factor": 0.5,      # Inverse of latency (faster is better)
        "cost_factor": 0.3,         # Inverse of cost (cheaper is better)
        "requirements_match": 3.0,  # Penalty/bonus for meeting specific task requirements
        "provider_preference": 0.2, # General preference for certain providers
        "availability": 1.0         # Ensure model is available
    }

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the ModelSelector.

        Args:
            config: Configuration object. Uses global Config singleton if None.
        """
        self.config = config or Config()
        # Load weights from config, falling back to class defaults
        self.selection_weights = self.config.get(
            "orchestration.model_selector.weights", 
            self.DEFAULT_SELECTION_WEIGHTS.copy() # Use a copy
        )
        self.min_availability_threshold = self.config.get(
            "orchestration.model_selector.min_availability_threshold", 0.1 # Don't select if less than 10% available
        )
        self.provider_preferences = self.config.get(
            "orchestration.model_selector.provider_preferences",
            {"gemini": 1.0, "openai": 0.9, "anthropic": 0.8, "ollama": 0.6, "mock": 0.1} # Example preferences
        )
        logger.info(f"ModelSelector initialized. Weights: {self.selection_weights}, Min Availability: {self.min_availability_threshold}")

    def select_models(self, selection_data: Dict[str, Any]) -> List[str]:
        """
        Selects the most appropriate model(s) based on the provided criteria.

        Args:
            selection_data: A dictionary containing:
                - 'model_capabilities': Dict[str, ModelCapability] - All available models and their capabilities.
                - 'task_type': TaskType - The type of task for which a model is needed.
                - 'context': Optional[Dict[str, Any]] - Task-specific context that might influence selection
                  (e.g., "context_length_needed", "preferred_provider", "budget_constraints").
                - 'num_models': int - The desired number of top models to return.
                - 'excluded_models': Optional[Set[str]] - A set of model_ids to exclude from selection.
                - 'debug_mode': bool - If true, enables verbose logging for selection process.

        Returns:
            A list of selected model_ids, sorted by score (highest first).
            Returns an empty list if no suitable models are found.
        """
        model_capabilities: Dict[str, ModelCapability] = selection_data.get("model_capabilities", {})
        task_type: Optional[TaskType] = selection_data.get("task_type")
        context: Dict[str, Any] = selection_data.get("context", {})
        num_models_to_select: int = selection_data.get("num_models", 1)
        excluded_models: Set[str] = selection_data.get("excluded_models", set())
        debug_mode: bool = selection_data.get("debug_mode", False)

        if not model_capabilities:
            logger.warning("ModelSelector: No model capabilities provided for selection.")
            return []
        if not task_type:
            logger.warning("ModelSelector: No task type provided for selection.")
            return []
        if not isinstance(task_type, TaskType): # pragma: no cover
            logger.error(f"ModelSelector: task_type must be an instance of TaskType enum, got {type(task_type)}")
            return []

        # Use a copy of configured weights, allow context to override them for this specific selection
        current_weights = self.selection_weights.copy()
        if "weight_adjustments" in context and isinstance(context["weight_adjustments"], dict): # pragma: no cover
            for weight_key, adjustment_factor in context["weight_adjustments"].items():
                if weight_key in current_weights and isinstance(adjustment_factor, (int, float)):
                    current_weights[weight_key] *= adjustment_factor # Apply as a multiplier
                    if debug_mode: logger.info(f"ModelSelector DEBUG: Adjusted weight for '{weight_key}' by factor {adjustment_factor:.2f} to {current_weights[weight_key]:.2f} based on task context.")
        
        if debug_mode: logger.info(f"ModelSelector DEBUG: Using effective weights: {current_weights}")

        candidate_scores: Dict[str, float] = {}
        for model_id, capability_obj in model_capabilities.items():
            if model_id in excluded_models:
                if debug_mode: logger.info(f"ModelSelector DEBUG: Skipping '{model_id}': explicitly excluded.")
                continue
            
            if capability_obj.availability < self.min_availability_threshold:
                if debug_mode: logger.info(f"ModelSelector DEBUG: Skipping '{model_id}': availability ({capability_obj.availability:.2f}) < threshold ({self.min_availability_threshold:.2f}).")
                continue

            score = self._calculate_model_score(capability_obj, task_type, current_weights, context, debug_mode)
            
            # Apply direct model score adjustments from context (e.g., user preference)
            if "model_score_adjustments" in context and isinstance(context["model_score_adjustments"], dict): # pragma: no cover
                adjustment_factor = context["model_score_adjustments"].get(model_id)
                if isinstance(adjustment_factor, (int, float)):
                    score *= adjustment_factor
                    if debug_mode: logger.info(f"ModelSelector DEBUG: Applied direct score adjustment factor {adjustment_factor:.2f} to '{model_id}', new score: {score:.3f}")

            if score > 0.001: # Only consider models with a meaningful positive score
                candidate_scores[model_id] = score
            elif debug_mode: # pragma: no cover
                logger.info(f"ModelSelector DEBUG: Skipping '{model_id}': score ({score:.3f}) too low.")

        if not candidate_scores:
            logger.warning(f"ModelSelector: No models scored positively for task type '{task_type.name}' with given context.")
            return []

        # Sort models by score (descending), then by a secondary tie-breaker (e.g., lower cost, then lower latency)
        def tie_breaker_key(model_id_score_tuple: Tuple[str, float]) -> Tuple[float, float, float]:
            model_id, score = model_id_score_tuple
            cap = model_capabilities[model_id]
            # Lower cost is better (negative makes it ascending for min-heap like behavior in sort)
            # Assuming cost per 1K total tokens for simplicity
            cost_per_k_total_tokens = cap.resource_usage.get("cost_per_kilo_total_tokens", 
                                      cap.resource_usage.get("cost_per_token", 0.0) * 1000) 
            # Lower latency is better
            avg_latency_seconds = cap.average_latency_ms / 1000.0
            return (-score, cost_per_k_total_tokens, avg_latency_seconds) # Sort by score desc, then cost asc, then latency asc

        sorted_models_with_scores = sorted(candidate_scores.items(), key=tie_breaker_key)
        
        selected_model_ids = [model_id for model_id, _ in sorted_models_with_scores[:num_models_to_select]]

        if debug_mode: # pragma: no cover
            logger.info(f"ModelSelector DEBUG: Scoreboard for task '{task_type.name}':")
            for mid_debug, score_debug in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True):
                logger.info(f"  - {mid_debug}: {score_debug:.3f}")
            logger.info(f"ModelSelector DEBUG: Top {len(selected_model_ids)} model(s) selected: {selected_model_ids}")

        return selected_model_ids

    def _calculate_model_score(self, 
                              capability: ModelCapability, 
                              task_type: TaskType, 
                              weights: Dict[str, float],
                              context: Dict[str, Any], 
                              debug_mode: bool) -> float:
        """Calculates a weighted score for a single model based on various factors."""
        
        model_id = capability.model_id
        score_components: Dict[str, float] = {}

        # 1. Base Capability Match Score
        score_components["capability_match"] = capability.get_capability_score(task_type)
        
        # 2. Success Rate
        score_components["success_rate"] = capability.success_rate
        
        # 3. Latency Factor (inverse, normalized: lower latency = higher factor)
        avg_latency_sec = capability.average_latency_ms / 1000.0
        # Add 0.1 to avoid division by zero and to slightly penalize very low (potentially unreliable) latencies
        score_components["latency_factor"] = 1.0 / (0.1 + avg_latency_sec) 
                                                 
        # 4. Cost Factor (inverse, normalized: lower cost = higher factor)
        # Using an estimated cost per 1K total tokens as a common metric.
        # If cost is 0 (free models), this factor should be high.
        cost_per_kilo_total_tokens = capability.resource_usage.get("cost_per_kilo_total_tokens",
                                        capability.resource_usage.get("cost_per_token", 0.0) * 1000) # Default if only per-token cost
        if cost_per_kilo_total_tokens <= 0.000001: # Effectively free or negligible
            score_components["cost_factor"] = 2.0 # High factor for free models
        else:
            score_components["cost_factor"] = 1.0 / (0.01 + cost_per_kilo_total_tokens) # Add 0.01 to avoid extreme values for very cheap models

        # 5. Provider Preference
        score_components["provider_preference"] = self.provider_preferences.get(capability.provider, 0.1) # Default low if not listed

        # 6. Requirements Match (penalty based)
        # Starts at 1.0 (full match), penalties reduce it.
        requirements_match_factor = 1.0
        task_requirements = context.get("task_requirements", {}) # Assume requirements are in context
        
        min_ctx_len_req = task_requirements.get("min_context_length")
        if isinstance(min_ctx_len_req, int) and capability.max_context_length < min_ctx_len_req:
            requirements_match_factor *= 0.1 # Heavy penalty
            if debug_mode: logger.debug(f"... {model_id} fails context length: {capability.max_context_length} < {min_ctx_len_req}")
        
        if task_requirements.get("supports_streaming") and not capability.supports_streaming:
            requirements_match_factor *= 0.5
            if debug_mode: logger.debug(f"... {model_id} fails streaming requirement.")
        
        if task_requirements.get("supports_function_calling") and not capability.supports_function_calling:
            requirements_match_factor *= 0.5
            if debug_mode: logger.debug(f"... {model_id} fails function calling requirement.")

        # Target model/provider requirements are strict (handled mostly by MMA before calling selector, or here as 0 factor)
        if task_requirements.get("target_model_id") and model_id != task_requirements.get("target_model_id"):
            requirements_match_factor = 0.0 
        if task_requirements.get("target_provider") and capability.provider != task_requirements.get("target_provider"):
            requirements_match_factor = 0.0
        
        score_components["requirements_match"] = requirements_match_factor

        # --- Calculate Final Weighted Score ---
        # Using a multiplicative approach where each weighted factor scales the score.
        # A base score could be set (e.g., 1.0) and then multiplied by each weighted component.
        # Or, sum weighted components. Multiplicative can be more selective.
        # Let's use weighted sum for more direct control via weights.
        final_score = 0.0
        weighted_contrib: Dict[str, float] = {}

        for factor_name, factor_value in score_components.items():
            weight = weights.get(factor_name, 0.0) # Default weight 0 if not specified
            contribution = factor_value * weight
            final_score += contribution
            weighted_contrib[factor_name] = contribution
            
        if debug_mode: # pragma: no cover
            logger.debug(f"ModelSelector DEBUG: Scoring for '{model_id}', Task: '{task_type.name}'")
            for factor, val in score_components.items(): logger.debug(f"  - Raw {factor}: {val:.3f}")
            for factor, contrib in weighted_contrib.items(): logger.debug(f"  - Weighted {factor} (weight {weights.get(factor,0):.2f}): {contrib:.3f}")
            logger.debug(f"  - Final Score for '{model_id}': {final_score:.3f}")
            
        return final_score
