#!/usr/bin/env python3
"""
LLM Interface Module
===================

Enhanced LLM interface for chemistry compound suggestions with better prompt 
engineering and error handling.

This module provides the EnhancedLLMInterface class that handles communication
with language models to generate compound suggestions based on available elements
and environmental conditions.
"""

import json
import logging
import re
import time
from typing import List, Optional

from ..data.data_models import CompoundSuggestion, EnvironmentalConditions

# Handle optional OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Create a dummy OpenAI class for type hints
    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass


class EnhancedLLMInterface:
    """Enhanced LLM interface with better prompt engineering and error handling."""
    
    def __init__(self, config):
        self.config = config
        self.client = self._setup_client()
        self.models = config.get('llm.models', [])
        self.current_model_index = 0
        
    def _setup_client(self) -> Optional[OpenAI]:
        """Setup OpenAI client for Hugging Face Router."""
        hf_token = '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'
        if not hf_token:
            logging.warning("No HF_TOKEN found. LLM suggestions will be disabled.")
            return None
        
        if not OPENAI_AVAILABLE:
            logging.warning("OpenAI library not available. LLM suggestions will be disabled.")
            return None
        
        try:
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
            )
            # Verify the client has the required attributes
            if not hasattr(client, 'chat') or not hasattr(client.chat, 'completions'):
                logging.error("OpenAI client missing required chat.completions attribute")
                return None
            logging.info("LLM client initialized successfully")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize LLM client: {e}")
            return None
    
    def query_for_compounds(self, element_names: List[str], 
                          environment: Optional[EnvironmentalConditions] = None) -> List[CompoundSuggestion]:
        """Query LLM for compound suggestions with enhanced prompting."""
        if not self.client or not hasattr(self.client, 'chat'):
            return []
        
        prompt = self._build_enhanced_prompt(element_names, environment)
        
        max_retries = self.config.get('llm.max_retries', 3)
        model = self._get_current_model()
        print(f"🤖 Querying {model} with elements: {element_names}")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info(f"Retry attempt {attempt + 1}/{max_retries}")
                    model = self._get_current_model()
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert chemist specializing in inorganic and planetary chemistry. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.get('llm.max_tokens', 800),
                    temperature=self.config.get('llm.temperature', 0.25),
                    timeout=self.config.get('llm.timeout', 15)
                )
                
                compounds = self._parse_response(response.choices[0].message.content)
                if compounds:
                    compound_formulas = [c.formula for c in compounds]
                    print(f"✅ LLM suggested {len(compounds)} compounds: {compound_formulas}")
                    logging.info(f"Successfully obtained {len(compounds)} compound suggestions")
                    return compounds
                
            except Exception as e:
                logging.warning(f"LLM query failed on attempt {attempt + 1}: {e}")
                self._rotate_model()
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logging.error("All LLM query attempts failed")
        return []
    
    def _build_enhanced_prompt(self, element_names: List[str], 
                             environment: Optional[EnvironmentalConditions]) -> str:
        """Build enhanced prompt with environmental context."""
        base_prompt = f"""
You are a chemistry expert specializing in compound formation and planetary chemistry.

Elements available: {', '.join(element_names)}

"""
        
        if environment:
            env_context = f"""
Environmental conditions:
- Temperature: {environment.temperature}°C
- Pressure: {environment.pressure} atm
- Atmosphere: {environment.atmosphere}
- pH: {environment.pH}
- Additional context: Consider stability under these specific conditions.

"""
            base_prompt += env_context
        
        task_prompt = """
Task: Generate 4-8 chemically realistic compounds using only the provided elements.

Requirements:
1. Each compound must use at least 2 of the provided elements
2. Prioritize well-known, experimentally confirmed compounds
3. Include both simple (binary) and complex (ternary/quaternary) compounds
4. Consider the environmental conditions for stability assessment
5. Avoid speculative or impossible compounds

Output format (JSON only, no additional text):
[
    {
        "formula": "chemical_formula",
        "name": "common_or_systematic_name",
        "type": "molecular|ionic|metallic|network",
        "stability": "high|medium|low"
    },
    ...
]
"""
        
        return base_prompt + task_prompt
    
    def _get_current_model(self) -> str:
        """Get current model with rotation support."""
        if not self.models:
            raise ValueError("No models configured")
        return self.models[self.current_model_index]
    
    def _rotate_model(self):
        """Rotate to next available model."""
        if len(self.models) > 1:
            self.current_model_index = (self.current_model_index + 1) % len(self.models)
    
    def _parse_response(self, response: str) -> List[CompoundSuggestion]:
        """Parse LLM response with enhanced error handling."""
        if not response:
            return []
        
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                lines = cleaned.split('\n')
                cleaned = '\n'.join(lines[1:-1])
            
            # Extract JSON array
            json_match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
            if not json_match:
                logging.warning("No JSON array found in LLM response")
                return []
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            compounds = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                try:
                    compound = CompoundSuggestion(
                        formula=item.get('formula', '').strip(),
                        name=item.get('name', '').strip(),
                        compound_type=item.get('type', 'unknown').strip().lower(),
                        stability=item.get('stability', 'unknown').strip().lower(),
                        source='llm'
                    )
                    
                    if compound.formula:  # Only add if formula is present
                        compounds.append(compound)
                        
                except Exception as e:
                    logging.warning(f"Error parsing compound item: {e}")
                    continue
            
            return compounds[:8]  # Limit to 8 compounds
            
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            return []