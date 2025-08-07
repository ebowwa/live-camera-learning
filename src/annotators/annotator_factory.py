"""
Factory for creating and configuring annotators.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from .base_annotator import BaseAnnotator
from .human_annotator import HumanAnnotator
from .gemini_annotator import GeminiAnnotator
from .multi_annotator import ConsensusAnnotator, FallbackAnnotator, WeightedAnnotator

logger = logging.getLogger(__name__)


class AnnotatorFactory:
    """
    Factory class for creating different types of annotators.
    
    Provides convenient methods to create and configure annotators
    based on configuration dictionaries or presets.
    """
    
    @staticmethod
    def create_human_annotator(config: Optional[Dict[str, Any]] = None) -> HumanAnnotator:
        """
        Create a human annotator.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Configured HumanAnnotator
            
        Example config:
            {
                'name': 'human_expert',
                'interactive_mode': True,
                'timeout_seconds': 300,
                'dataset_dir': 'captures/dataset'
            }
        """
        config = config or {}
        
        return HumanAnnotator(
            name=config.get('name', 'human'),
            knn_classifier=config.get('knn_classifier'),
            failed_dir=config.get('failed_dir', 'captures/failed'),
            dataset_dir=config.get('dataset_dir', 'captures/dataset'),
            interactive_mode=config.get('interactive_mode', False),
            timeout_seconds=config.get('timeout_seconds', 300.0)
        )
    
    @staticmethod
    def create_gemini_annotator(config: Optional[Dict[str, Any]] = None) -> GeminiAnnotator:
        """
        Create a Gemini AI annotator.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Configured GeminiAnnotator
            
        Example config:
            {
                'name': 'gemini_vision',
                'model_name': 'gemini-2.5-flash',
                'confidence_threshold': 0.7,
                'api_key': 'your-api-key'
            }
        """
        config = config or {}
        
        return GeminiAnnotator(
            name=config.get('name', 'gemini'),
            api_key=config.get('api_key'),
            model_name=config.get('model_name', 'gemini-2.5-flash'),
            confidence_threshold=config.get('confidence_threshold', 0.7),
            max_retries=config.get('max_retries', 3),
            timeout_seconds=config.get('timeout_seconds', 30.0)
        )
    
    @staticmethod
    def create_consensus_annotator(annotators: List[BaseAnnotator], 
                                 config: Optional[Dict[str, Any]] = None) -> ConsensusAnnotator:
        """
        Create a consensus annotator.
        
        Args:
            annotators: List of base annotators to combine
            config: Optional configuration dictionary
            
        Returns:
            Configured ConsensusAnnotator
        """
        config = config or {}
        
        return ConsensusAnnotator(
            annotators=annotators,
            name=config.get('name', 'consensus'),
            min_consensus_ratio=config.get('min_consensus_ratio', 0.5),
            confidence_weighting=config.get('confidence_weighting', True)
        )
    
    @staticmethod
    def create_fallback_annotator(annotators: List[BaseAnnotator], 
                                config: Optional[Dict[str, Any]] = None) -> FallbackAnnotator:
        """
        Create a fallback annotator.
        
        Args:
            annotators: List of annotators in order of preference
            config: Optional configuration dictionary
            
        Returns:
            Configured FallbackAnnotator
        """
        config = config or {}
        
        return FallbackAnnotator(
            annotators=annotators,
            name=config.get('name', 'fallback'),
            min_confidence=config.get('min_confidence', 0.5)
        )
    
    @staticmethod
    def create_weighted_annotator(annotators: List[BaseAnnotator], 
                                config: Optional[Dict[str, Any]] = None) -> WeightedAnnotator:
        """
        Create a weighted annotator.
        
        Args:
            annotators: List of annotators to combine
            config: Optional configuration dictionary
            
        Returns:
            Configured WeightedAnnotator
        """
        config = config or {}
        
        return WeightedAnnotator(
            annotators=annotators,
            weights=config.get('weights'),
            name=config.get('name', 'weighted'),
            reliability_weighting=config.get('reliability_weighting', True)
        )
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseAnnotator:
        """
        Create annotator from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured annotator
            
        Example configurations:
        
        Single annotators:
        {
            'type': 'human',
            'config': {'interactive_mode': True}
        }
        
        {
            'type': 'gemini',
            'config': {'model_name': 'gemini-2.5-flash'}
        }
        
        Multi-annotators:
        {
            'type': 'consensus',
            'annotators': [
                {'type': 'gemini', 'config': {}},
                {'type': 'human', 'config': {}}
            ],
            'config': {'min_consensus_ratio': 0.6}
        }
        
        {
            'type': 'fallback',
            'annotators': [
                {'type': 'gemini', 'config': {}},
                {'type': 'human', 'config': {'interactive_mode': True}}
            ]
        }
        """
        annotator_type = config.get('type')
        annotator_config = config.get('config', {})
        
        if annotator_type == 'human':
            return cls.create_human_annotator(annotator_config)
        elif annotator_type == 'gemini':
            return cls.create_gemini_annotator(annotator_config)
        elif annotator_type in ['consensus', 'fallback', 'weighted']:
            # Multi-annotators need sub-annotators
            sub_annotator_configs = config.get('annotators', [])
            sub_annotators = []
            
            for sub_config in sub_annotator_configs:
                sub_annotator = cls.create_from_config(sub_config)
                sub_annotators.append(sub_annotator)
            
            if annotator_type == 'consensus':
                return cls.create_consensus_annotator(sub_annotators, annotator_config)
            elif annotator_type == 'fallback':
                return cls.create_fallback_annotator(sub_annotators, annotator_config)
            elif annotator_type == 'weighted':
                return cls.create_weighted_annotator(sub_annotators, annotator_config)
        
        raise ValueError(f"Unknown annotator type: {annotator_type}")
    
    @classmethod
    def create_preset(cls, preset_name: str, **kwargs) -> BaseAnnotator:
        """
        Create annotator from preset configuration.
        
        Args:
            preset_name: Name of preset ('gemini_only', 'human_only', 'ai_first', 'consensus_ai_human', etc.)
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured annotator
        """
        presets = {
            'gemini_only': {
                'type': 'gemini',
                'config': {}
            },
            
            'human_only': {
                'type': 'human',
                'config': {'interactive_mode': True}
            },
            
            'ai_first': {
                'type': 'fallback',
                'annotators': [
                    {'type': 'gemini', 'config': {}},
                    {'type': 'human', 'config': {'interactive_mode': True}}
                ],
                'config': {'min_confidence': 0.7}
            },
            
            'human_first': {
                'type': 'fallback', 
                'annotators': [
                    {'type': 'human', 'config': {}},
                    {'type': 'gemini', 'config': {}}
                ]
            },
            
            'consensus_ai_human': {
                'type': 'consensus',
                'annotators': [
                    {'type': 'gemini', 'config': {}},
                    {'type': 'human', 'config': {}}
                ],
                'config': {'min_consensus_ratio': 0.6}
            },
            
            'weighted_ai_heavy': {
                'type': 'weighted',
                'annotators': [
                    {'type': 'gemini', 'config': {}},
                    {'type': 'human', 'config': {}}
                ],
                'config': {'weights': [0.8, 0.2]}
            },
            
            'weighted_human_heavy': {
                'type': 'weighted',
                'annotators': [
                    {'type': 'gemini', 'config': {}},
                    {'type': 'human', 'config': {}}
                ],
                'config': {'weights': [0.3, 0.7]}
            }
        }
        
        if preset_name not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
        
        preset_config = presets[preset_name].copy()
        
        # Apply any configuration overrides
        if kwargs:
            if 'config' in preset_config:
                preset_config['config'].update(kwargs)
            else:
                preset_config['config'] = kwargs
        
        return cls.create_from_config(preset_config)
    
    @staticmethod
    def get_available_presets() -> List[str]:
        """Get list of available preset names."""
        return [
            'gemini_only',
            'human_only', 
            'ai_first',
            'human_first',
            'consensus_ai_human',
            'weighted_ai_heavy',
            'weighted_human_heavy'
        ]
    
    @staticmethod
    def describe_preset(preset_name: str) -> str:
        """Get description of a preset."""
        descriptions = {
            'gemini_only': "Uses only Gemini AI for annotations",
            'human_only': "Uses only human annotations (interactive mode)",
            'ai_first': "Tries Gemini first, falls back to human if confidence < 0.7",
            'human_first': "Tries human first, falls back to Gemini",
            'consensus_ai_human': "Requires consensus between Gemini and human (60% agreement)",
            'weighted_ai_heavy': "Weighted combination favoring AI (80% AI, 20% human)",
            'weighted_human_heavy': "Weighted combination favoring human (30% AI, 70% human)"
        }
        return descriptions.get(preset_name, "Unknown preset")


def create_dual_annotator(ai_config: Optional[Dict[str, Any]] = None,
                         human_config: Optional[Dict[str, Any]] = None,
                         combination_strategy: str = 'fallback') -> BaseAnnotator:
    """
    Convenience function to create a dual AI+Human annotator system.
    
    Args:
        ai_config: Configuration for AI annotator (Gemini)
        human_config: Configuration for human annotator
        combination_strategy: How to combine ('fallback', 'consensus', 'weighted')
        
    Returns:
        Configured dual annotator
        
    Example:
        # AI-first with human fallback
        annotator = create_dual_annotator(
            ai_config={'confidence_threshold': 0.8},
            human_config={'interactive_mode': True},
            combination_strategy='fallback'
        )
        
        # Consensus between AI and human
        annotator = create_dual_annotator(
            combination_strategy='consensus'
        )
    """
    factory = AnnotatorFactory()
    
    # Create individual annotators
    ai_annotator = factory.create_gemini_annotator(ai_config)
    human_annotator = factory.create_human_annotator(human_config)
    
    annotators = [ai_annotator, human_annotator]
    
    # Create combined annotator based on strategy
    if combination_strategy == 'fallback':
        return factory.create_fallback_annotator(annotators)
    elif combination_strategy == 'consensus':
        return factory.create_consensus_annotator(annotators)
    elif combination_strategy == 'weighted':
        return factory.create_weighted_annotator(annotators)
    else:
        raise ValueError(f"Unknown combination strategy: {combination_strategy}")