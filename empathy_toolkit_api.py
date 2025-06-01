#!/usr/bin/env python3
"""
Empathy Toolkit Unified API

This module provides a unified interface for all empathy-related functionality,
including the empathy scoring framework, model training, and data processing.
"""

import os
import asyncio
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

# Import the empathy scoring system
from analysis.empathy_scorer import (
    EmpathyScoringSystem, 
    EmpathyScore, 
    EmpathyFeedback, 
    BatchEmpathyAnalyzer
)

class EmpathyToolkit:
    """Unified API for the Empathy Toolkit"""
    
    def __init__(
        self, 
        use_gpu: bool = False, 
        models: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        use_cached_results: bool = True,
        cache_dir: str = ".empathy_cache"
    ):
        """
        Initialize the Empathy Toolkit with specified options.
        
        Args:
            use_gpu: Whether to use GPU for model training and inference
            models: Dictionary mapping model types to specific models
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            use_cached_results: Whether to cache API results
            cache_dir: Directory for caching results
        """
        self.use_gpu = use_gpu
        self.models = models or {"primary": "gpt-4o", "feedback": "gpt-3.5-turbo"}
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Initialize the empathy scoring system
        self.empathy_scorer = EmpathyScoringSystem(
            primary_model=self.models["primary"],
            feedback_model=self.models["feedback"],
            api_key=self.api_key,
            use_cached_results=use_cached_results,
            cache_dir=cache_dir
        )
            
        # Initialize dimension descriptions
        self.dimension_descriptions = {
            "Cognitive": "The ability to understand another person's perspective, thoughts, and feelings. It involves recognizing and comprehending the mental state of others.",
            "Emotional": "The capacity to share and resonate with another person's emotions. This involves feeling what another person is feeling, creating emotional connection.",
            "Behavioral": "The observable actions and responses that demonstrate empathy, including supportive communication, active listening, and appropriate responses.",
            "Contextual": "Understanding and responding to the specific situation or context in which the interaction occurs, acknowledging unique circumstances.",
            "Cultural": "Awareness and respect for cultural differences in expressing and experiencing emotions, recognizing diverse cultural norms and values.",
            "Total": "An overall assessment of empathy across all dimensions, representing the comprehensive empathic quality of the response."
        }
    
    async def score_empathy(
        self, 
        response: str, 
        context: str = "",
        include_feedback: bool = True
    ) -> Tuple[EmpathyScore, Optional[EmpathyFeedback]]:
        """
        Score empathy in a response and get detailed feedback.
        
        Args:
            response: The response text to evaluate
            context: The context/prompt that preceded the response
            include_feedback: Whether to include detailed feedback
            
        Returns:
            Tuple containing (EmpathyScore, EmpathyFeedback)
        """
        return await self.empathy_scorer.score_response(response, context, include_feedback)
    
    async def batch_score_empathy(
        self, 
        items: List[Dict[str, Any]],
        include_feedback: bool = False,
        max_concurrency: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Score empathy for multiple items in batch.
        
        Args:
            items (list): List of dictionaries, each containing 'id', 'response', and 'context' keys
            include_feedback (bool, optional): Whether to include detailed feedback. Defaults to False.
            max_concurrency (int, optional): Maximum number of concurrent API calls. Defaults to 5.
            
        Returns:
            list: List of dictionaries with score results added
        """
        results = []
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_item(item):
            """Process a single item with semaphore-controlled concurrency"""
            async with semaphore:
                try:
                    score, feedback = await self.empathy_scorer.score_response(
                        response=item['response'],
                        context=item.get('context', ''),
                        include_feedback=include_feedback
                    )
                    
                    result = item.copy()
                    result['score'] = score
                    if include_feedback:
                        result['feedback'] = feedback
                    
                    return result
                except Exception as e:
                    # Return the item with an error flag rather than failing the whole batch
                    result = item.copy()
                    result['error'] = str(e)
                    return result
        
        # Create tasks for all items
        tasks = [process_item(item) for item in items]
        
        # Process all items concurrently with controlled concurrency
        completed_results = await asyncio.gather(*tasks)
        
        # Filter out any None results (though there shouldn't be any with our error handling)
        results = [r for r in completed_results if r is not None]
            
        return results
    
    def train_empathy_model(
        self, 
        data_file: str, 
        output_dir: str, 
        base_model: str = "distilgpt2",
        epochs: int = 3,
        batch_size: int = 2,
        cpu_optimize: bool = False,
        use_mixed_precision: bool = True
    ) -> Dict[str, Any]:
        """
        Train a custom empathy model using the provided dataset.
        
        Args:
            data_file: Path to training data file
            output_dir: Directory to save the trained model
            base_model: Base model to fine-tune
            epochs: Number of training epochs
            batch_size: Training batch size
            cpu_optimize: Whether to optimize for CPU (if GPU not available)
            use_mixed_precision: Whether to use mixed precision training
            
        Returns:
            Dictionary with training results and metrics
        """
        from model_training.basic_empathy_trainer import train_empathy_model
        
        # Override CPU optimization if GPU is specifically requested
        if self.use_gpu:
            cpu_optimize = False
            
        # Call the training function
        return train_empathy_model(
            data_file=data_file,
            output_dir=output_dir,
            base_model=base_model,
            epochs=epochs,
            batch_size=batch_size,
            cpu_optimize=cpu_optimize,
            force_gpu=self.use_gpu,
            use_mixed_precision=use_mixed_precision
        )
    
    def get_dimension_descriptions(self) -> Dict[str, str]:
        """Get descriptions of the empathy dimensions
        
        Returns:
            Dictionary mapping dimension names to their descriptions
        """
        return self.dimension_descriptions
    
    async def enhance_dataset_with_scores(
        self,
        dataset_path: str,
        output_path: str,
        response_column: str,
        context_column: Optional[str] = None,
        include_feedback: bool = False,
        max_concurrency: int = 5
    ) -> bool:
        """Enhance a dataset with empathy scores
        
        Args:
            dataset_path: Path to the dataset file (CSV or JSONL)
            output_path: Path to save the enhanced dataset
            response_column: Column containing the responses to score
            context_column: Optional column containing context for the responses
            include_feedback: Whether to include detailed feedback
            max_concurrency: Maximum number of concurrent API calls
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load dataset directly (avoid nested async calls)
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.jsonl'):
                import jsonlines
                records = []
                with jsonlines.open(dataset_path) as reader:
                    for obj in reader:
                        records.append(obj)
                df = pd.DataFrame(records)
            else:
                raise ValueError("Dataset must be CSV or JSONL")
            
            # Extract responses and contexts
            responses = df[response_column].tolist()
            contexts = df[context_column].tolist() if context_column else [""] * len(responses)
            
            # Prepare batch items
            items = [
                {"id": i, "response": resp, "context": ctx} 
                for i, (resp, ctx) in enumerate(zip(responses, contexts))
            ]
            
            # Process in batch directly (avoiding nested async)
            batch_results = await self.batch_score_empathy(
                items=items,
                include_feedback=include_feedback,
                max_concurrency=max_concurrency
            )
            
            # Extract scores and create columns
            empathy_columns = {
                f"empathy_{dim}": [] for dim in ["cognitive", "emotional", "behavioral", "contextual", "cultural", "total"]
            }
            
            for result in batch_results:
                if "score" in result:
                    # Convert EmpathyScore object to dictionary if needed
                    score_dict = result["score"].to_dict() if hasattr(result["score"], "to_dict") else result["score"]
                    
                    for dim in empathy_columns.keys():
                        dim_name = dim.replace("empathy_", "")
                        empathy_columns[dim].append(score_dict.get(dim_name, 0.0))
                else:
                    # Handle missing scores
                    for dim in empathy_columns.keys():
                        empathy_columns[dim].append(0.0)
            
            # Add columns to dataframe
            for col, values in empathy_columns.items():
                df[col] = values
            
            # Save enhanced dataset
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.jsonl'):
                with jsonlines.open(output_path, mode='w') as writer:
                    for record in df.to_dict('records'):
                        writer.write(record)
            else:
                df.to_csv(output_path, index=False)
            
            return True
        except Exception as e:
            print(f"Error enhancing dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def get_empathy_dimensions(self) -> Dict[str, str]:
        """
        Get descriptions of all empathy dimensions used in scoring.
        
        Returns:
            Dictionary mapping dimension names to descriptions
        """
        return {
            "cognitive": "Understanding others' perspective and mental state",
            "emotional": "Resonating with and acknowledging others' feelings",
            "behavioral": "Expressing empathy through communication",
            "contextual": "Recognizing situational factors affecting the person",
            "cultural": "Cultural appropriateness and sensitivity"
        }


# Example usage
async def example_usage():
    # Initialize toolkit
    toolkit = EmpathyToolkit()
    
    # Score a single response
    response = "I understand how difficult this must be for you."
    context = "I just lost my job and I'm feeling overwhelmed."
    
    score, feedback = await toolkit.score_empathy(response, context)
    print(f"Total empathy score: {score.total}/10")
    
    # Dimensions
    for dimension, value in score.to_dict().items():
        if dimension != "total":
            print(f"- {dimension.capitalize()}: {value}/10")
    
    # Feedback
    if feedback:
        print("\nStrengths:")
        for strength in feedback.strengths:
            print(f"+ {strength}")
            
        print("\nAreas for Improvement:")
        for area in feedback.areas_for_improvement:
            print(f"- {area}")


if __name__ == "__main__":
    # Run example when script is executed directly
    asyncio.run(example_usage())
