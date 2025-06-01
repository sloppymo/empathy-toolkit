#!/usr/bin/env python3
"""
Multi-dimensional Empathy Scoring Framework

This module provides comprehensive empathy scoring capabilities using a multi-model
approach to evaluate different dimensions of empathy in text responses.
"""

import os
import json
import httpx
from typing import Dict, List, Tuple, Union, Optional
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmpathyDimension(Enum):
    """Enum representing different dimensions of empathy"""
    COGNITIVE = "cognitive"  # Understanding others' mental states
    EMOTIONAL = "emotional"  # Sharing/resonating with others' feelings
    BEHAVIORAL = "behavioral"  # Expressing empathy through actions/words
    CONTEXTUAL = "contextual"  # Awareness of situational factors
    CULTURAL = "cultural"  # Cultural appropriateness of empathy expression


@dataclass
class EmpathyScore:
    """Class to store and calculate empathy scores across dimensions"""
    cognitive: float = 0.0  # 0-10 scale
    emotional: float = 0.0  # 0-10 scale
    behavioral: float = 0.0  # 0-10 scale
    contextual: float = 0.0  # 0-10 scale
    cultural: float = 0.0  # 0-10 scale
    total: float = 0.0  # Weighted average
    
    def calculate_total(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted total score"""
        if weights is None:
            weights = {
                "cognitive": 0.25,
                "emotional": 0.25,
                "behavioral": 0.25,
                "contextual": 0.15,
                "cultural": 0.10
            }
            
        dimensions = [
            (self.cognitive, weights.get("cognitive", 0.25)),
            (self.emotional, weights.get("emotional", 0.25)),
            (self.behavioral, weights.get("behavioral", 0.25)),
            (self.contextual, weights.get("contextual", 0.15)),
            (self.cultural, weights.get("cultural", 0.10))
        ]
        
        self.total = sum(score * weight for score, weight in dimensions)
        return self.total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_pandas_series(self) -> pd.Series:
        """Convert to pandas Series"""
        return pd.Series(self.to_dict())


@dataclass
class EmpathyFeedback:
    """Class to store detailed feedback about empathy scores"""
    strengths: List[str]
    areas_for_improvement: List[str]
    suggestions: List[str]
    dimension_feedback: Dict[str, str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class EmpathyScoringSystem:
    """Comprehensive system for scoring empathy across multiple dimensions"""
    
    def __init__(
        self, 
        primary_model: str = "gpt-4o", 
        feedback_model: str = "gpt-3.5-turbo",
        api_key: str = None,
        use_cached_results: bool = True,
        cache_dir: str = ".empathy_cache"
    ):
        """
        Initialize the empathy scoring system
        
        Args:
            primary_model: Model to use for detailed scoring
            feedback_model: Model to use for feedback generation
            api_key: OpenAI API key (uses env var OPENAI_API_KEY if not provided)
            use_cached_results: Whether to cache and reuse results
            cache_dir: Directory to store cached results
        """
        self.primary_model = primary_model
        self.feedback_model = feedback_model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No API key provided. Set OPENAI_API_KEY environment variable.")
        
        # Cache configuration
        self.use_cached_results = use_cached_results
        self.cache_dir = cache_dir
        if use_cached_results and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Load rubrics and scoring criteria
        self.scoring_criteria = self._load_scoring_criteria()
            
    def _load_scoring_criteria(self) -> Dict:
        """Load scoring criteria from embedded definitions"""
        # In production, these would be loaded from files for easier updates
        return {
            "cognitive": {
                "description": "Understanding others' perspective and mental state",
                "indicators": [
                    "Accurately identifies thoughts and beliefs of others",
                    "Demonstrates perspective-taking",
                    "Recognizes underlying concerns beyond surface statements",
                    "Avoids mind-reading or projecting own assumptions"
                ],
                "examples": {
                    "high": "I understand you're worried about the potential side effects based on your past experiences.",
                    "medium": "You seem concerned about the treatment.",
                    "low": "The treatment is perfectly safe, you shouldn't worry."
                }
            },
            "emotional": {
                "description": "Resonating with and acknowledging others' feelings",
                "indicators": [
                    "Names or reflects specific emotions",
                    "Validates feelings as understandable",
                    "Shows emotional resonance in language",
                    "Responds proportionately to emotional intensity"
                ],
                "examples": {
                    "high": "That sounds incredibly frustrating and overwhelming. It makes sense you'd feel anxious.",
                    "medium": "That must be difficult to deal with.",
                    "low": "Let's focus on the facts rather than emotions."
                }
            },
            "behavioral": {
                "description": "Expressing empathy through communication",
                "indicators": [
                    "Uses supportive language and tone",
                    "Demonstrates active listening through reflection",
                    "Offers appropriate assistance or resources",
                    "Balances problem-solving with emotional support"
                ],
                "examples": {
                    "high": "I'm here to listen and support you. Would it help to talk more about your concerns or would you prefer we discuss options?",
                    "medium": "Here are some options that might help you.",
                    "low": "You should just try this solution I'm suggesting."
                }
            },
            "contextual": {
                "description": "Recognizing situational factors affecting the person",
                "indicators": [
                    "Acknowledges external circumstances influencing experience",
                    "Considers life context when responding",
                    "Recognizes practical constraints",
                    "Adapts response to situational appropriateness"
                ],
                "examples": {
                    "high": "Living far from medical facilities while managing chronic pain creates real barriers to consistent care.",
                    "medium": "Your location might make it harder to get treatment.",
                    "low": "Everyone faces challenges, you just need to figure it out."
                }
            },
            "cultural": {
                "description": "Cultural appropriateness and sensitivity",
                "indicators": [
                    "Respects cultural differences in expressing/receiving empathy",
                    "Avoids imposing cultural assumptions",
                    "Accommodates differing communication styles",
                    "Acknowledges cultural contexts when relevant"
                ],
                "examples": {
                    "high": "I understand in your community, these decisions often involve family consultation. Would you like to discuss how to approach this conversation with them?",
                    "medium": "Different cultures have different approaches to this issue.",
                    "low": "You should just be direct about it, that's always best."
                }
            }
        }
    
    def _generate_scoring_prompt(self, response: str, context: str) -> str:
        """Generate a detailed prompt for scoring empathy dimensions"""
        criteria_json = json.dumps(self.scoring_criteria, indent=2)
        
        prompt = f"""
You are an expert in evaluating empathetic responses with training in psychology, counseling, and communication.

TASK: Score the following response across five dimensions of empathy, on a scale of 0-10.

CONTEXT OF CONVERSATION:
{context}

RESPONSE TO EVALUATE:
{response}

SCORING CRITERIA:
{criteria_json}

INSTRUCTIONS:
1. Carefully analyze the response against each dimension's indicators
2. Assign a score from 0-10 for each dimension, where:
   - 0-2: Shows minimal or no evidence of this empathy dimension
   - 3-4: Shows limited evidence with major missed opportunities
   - 5-6: Shows moderate evidence with some missed opportunities
   - 7-8: Shows strong evidence with minor missed opportunities
   - 9-10: Shows exceptional evidence of this dimension

3. Provide your scoring as valid JSON matching this format exactly:
{{
  "cognitive": <score>,
  "emotional": <score>,
  "behavioral": <score>,
  "contextual": <score>, 
  "cultural": <score>,
  "reasoning": {{
    "cognitive": "brief justification",
    "emotional": "brief justification",
    "behavioral": "brief justification",
    "contextual": "brief justification",
    "cultural": "brief justification"
  }}
}}

Your evaluation should be objective, consistent, and based solely on the evidence in the response.
"""
        return prompt
    
    def _generate_feedback_prompt(self, response: str, context: str, scores: Dict) -> str:
        """Generate a prompt for detailed feedback based on scores"""
        return f"""
You are an empathy coach helping improve communication skills.

CONTEXT OF CONVERSATION:
{context}

RESPONSE THAT WAS EVALUATED:
{response}

EMPATHY SCORES GIVEN (0-10 scale):
{json.dumps(scores, indent=2)}

TASK: Provide constructive feedback on the response's empathetic qualities.
Focus on being specific, actionable, and balanced between strengths and areas for improvement.

Provide your feedback as valid JSON matching this format exactly:
{{
  "strengths": [
    "specific strength 1",
    "specific strength 2",
    "specific strength 3"
  ],
  "areas_for_improvement": [
    "specific area 1",
    "specific area 2"
  ],
  "suggestions": [
    "actionable suggestion 1",
    "actionable suggestion 2",
    "example alternative phrasing"
  ],
  "dimension_feedback": {{
    "cognitive": "specific feedback on cognitive empathy",
    "emotional": "specific feedback on emotional empathy",
    "behavioral": "specific feedback on behavioral empathy",
    "contextual": "specific feedback on contextual awareness",
    "cultural": "specific feedback on cultural sensitivity"
  }}
}}
"""
    
    async def _call_openai_api(self, prompt: str, model: str) -> Dict:
        """Call OpenAI API asynchronously"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0.1,  # Low temperature for consistent scoring
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=30.0)
                response.raise_for_status()
                result = response.json()
                return json.loads(result["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def _get_cache_key(self, response: str, context: str) -> str:
        """Generate a unique cache key for a response-context pair"""
        import hashlib
        content = f"{response}|{context}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def _check_cache(self, cache_key: str, cache_type: str) -> Optional[Dict]:
        """Check if result is cached"""
        if not self.use_cached_results:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_type}_{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache read error: {str(e)}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, cache_type: str, data: Dict) -> None:
        """Save result to cache"""
        if not self.use_cached_results:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_type}_{cache_key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")
    
    async def score_response(
        self, 
        response: str, 
        context: str = "", 
        include_feedback: bool = True
    ) -> Tuple[EmpathyScore, Optional[EmpathyFeedback]]:
        """
        Score a response across multiple empathy dimensions
        
        Args:
            response: The response text to evaluate
            context: Contextual information (e.g., the original message being responded to)
            include_feedback: Whether to generate detailed feedback
            
        Returns:
            Tuple of (EmpathyScore, EmpathyFeedback) objects
        """
        cache_key = self._get_cache_key(response, context)
        
        # Try to get scores from cache
        cached_scores = self._check_cache(cache_key, "scores")
        if cached_scores:
            logger.info(f"Using cached scoring results")
            scores_dict = cached_scores
        else:
            # Generate new scores
            scoring_prompt = self._generate_scoring_prompt(response, context)
            scores_result = await self._call_openai_api(scoring_prompt, self.primary_model)
            
            # Extract and validate scores
            scores_dict = {
                "cognitive": float(scores_result.get("cognitive", 0)),
                "emotional": float(scores_result.get("emotional", 0)),
                "behavioral": float(scores_result.get("behavioral", 0)),
                "contextual": float(scores_result.get("contextual", 0)),
                "cultural": float(scores_result.get("cultural", 0)),
                "reasoning": scores_result.get("reasoning", {})
            }
            
            # Cache the results
            self._save_to_cache(cache_key, "scores", scores_dict)
        
        # Create EmpathyScore object
        empathy_score = EmpathyScore(
            cognitive=scores_dict["cognitive"],
            emotional=scores_dict["emotional"],
            behavioral=scores_dict["behavioral"],
            contextual=scores_dict["contextual"],
            cultural=scores_dict["cultural"]
        )
        empathy_score.calculate_total()
        
        # Generate feedback if requested
        feedback = None
        if include_feedback:
            # Try to get feedback from cache
            cached_feedback = self._check_cache(cache_key, "feedback")
            if cached_feedback:
                logger.info(f"Using cached feedback results")
                feedback_dict = cached_feedback
            else:
                # Generate new feedback
                feedback_prompt = self._generate_feedback_prompt(
                    response, context, 
                    {k: v for k, v in scores_dict.items() if k != "reasoning"}
                )
                feedback_dict = await self._call_openai_api(feedback_prompt, self.feedback_model)
                
                # Cache the results
                self._save_to_cache(cache_key, "feedback", feedback_dict)
            
            # Create EmpathyFeedback object
            feedback = EmpathyFeedback(
                strengths=feedback_dict.get("strengths", []),
                areas_for_improvement=feedback_dict.get("areas_for_improvement", []),
                suggestions=feedback_dict.get("suggestions", []),
                dimension_feedback=feedback_dict.get("dimension_feedback", {})
            )
        
        return empathy_score, feedback
    
    def batch_score_responses(
        self,
        responses: List[str],
        contexts: List[str] = None,
        include_feedback: bool = False
    ) -> pd.DataFrame:
        """
        Score multiple responses and return results as a DataFrame
        
        Args:
            responses: List of response texts to evaluate
            contexts: List of context strings (must match length of responses)
            include_feedback: Whether to include detailed feedback
            
        Returns:
            DataFrame with scoring results
        """
        import asyncio
        
        if contexts is None:
            contexts = [""] * len(responses)
        
        if len(responses) != len(contexts):
            raise ValueError("responses and contexts must have the same length")
        
        # Use ThreadPoolExecutor to process multiple responses concurrently
        async def process_batch():
            tasks = []
            for response, context in zip(responses, contexts):
                tasks.append(self.score_response(response, context, include_feedback))
            return await asyncio.gather(*tasks)
        
        # Run the async batch processing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(process_batch())
        
        # Prepare DataFrame
        df_data = []
        for i, (score, feedback) in enumerate(results):
            row = {
                "response_id": i,
                "response_text": responses[i][:100] + "..." if len(responses[i]) > 100 else responses[i],
                **score.to_dict()
            }
            
            if include_feedback and feedback:
                # Add selected feedback items
                row["strengths"] = "; ".join(feedback.strengths[:2]) if feedback.strengths else ""
                row["areas_for_improvement"] = "; ".join(feedback.areas_for_improvement[:2]) if feedback.areas_for_improvement else ""
                row["suggestions"] = "; ".join(feedback.suggestions[:2]) if feedback.suggestions else ""
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def visualize_scores(self, scores: Union[EmpathyScore, List[EmpathyScore], pd.DataFrame]) -> None:
        """
        Visualize empathy scores using a radar chart
        
        Args:
            scores: EmpathyScore object, list of EmpathyScore objects, or DataFrame
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
            import seaborn as sns
            from matplotlib.path import Path
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D
        except ImportError:
            logger.error("Visualization requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
            return
        
        # Convert input to DataFrame
        if isinstance(scores, EmpathyScore):
            df = pd.DataFrame([scores.to_dict()])
        elif isinstance(scores, list) and all(isinstance(s, EmpathyScore) for s in scores):
            df = pd.DataFrame([s.to_dict() for s in scores])
        elif isinstance(scores, pd.DataFrame):
            df = scores
        else:
            raise ValueError("scores must be an EmpathyScore, List[EmpathyScore], or DataFrame")
        
        # Extract dimensions, excluding 'total'
        dimensions = [col for col in df.columns if col in ['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural']]
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set radar chart parameters
        angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Set aesthetics
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], dimensions)
        
        # Set y limits
        ax.set_ylim(0, 10)
        ax.yaxis.set_major_locator(ticker.FixedLocator(range(0, 11, 2)))
        
        # Plot each response
        for i, row in df.iterrows():
            values = [row[dim] for dim in dimensions]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Response {i+1}")
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        if len(df) > 1:
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Empathy Dimensions Radar Chart", size=15, color='gray', y=1.1)
        plt.tight_layout()
        plt.show()


class BatchEmpathyAnalyzer:
    """Class for analyzing empathy patterns across multiple responses"""
    
    def __init__(self, scorer: EmpathyScoringSystem):
        """
        Initialize with an empathy scoring system
        
        Args:
            scorer: Instance of EmpathyScoringSystem
        """
        self.scorer = scorer
    
    async def analyze_dataset(
        self, 
        dataset_path: str, 
        response_column: str, 
        context_column: str = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Analyze empathy in a dataset
        
        Args:
            dataset_path: Path to CSV or JSONL file
            response_column: Column name containing responses to evaluate
            context_column: Column name with context (optional)
            output_path: Path to save results (optional)
            
        Returns:
            DataFrame with analysis results
        """
        # Load dataset
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
        
        # Score all responses
        results_df = self.scorer.batch_score_responses(responses, contexts)
        
        # Merge with original dataset
        result = pd.concat([df, results_df.drop(['response_id', 'response_text'], axis=1)], axis=1)
        
        # Save if requested
        if output_path:
            if output_path.endswith('.csv'):
                result.to_csv(output_path, index=False)
            elif output_path.endswith('.jsonl'):
                with jsonlines.open(output_path, mode='w') as writer:
                    for record in result.to_dict('records'):
                        writer.write(record)
            else:
                result.to_csv(output_path, index=False)
                
        return result
    
    def generate_insights(self, analysis_results: pd.DataFrame) -> Dict:
        """
        Generate insights from analysis results
        
        Args:
            analysis_results: DataFrame with analysis results
            
        Returns:
            Dictionary of insights
        """
        # Calculate statistics
        stats = {
            "mean_scores": {
                dim: analysis_results[dim].mean() 
                for dim in ['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural', 'total']
            },
            "median_scores": {
                dim: analysis_results[dim].median() 
                for dim in ['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural', 'total']
            },
            "strongest_dimension": analysis_results[['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural']].mean().idxmax(),
            "weakest_dimension": analysis_results[['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural']].mean().idxmin(),
            "high_empathy_count": len(analysis_results[analysis_results['total'] >= 7]),
            "medium_empathy_count": len(analysis_results[(analysis_results['total'] >= 4) & (analysis_results['total'] < 7)]),
            "low_empathy_count": len(analysis_results[analysis_results['total'] < 4]),
            "dimension_correlations": analysis_results[['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural']].corr().to_dict()
        }
        
        # Calculate balanced score distribution
        stats["balance_score"] = np.mean([
            np.std(row[['cognitive', 'emotional', 'behavioral', 'contextual', 'cultural']]) 
            for _, row in analysis_results.iterrows()
        ])
        
        return stats


# Simple example usage
async def example_usage():
    # Initialize the scoring system
    scorer = EmpathyScoringSystem()
    
    # Example response to evaluate
    context = "Patient: I've been feeling overwhelmed with my new diagnosis. I don't know how I'll manage everything."
    response = "I understand this is a difficult time for you. A new diagnosis can feel overwhelming, and it's completely normal to worry about how you'll cope. Many patients feel similarly when facing new health challenges. What specific aspects are you most concerned about? We can break this down into manageable steps and explore resources that might help you."
    
    # Score the response
    score, feedback = await scorer.score_response(response, context)
    
    print(f"Empathy Scores (0-10 scale):")
    print(f"Cognitive: {score.cognitive:.1f}")
    print(f"Emotional: {score.emotional:.1f}")
    print(f"Behavioral: {score.behavioral:.1f}")
    print(f"Contextual: {score.contextual:.1f}")
    print(f"Cultural: {score.cultural:.1f}")
    print(f"Total Score: {score.total:.1f}")
    
    if feedback:
        print("\nStrengths:")
        for strength in feedback.strengths:
            print(f"- {strength}")
            
        print("\nAreas for Improvement:")
        for area in feedback.areas_for_improvement:
            print(f"- {area}")
            
        print("\nSuggestions:")
        for suggestion in feedback.suggestions:
            print(f"- {suggestion}")
    
    # Visualize results
    scorer.visualize_scores(score)


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
