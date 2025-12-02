"""Text and context understanding service."""

import logging
import uuid
from typing import Dict, Any, List, Optional

# Try to import numpy (optional for basic functionality)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from src.config import settings

logger = logging.getLogger(__name__)

# Try to import transformers for GPT/BERT
# Note: we intentionally catch broad exceptions here because some environments
# have partial installs (e.g., missing SciPy/libgfortran) that cause runtime
# import errors instead of a clean ImportError. In those cases we degrade
# gracefully to heuristic-based analysis instead of failing hard.
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers stack not fully available. "
        "GPT/BERT comment quality will use fallback heuristics."
    )


class TextUnderstandingService:
    """Service for text and context understanding."""
    
    def __init__(self):
        self.service_id = str(uuid.uuid4())
        
        # Initialize GPT/BERT models for comment quality analysis
        self.sentiment_analyzer = None
        self.device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._initialize_comment_quality_models()
        
        logger.info(f"Initialized TextUnderstandingService: {self.service_id}")
    
    def _initialize_comment_quality_models(self):
        """Initialize GPT/BERT models for comment quality analysis."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Comment quality will use fallback.")
            return
        
        try:
            # Use a lightweight BERT-based sentiment analysis model
            # This model is good for sentiment analysis and can be used for comment quality
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info(f"Initialized BERT sentiment analyzer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}, trying fallback model: {e}")
                # Fallback to a simpler model
                try:
                    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        device=0 if self.device == "cuda" else -1,
                        return_all_scores=True
                    )
                    logger.info(f"Initialized fallback sentiment analyzer: {model_name}")
                except Exception as e2:
                    logger.error(f"Failed to load fallback sentiment model: {e2}")
                    self.sentiment_analyzer = None
        except Exception as e:
            logger.error(f"Error initializing comment quality models: {e}")
            self.sentiment_analyzer = None
    
    async def analyze_text_content(
        self,
        caption: Optional[str] = None,
        description: Optional[str] = None,
        hashtags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze text content for virality indicators.
        
        Args:
            caption: Content caption
            description: Content description
            hashtags: List of hashtags
            metadata: Additional metadata
        
        Returns:
            Dictionary containing:
            - trend_proximity: Trend + semantic proximity scoring
            - emotion: Emotion detection
            - intent: Intent detection
            - virality_triggers: Virality trigger classification
            - brand_safety: Brand safety + risk flagging
            - hook_efficiency: Hook efficiency + compression scoring
        """
        logger.info("Analyzing text content")
        
        # Combine all text
        full_text = self._combine_text(caption, description, hashtags)
        
        # Analyze different aspects
        trend_proximity = await self._calculate_trend_proximity(full_text, hashtags)
        emotion = await self._detect_emotion(full_text)
        intent = await self._detect_intent(full_text)
        virality_triggers = await self._classify_virality_triggers(full_text)
        brand_safety = await self._assess_brand_safety(full_text)
        hook_efficiency = await self._score_hook_efficiency(caption, description)
        
        # Analyze comment quality (Algorithm 1 requirement)
        comment_quality = await self._analyze_comment_quality(metadata)
        
        result = {
            "trend_proximity": trend_proximity,
            "emotion": emotion,
            "intent": intent,
            "virality_triggers": virality_triggers,
            "brand_safety": brand_safety,
            "hook_efficiency": hook_efficiency,
            "comment_quality": comment_quality
        }
        
        logger.info("Text analysis complete")
        return result
    
    def _combine_text(
        self,
        caption: Optional[str],
        description: Optional[str],
        hashtags: Optional[List[str]]
    ) -> str:
        """Combine all text content."""
        parts = []
        if caption:
            parts.append(caption)
        if description:
            parts.append(description)
        if hashtags:
            parts.append(" ".join(hashtags))
        return " ".join(parts)
    
    async def _calculate_trend_proximity(
        self,
        text: str,
        hashtags: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Calculate trend and semantic proximity scoring."""
        try:
            # Common trending keywords/hashtags (can be expanded with real trend data)
            trending_keywords = [
                "viral", "trending", "fyp", "foryou", "foryoupage", "explore",
                "viralvideo", "trendingnow", "mustsee", "watchthis", "amazing",
                "incredible", "mindblowing", "gamechanger", "breaking", "news"
            ]
            
            text_lower = text.lower()
            hashtags_lower = [h.lower().replace("#", "") for h in (hashtags or [])]
            
            # Count trending keyword matches
            trend_matches = 0
            for keyword in trending_keywords:
                if keyword in text_lower:
                    trend_matches += 1
                for hashtag in hashtags_lower:
                    if keyword in hashtag:
                        trend_matches += 1
            
            # Normalize trend score (0-1)
            max_possible_matches = len(trending_keywords) + len(hashtags_lower) if hashtags_lower else len(trending_keywords)
            trend_score = min(trend_matches / max(max_possible_matches, 1), 1.0)
            
            # Semantic similarity (simplified - would use embeddings in production)
            # For now, use keyword-based approach
            semantic_score = trend_score * 0.8  # Simplified
            
            # Extract trending topics found
            trending_topics = []
            for keyword in trending_keywords:
                if keyword in text_lower or any(keyword in h for h in hashtags_lower):
                    trending_topics.append(keyword)
            
            return {
                "trend_score": float(trend_score),
                "semantic_similarity": float(semantic_score),
                "trending_topics": trending_topics[:10]  # Limit to 10
            }
        except Exception as e:
            logger.error(f"Error calculating trend proximity: {e}")
            return {
                "trend_score": 0.0,
                "semantic_similarity": 0.0,
                "trending_topics": []
            }
    
    async def _detect_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotions in text."""
        try:
            # Use sentiment analyzer if available (can be extended for emotion)
            if self.sentiment_analyzer and TRANSFORMERS_AVAILABLE:
                try:
                    # Truncate long text
                    text_truncated = text[:512]
                    results = self.sentiment_analyzer(text_truncated)
                    
                    # Map sentiment to emotions (simplified)
                    # Positive sentiment -> joy, surprise
                    # Negative sentiment -> sadness, anger, fear
                    if isinstance(results, list) and len(results) > 0:
                        # Get all scores
                        scores = {}
                        for result in results:
                            if isinstance(result, list):
                                for item in result:
                                    label = item.get("label", "").lower()
                                    score = item.get("score", 0.0)
                                    scores[label] = score
                            else:
                                label = result.get("label", "").lower()
                                score = result.get("score", 0.0)
                                scores[label] = score
                        
                        # Map to emotions
                        positive_score = scores.get("positive", 0.0) or scores.get("pos", 0.0)
                        negative_score = scores.get("negative", 0.0) or scores.get("neg", 0.0)
                        neutral_score = scores.get("neutral", 0.0) or (1.0 - positive_score - negative_score)
                        
                        return {
                            "joy": float(positive_score * 0.6),
                            "sadness": float(negative_score * 0.4),
                            "anger": float(negative_score * 0.3),
                            "fear": float(negative_score * 0.2),
                            "surprise": float(positive_score * 0.4),
                            "neutral": float(neutral_score)
                        }
                except Exception as e:
                    logger.warning(f"Error in emotion detection with model: {e}")
            
            # Fallback: keyword-based emotion detection
            text_lower = text.lower()
            
            # Emotion keywords
            joy_keywords = ["happy", "joy", "excited", "amazing", "love", "great", "wonderful", "fantastic", "😊", "😄", "❤️"]
            sadness_keywords = ["sad", "depressed", "unhappy", "disappointed", "😢", "😔"]
            anger_keywords = ["angry", "mad", "furious", "annoyed", "hate", "😠", "😡"]
            fear_keywords = ["scared", "afraid", "worried", "anxious", "fear", "😨", "😰"]
            surprise_keywords = ["wow", "surprised", "shocked", "unexpected", "incredible", "😲", "🤯"]
            
            joy_count = sum(1 for kw in joy_keywords if kw in text_lower)
            sadness_count = sum(1 for kw in sadness_keywords if kw in text_lower)
            anger_count = sum(1 for kw in anger_keywords if kw in text_lower)
            fear_count = sum(1 for kw in fear_keywords if kw in text_lower)
            surprise_count = sum(1 for kw in surprise_keywords if kw in text_lower)
            
            total_emotion_words = joy_count + sadness_count + anger_count + fear_count + surprise_count
            
            if total_emotion_words > 0:
                return {
                    "joy": float(joy_count / max(total_emotion_words, 1)),
                    "sadness": float(sadness_count / max(total_emotion_words, 1)),
                    "anger": float(anger_count / max(total_emotion_words, 1)),
                    "fear": float(fear_count / max(total_emotion_words, 1)),
                    "surprise": float(surprise_count / max(total_emotion_words, 1)),
                    "neutral": float(max(0, 1.0 - sum([joy_count, sadness_count, anger_count, fear_count, surprise_count]) / max(total_emotion_words, 1)))
                }
            
            return {
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "neutral": 1.0
            }
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return {
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "neutral": 1.0
            }
    
    async def _detect_intent(self, text: str) -> Dict[str, Any]:
        """Detect intent in text."""
        try:
            text_lower = text.lower()
            
            # Intent categories with keywords
            intent_keywords = {
                "informational": ["how", "what", "why", "when", "where", "explain", "tell", "learn", "know"],
                "entertainment": ["funny", "laugh", "joke", "comedy", "entertain", "watch", "enjoy"],
                "promotional": ["buy", "sale", "discount", "offer", "deal", "promo", "shop", "purchase"],
                "educational": ["teach", "learn", "tutorial", "guide", "lesson", "course", "education"],
                "social": ["share", "comment", "like", "follow", "tag", "mention", "connect"],
                "call_to_action": ["click", "visit", "check", "try", "download", "subscribe", "follow"],
                "question": ["?", "ask", "wonder", "curious", "question"],
                "opinion": ["think", "believe", "feel", "opinion", "view", "perspective"]
            }
            
            intent_scores = {}
            for intent, keywords in intent_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > 0:
                    intent_scores[intent] = score / len(keywords)  # Normalize
            
            if intent_scores:
                # Get primary intent
                primary_intent = max(intent_scores, key=intent_scores.get)
                intent_confidence = intent_scores[primary_intent]
                
                # Get all intents sorted by score
                intent_categories = sorted(
                    [(intent, score) for intent, score in intent_scores.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # Top 5
                
                return {
                    "primary_intent": primary_intent,
                    "intent_confidence": float(min(intent_confidence, 1.0)),
                    "intent_categories": [{"intent": intent, "score": float(score)} for intent, score in intent_categories]
                }
            
            return {
                "primary_intent": "unknown",
                "intent_confidence": 0.0,
                "intent_categories": []
            }
        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            return {
                "primary_intent": "unknown",
                "intent_confidence": 0.0,
                "intent_categories": []
            }
    
    async def _classify_virality_triggers(self, text: str) -> List[Dict[str, Any]]:
        """Classify virality triggers in text."""
        try:
            text_lower = text.lower()
            triggers = []
            
            # Virality trigger patterns
            trigger_patterns = {
                "curiosity_gap": {
                    "keywords": ["secret", "hidden", "nobody knows", "you won't believe", "shocking", "revealed"],
                    "score": 0.0
                },
                "emotional_appeal": {
                    "keywords": ["heartbreaking", "inspiring", "emotional", "touching", "powerful", "moving"],
                    "score": 0.0
                },
                "controversy": {
                    "keywords": ["controversial", "debate", "disagree", "unpopular opinion", "hot take"],
                    "score": 0.0
                },
                "urgency": {
                    "keywords": ["now", "urgent", "limited time", "don't miss", "hurry", "act fast"],
                    "score": 0.0
                },
                "social_proof": {
                    "keywords": ["everyone", "millions", "viral", "trending", "popular", "famous"],
                    "score": 0.0
                },
                "question_hook": {
                    "keywords": ["did you know", "have you ever", "what if", "why does", "how come"],
                    "score": 0.0
                },
                "number_hook": {
                    "keywords": ["top 10", "5 ways", "3 reasons", "7 secrets", "number one"],
                    "score": 0.0
                },
                "exclamation": {
                    "keywords": ["!", "amazing", "incredible", "unbelievable", "wow"],
                    "score": 0.0
                }
            }
            
            # Calculate scores for each trigger
            for trigger_name, trigger_data in trigger_patterns.items():
                matches = sum(1 for kw in trigger_data["keywords"] if kw in text_lower)
                if matches > 0:
                    score = min(matches / len(trigger_data["keywords"]), 1.0)
                    triggers.append({
                        "trigger": trigger_name,
                        "score": float(score),
                        "confidence": float(min(score * 1.2, 1.0))
                    })
            
            # Sort by score
            triggers.sort(key=lambda x: x["score"], reverse=True)
            
            return triggers[:10]  # Return top 10 triggers
        except Exception as e:
            logger.error(f"Error classifying virality triggers: {e}")
            return []
    
    async def _assess_brand_safety(self, text: str) -> Dict[str, Any]:
        """
        Assess brand safety + risk flagging.
        
        Returns:
            Dictionary with safety score and risk flags
        """
        try:
            text_lower = text.lower()
            risk_flags = []
            flagged_content = []
            
            # Risk categories
            risk_keywords = {
                "profanity": ["damn", "hell", "crap", "stupid", "idiot", "fool"],
                "violence": ["kill", "fight", "attack", "violence", "weapon", "gun", "knife"],
                "drugs": ["drug", "cocaine", "marijuana", "weed", "alcohol abuse", "drunk"],
                "adult_content": ["sex", "nude", "porn", "explicit", "adult"],
                "hate_speech": ["hate", "racist", "discrimination", "prejudice", "offensive"],
                "misinformation": ["fake news", "conspiracy", "hoax", "false", "lie"],
                "spam": ["click here", "free money", "guaranteed", "act now", "limited offer"]
            }
            
            safety_score = 1.0
            total_risks = 0
            
            for risk_category, keywords in risk_keywords.items():
                matches = [kw for kw in keywords if kw in text_lower]
                if matches:
                    total_risks += len(matches)
                    risk_flags.append({
                        "category": risk_category,
                        "severity": "high" if risk_category in ["violence", "hate_speech", "adult_content"] else "medium",
                        "matches": len(matches)
                    })
                    flagged_content.extend(matches[:3])  # Limit flagged content
                    # Reduce safety score
                    safety_score -= 0.15 if risk_category in ["violence", "hate_speech", "adult_content"] else 0.10
            
            # Normalize safety score
            safety_score = max(0.0, min(1.0, safety_score))
            
            # Determine risk level
            if total_risks == 0:
                risk_level = "low"
            elif total_risks <= 2:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            # Check for excessive exclamation/question marks (spam indicator)
            exclamation_count = text.count("!")
            question_count = text.count("?")
            if exclamation_count > 5 or question_count > 5:
                risk_flags.append({
                    "category": "spam",
                    "severity": "low",
                    "matches": exclamation_count + question_count
                })
                safety_score = max(0.0, safety_score - 0.05)
            
            return {
                "safety_score": float(safety_score),
                "risk_level": risk_level,
                "risk_flags": risk_flags,
                "flagged_content": list(set(flagged_content))[:10]  # Unique, limit to 10
            }
        except Exception as e:
            logger.error(f"Error assessing brand safety: {e}")
            return {
                "safety_score": 1.0,
                "risk_level": "low",
                "risk_flags": [],
                "flagged_content": []
            }
    
    async def _score_hook_efficiency(
        self,
        caption: Optional[str],
        description: Optional[str]
    ) -> Dict[str, Any]:
        """
        Score hook efficiency + compression.
        
        Analyzes how effectively the text captures attention.
        """
        try:
            # Combine text
            full_text = (caption or "") + " " + (description or "")
            full_text = full_text.strip()
            
            if not full_text:
                return {
                    "hook_score": 0.0,
                    "compression_score": 0.0,
                    "attention_capture": 0.0
                }
            
            # Hook efficiency factors
            # 1. First 3 words impact (hook location)
            words = full_text.split()
            first_3_words = " ".join(words[:3]).lower() if len(words) >= 3 else full_text.lower()
            
            # Hook keywords (attention-grabbing words)
            hook_keywords = [
                "you", "your", "this", "these", "amazing", "incredible", "secret",
                "shocking", "revealed", "watch", "see", "check", "look", "wow"
            ]
            hook_in_first_3 = sum(1 for kw in hook_keywords if kw in first_3_words)
            hook_location_score = min(hook_in_first_3 / 3.0, 1.0)
            
            # 2. Question or exclamation in first sentence
            first_sentence = full_text.split(".")[0] if "." in full_text else full_text.split("!")[0] if "!" in full_text else full_text
            has_question = "?" in first_sentence
            has_exclamation = "!" in first_sentence
            punctuation_score = 0.5 if (has_question or has_exclamation) else 0.0
            
            # 3. Length optimization (optimal: 20-200 characters for hook)
            text_length = len(full_text)
            if 20 <= text_length <= 200:
                length_score = 1.0
            elif text_length < 20:
                length_score = text_length / 20.0
            else:
                # Penalize very long text
                length_score = max(0.0, 1.0 - (text_length - 200) / 300.0)
            
            # 4. Compression score (information density)
            # Count meaningful words vs filler words
            filler_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
            words_lower = [w.lower().strip(".,!?") for w in words]
            meaningful_words = [w for w in words_lower if w not in filler_words and len(w) > 2]
            compression_ratio = len(meaningful_words) / max(len(words), 1)
            compression_score = min(compression_ratio * 1.5, 1.0)
            
            # 5. Attention capture (combination of factors)
            attention_capture = (
                hook_location_score * 0.3 +
                punctuation_score * 0.2 +
                length_score * 0.2 +
                compression_score * 0.3
            )
            
            # Overall hook score
            hook_score = (
                hook_location_score * 0.4 +
                punctuation_score * 0.3 +
                length_score * 0.3
            )
            
            return {
                "hook_score": float(hook_score),
                "compression_score": float(compression_score),
                "attention_capture": float(attention_capture)
            }
        except Exception as e:
            logger.error(f"Error scoring hook efficiency: {e}")
            return {
                "hook_score": 0.0,
                "compression_score": 0.0,
                "attention_capture": 0.0
            }
    
    async def _analyze_comment_quality(
        self,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze comment quality (Algorithm 1 requirement).
        
        Uses GPT/BERT for sentiment and semantic analysis of comments.
        
        Args:
            metadata: Content metadata (may contain comments)
        
        Returns:
            Dictionary with quality score and analysis
        """
        comments = metadata.get("comments", []) if metadata else []
        
        if not comments:
            return {
                "quality_score": 0.5,  # Default score if no comments
                "sentiment_score": 0.5,
                "engagement_score": 0.0,
                "comment_count": 0,
                "avg_comment_length": 0.0,
                "sentiment_distribution": {},
                "spam_ratio": 0.0
            }
        
        comment_count = len(comments)
        
        # Extract comment texts
        comment_texts = []
        for comment in comments:
            if isinstance(comment, dict):
                text = comment.get("text", "") or comment.get("content", "")
            elif isinstance(comment, str):
                text = comment
            else:
                text = str(comment)
            
            if text and len(text.strip()) > 0:
                comment_texts.append(text.strip())
        
        if not comment_texts:
            return {
                "quality_score": 0.5,
                "sentiment_score": 0.5,
                "engagement_score": 0.0,
                "comment_count": comment_count,
                "avg_comment_length": 0.0,
                "sentiment_distribution": {},
                "spam_ratio": 0.0
            }
        
        # Calculate basic metrics
        avg_length = sum(len(text) for text in comment_texts) / len(comment_texts)
        
        # Analyze sentiment using BERT/GPT model
        sentiment_scores = []
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        spam_indicators = 0
        
        if self.sentiment_analyzer and TRANSFORMERS_AVAILABLE:
            try:
                # Process comments in batches to avoid memory issues
                batch_size = 32
                for i in range(0, len(comment_texts), batch_size):
                    batch = comment_texts[i:i + batch_size]
                    # Truncate long comments (BERT has token limits)
                    batch = [text[:512] for text in batch]
                    
                    try:
                        results = self.sentiment_analyzer(batch)
                        
                        for result in results:
                            if isinstance(result, list) and len(result) > 0:
                                # Get the highest confidence sentiment
                                best_sentiment = max(result, key=lambda x: x.get("score", 0))
                                label = best_sentiment.get("label", "").lower()
                                score = best_sentiment.get("score", 0.5)
                                
                                sentiment_scores.append(score)
                                
                                # Classify sentiment
                                if "positive" in label or "pos" in label:
                                    sentiment_distribution["positive"] += 1
                                elif "negative" in label or "neg" in label:
                                    sentiment_distribution["negative"] += 1
                                else:
                                    sentiment_distribution["neutral"] += 1
                                
                                # Detect spam indicators (very low sentiment confidence + short text)
                                if score < 0.3 and len(batch[i % batch_size]) < 10:
                                    spam_indicators += 1
                    except Exception as e:
                        logger.warning(f"Error analyzing sentiment batch: {e}")
                        # Fallback: use neutral sentiment
                        sentiment_scores.extend([0.5] * len(batch))
                        sentiment_distribution["neutral"] += len(batch)
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                # Fallback to heuristics
                sentiment_scores = [0.5] * len(comment_texts)
        else:
            # Fallback: use heuristics for sentiment
            sentiment_scores = [0.5] * len(comment_texts)
            sentiment_distribution["neutral"] = len(comment_texts)
        
        # Calculate sentiment score (average of all comment sentiments)
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        else:
            avg_sentiment = 0.5
        
        # Calculate spam ratio
        spam_ratio = spam_indicators / max(comment_count, 1)
        
        # Calculate engagement score based on comment count and quality
        engagement_score = min(comment_count / 50.0, 1.0)
        
        # Calculate quality score using multiple factors:
        # 1. Sentiment positivity (positive comments indicate quality)
        positive_ratio = sentiment_distribution.get("positive", 0) / max(comment_count, 1)
        
        # 2. Average comment length (longer comments often more thoughtful)
        length_score = min(avg_length / 100.0, 1.0)
        
        # 3. Spam ratio (lower is better)
        spam_penalty = 1.0 - min(spam_ratio, 0.5)  # Cap penalty at 50% spam
        
        # 4. Engagement (more comments = more engagement)
        engagement_factor = min(comment_count / 100.0, 1.0)
        
        # Weighted quality score
        quality_score = (
            positive_ratio * 0.35 +           # Sentiment weight
            length_score * 0.25 +             # Length weight
            spam_penalty * 0.25 +             # Spam penalty weight
            engagement_factor * 0.15          # Engagement weight
        )
        
        # Ensure quality score is in [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Normalize sentiment distribution to ratios
        total_sentiments = sum(sentiment_distribution.values())
        if total_sentiments > 0:
            sentiment_distribution = {
                k: v / total_sentiments 
                for k, v in sentiment_distribution.items()
            }
        
        return {
            "quality_score": float(quality_score),
            "sentiment_score": float(avg_sentiment),
            "engagement_score": float(engagement_score),
            "comment_count": comment_count,
            "avg_comment_length": float(avg_length),
            "sentiment_distribution": sentiment_distribution,
            "spam_ratio": float(spam_ratio),
            "positive_ratio": float(positive_ratio)
        }


