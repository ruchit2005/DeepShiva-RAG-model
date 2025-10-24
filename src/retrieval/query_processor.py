from typing import Optional, Dict, Any, List
from langchain_community.chat_models import ChatOpenAI
from config.settings import settings
from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.medical_terminology import expand_query_with_ayurvedic_terms
import logging
import re

logger = logging.getLogger(__name__)


class Gatekeeper:
    """
    Checks if queries are specific enough before processing.
    Generates clarification questions for vague requests.
    """
    
    def __init__(self):
        """Initialize gatekeeper with LLM."""
        self.enabled = False
        self.llm = None
        
        if settings.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.enabled = True
                logger.info("Gatekeeper initialized with LLM support")
            except Exception as e:
                logger.warning(f"Gatekeeper initialization failed: {e}. Running without gatekeeper.")
                self.enabled = False
        else:
            logger.info("Gatekeeper disabled (no OpenAI API key)")
    
    def check_query_clarity(self, query: str) -> Dict[str, Any]:
        """
        Check if query is specific enough to answer precisely.
        
        Args:
            query: User's query
        
        Returns:
            Dict with 'is_clear' (bool) and 'clarification' (str or None)
        """
        if not self.enabled:
            return {'is_clear': True, 'clarification': None}
        
        prompt = f"""You are an expert at identifying ambiguous queries in a medical/Ayurvedic knowledge base.

Analyze this user query and determine if it's specific enough to answer precisely.

A SPECIFIC query (ACCEPT these):
- Medical symptoms or health conditions (e.g., "stomach ache", "fever", "headache")
- Treatment inquiries (e.g., "treatment for diabetes", "remedy for cold")
- Disease or condition names (e.g., "what is Atisara?", "causes of jaundice")
- Specific medical questions (e.g., "dosage of Ashwagandha", "side effects of...")
- Examples: "I have stomach ache", "What causes excessive thirst?", "Treatment for piles"

An AMBIGUOUS query (REJECT these):
- Extremely vague without ANY medical context (e.g., "tell me something", "what's this about?")
- Open-ended non-medical queries (e.g., "give me an overview", "what do you know?")
- Queries with zero medical/health terms
- Examples: "Tell me about everything", "What's in this database?", "Give me information"

IMPORTANT: Medical symptom descriptions are ALWAYS specific enough, even if conversational.
"I am having stomach ache" is SPECIFIC. "Tell me about health" is AMBIGUOUS.

Query: "{query}"

If the query has ANY medical context or symptom, respond with just "OK".
If the query is completely vague with NO medical context, formulate ONE specific clarification question.

Response:"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            
            if response.upper() == "OK":
                logger.info(f"Query approved: '{query}'")
                return {'is_clear': True, 'clarification': None}
            else:
                logger.info(f"Query needs clarification: '{query}'")
                return {'is_clear': False, 'clarification': response}
        
        except Exception as e:
            logger.error(f"Gatekeeper error: {e}")
            # Fail open - allow query through
            return {'is_clear': True, 'clarification': None}


class QueryOptimizer:
    """
    Optimizes user queries for better retrieval performance.
    Rewrites vague queries into more precise, keyword-rich versions.
    Intelligently decides when optimization is needed based on query characteristics.
    """
    
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None,
                 similarity_threshold: float = 0.6,
                 lexical_threshold: float = 0.3):
        """Initialize query optimizer with LLM and optional embedding verifier.

        Args:
            embedding_manager: Optional EmbeddingManager used to verify optimized queries
            similarity_threshold: Minimum cosine similarity between original and optimized query embeddings
            lexical_threshold: Minimum lexical overlap ratio required between original and optimized queries
        """
        self.enabled = False
        self.llm = None
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold
        self.lexical_threshold = lexical_threshold

        if settings.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.enabled = True
                logger.info("QueryOptimizer initialized with LLM support")
            except Exception as e:
                logger.warning(f"QueryOptimizer initialization failed: {e}. Using original queries.")
                self.enabled = False
        else:
            logger.info("QueryOptimizer disabled (no OpenAI API key)")
    
    def should_optimize_query(self, query: str) -> bool:
        """
        Intelligently decide if query needs optimization based on characteristics.
        
        Optimization is SKIPPED when:
        - Query contains multiple specific medical/technical terms
        - Query has precise symptom descriptions
        - Query uses domain-specific terminology
        - Query is already detailed and specific
        
        Optimization is APPLIED when:
        - Query is vague or too short (<5 words)
        - Query lacks domain-specific terms
        - Query is conversational without technical precision
        - Query asks general questions
        
        Args:
            query: User's query
        
        Returns:
            bool: True if optimization would help, False if query is already precise
        """
        if not self.enabled:
            return False
        
        # 1. Length heuristic - very short queries might need expansion
        word_count = len(query.split())
        if word_count <= 3:
            logger.debug(f"Query optimization needed: too short ({word_count} words)")
            return True
        
        # 2. Check for medical/technical terminology patterns
        # Medical terms often have specific patterns
        medical_patterns = [
            r'\b(thirst|pain|ache|fever|bleeding|swelling|discharge|dullness|wetness|nausea|vomit|cough|rash)\b',
            r'\b(symptom|syndrome|disease|disorder|condition|infection)\b',
            r'\b(excessive|chronic|acute|severe|mild|persistent|recurring)\b',
            r'\b(abdomen|stomach|anal|chest|head|joint|muscle|throat|back)\b',
            r'\b(treatment|therapy|medication|diagnosis|remedy)\b',
        ]
        
        medical_term_count = sum(
            len(re.findall(pattern, query.lower())) 
            for pattern in medical_patterns
        )
        
        # If query has 3+ medical terms, it's likely already precise
        if medical_term_count >= 3:
            logger.debug(f"Query optimization skipped: contains {medical_term_count} medical terms (already precise)")
            return False
        
        # 3. Check for precise medical symptom descriptions
        # Queries with comma-separated symptoms are usually specific
        if ',' in query or ' and ' in query:
            symptom_count = query.count(',') + query.count(' and ')
            if symptom_count >= 2 and medical_term_count >= 2:
                logger.debug(f"Query optimization skipped: multiple specific symptoms listed")
                return False
        
        # 4. Check for vague/conversational patterns that need optimization
        vague_patterns = [
            r'\b(what is|tell me|give me|show me|find|search)\b',
            r'\b(overview|information|details|about)\b',
            r'\b(help|advice|suggest|recommend)\b',
            r'\b(general|overall|any|some)\b',
            r'\b(i am having|i have|i feel)\b',  # Conversational symptom reporting
        ]
        
        vague_term_count = sum(
            len(re.findall(pattern, query.lower())) 
            for pattern in vague_patterns
        )
        
        # If query has conversational patterns (even with 1 match), consider optimizing
        # unless it has many technical terms
        if vague_term_count >= 1 and medical_term_count < 3:
            logger.debug(f"Query optimization needed: conversational pattern with limited technical depth")
            return True
        
        if vague_term_count >= 2:
            logger.debug(f"Query optimization needed: contains {vague_term_count} vague terms")
            return True
        
        # 5. Check if query is a complete sentence vs keyword list
        # Questions and complete sentences often benefit from optimization
        if query.endswith('?') and word_count > 8:
            # Long questions might be conversational
            technical_ratio = medical_term_count / word_count
            if technical_ratio < 0.3:  # Less than 30% technical terms
                logger.debug(f"Query optimization needed: conversational question with low technical density ({technical_ratio:.1%})")
                return True
        
        # 6. Default: for medium-length queries with some but not many technical terms
        if word_count >= 5 and medical_term_count < 3:
            logger.debug(f"Query optimization needed: medium length with limited technical terms")
            return True
        
        # Default: don't optimize if query seems reasonably specific
        logger.debug(f"Query optimization skipped: query appears sufficiently precise")
        return False
    
    def optimize_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite query to be more effective for semantic search.
        Intelligently decides whether optimization is needed first.
        
        Args:
            query: Original user query
            context: Optional context about the domain
        
        Returns:
            Optimized query string (or original if optimization not needed)
        """
        if not self.enabled:
            return query
        
        # Step 1: Expand with Ayurvedic terminology first
        expanded_query = expand_query_with_ayurvedic_terms(query)
        if expanded_query != query:
            logger.info(f"Added Ayurvedic terms: '{query}' -> '{expanded_query}'")
            query = expanded_query
        
        # Step 2: Smart decision - check if LLM optimization would help
        if not self.should_optimize_query(query):
            logger.info(f"Query used as-is (already precise): '{query}'")
            return query
        
        context_info = context or "medical and Ayurvedic treatment documents"
        
        prompt = f"""You are an expert at optimizing search queries for medical/Ayurvedic knowledge retrieval.

Your task: Transform the user's query into a better search query for finding relevant information in {context_info}.

CRITICAL RULES FOR MEDICAL QUERIES:
1. PRESERVE all medical terms and symptoms mentioned
2. Only add closely related medical synonyms if helpful
3. DO NOT add generic terms like "medical reports", "patient case studies", "diagnostic criteria"
4. DO NOT over-expand - keep queries focused
5. Remove conversational filler ("i am having" → keep symptom)

Examples:
- "I have stomach ache" → "stomach ache abdominal pain gastric discomfort"
- "excessive thirst and fever" → "excessive thirst polydipsia fever pyrexia"
- "treatment for diabetes" → "diabetes treatment management Ayurvedic remedies"

Guidelines:
- Add medical synonyms or related terms ONLY if directly relevant
- Keep it concise (max 10-15 words)
- Focus on the core medical terms
- Remove filler words like "I am", "I have", "tell me"

Original Query: "{query}"

Optimized Query (concise, focused, no generic terms):"""
        
        try:
            optimized = self.llm.invoke(prompt).content.strip()

            # Remove quotes if LLM added them
            optimized = optimized.strip('"').strip("'")

            # Verification: if embedding manager available, ensure optimizer did not drift
            if self.embedding_manager:
                try:
                    orig_emb = self.embedding_manager.embed_query(query)
                    opt_emb = self.embedding_manager.embed_query(optimized)
                    similarity = self.embedding_manager.compute_similarity(orig_emb, opt_emb)

                    # Lexical overlap heuristic
                    words_orig = set(w for w in re.findall(r"\w+", query.lower()) if len(w) > 2)
                    words_opt = set(w for w in re.findall(r"\w+", optimized.lower()) if len(w) > 2)
                    overlap_ratio = (len(words_orig & words_opt) / max(1, len(words_orig))) if words_orig else 0.0

                    logger.debug(f"Optimizer verification: sim={similarity:.3f}, overlap={overlap_ratio:.3f}")

                    # More lenient thresholds: accept if EITHER similarity OR lexical overlap is decent
                    # This allows expansion of short queries while preventing total drift
                    if similarity < 0.4 and overlap_ratio < 0.2:  # Both very low = total drift
                        logger.warning(
                            "Optimized query appears to have drifted from original (low semantic/lexical overlap)."
                            " Falling back to original query." 
                        )
                        return query
                except Exception as e:
                    logger.warning(f"Error during optimized-query verification: {e}")

            logger.info(f"Query optimized: '{query}' -> '{optimized}'")
            return optimized

        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return query  # Fall back to original
    
    def generate_multiple_queries(self, query: str, num_variations: int = 3) -> list[str]:
        """
        Generate multiple query variations for multi-query retrieval.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
        
        Returns:
            List of query variations
        """
        if not self.enabled:
            return [query]
        
        prompt = f"""Generate {num_variations} different ways to phrase this query for search.
Each variation should emphasize different aspects or keywords.

Original Query: "{query}"

Provide exactly {num_variations} alternative phrasings, one per line:"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Include original query
            all_queries = [query] + variations[:num_variations]
            
            logger.info(f"Generated {len(all_queries)} query variations")
            return all_queries
        
        except Exception as e:
            logger.error(f"Query variation generation failed: {e}")
            return [query]


class Auditor:
    """
    Validates retrieval results for quality and consistency.
    Can trigger re-planning if results are weak.
    """
    
    def __init__(self):
        """Initialize auditor with LLM."""
        self.enabled = False
        self.llm = None
        
        if settings.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.enabled = True
                logger.info("Auditor initialized with LLM support")
            except Exception as e:
                logger.warning(f"Auditor initialization failed: {e}. Running without validation.")
                self.enabled = False
        else:
            logger.info("Auditor disabled (no OpenAI API key)")
    
    def validate_results(self, query: str, results: list[Dict[str, Any]], 
                        threshold: float = 0.3) -> Dict[str, Any]:
        """
        Validate if retrieval results are sufficient to answer the query.
        
        Args:
            query: Original query
            results: List of retrieved documents
            threshold: Minimum quality threshold
        
        Returns:
            Dict with 'is_valid', 'confidence', 'issues', and 'suggestion'
        """
        if not self.enabled or not results:
            return {
                'is_valid': len(results) > 0,
                'confidence': 0.5 if results else 0.0,
                'issues': [] if results else ["No results found"],
                'suggestion': None
            }
        
        # Check basic quality indicators
        issues = []
        
        # 1. Check if we have enough results
        if len(results) < 2:
            issues.append("Too few results retrieved")
        
        # 2. Check similarity scores
        avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results)
        if avg_similarity < threshold:
            issues.append(f"Low average similarity score: {avg_similarity:.2f}")
        
        # 3. Check diversity of sources
        sources = set(r.get('metadata', {}).get('source', 'unknown') for r in results)
        if len(sources) == 1 and len(results) > 3:
            issues.append("Results from only one source (low diversity)")
        
        # 4. LLM-based validation
        try:
            # Sample top 3 results for LLM review
            sample_results = results[:3]
            content_samples = "\n\n".join([
                f"Result {i+1}: {r.get('content', '')[:200]}..."
                for i, r in enumerate(sample_results)
            ])
            
            prompt = f"""Evaluate if these search results can answer the user's query.

Query: "{query}"

Top Results:
{content_samples}

Evaluation:
1. Can these results answer the query? (yes/no)
2. Confidence level (0-1)
3. Any issues or gaps?
4. Suggestion for improvement (if needed)

Provide concise response:"""
            
            response = self.llm.invoke(prompt).content.strip()
            
            # Parse response (simple heuristic)
            is_valid = "yes" in response.lower()[:50]
            
            logger.info(f"Auditor validation: {'PASS' if is_valid else 'FAIL'}")
            
            return {
                'is_valid': is_valid and len(issues) < 2,
                'confidence': avg_similarity,
                'issues': issues,
                'suggestion': response if not is_valid else None
            }
        
        except Exception as e:
            logger.error(f"Auditor validation error: {e}")
            return {
                'is_valid': len(issues) < 2,
                'confidence': avg_similarity,
                'issues': issues,
                'suggestion': None
            }


class Strategist:
    """
    Intelligently selects the best retrieval strategy based on query characteristics.
    Decides between: basic retrieval, MMR (diversity), hybrid search, or context-aware retrieval.
    """
    
    def __init__(self):
        """Initialize strategist."""
        self.enabled = True
        logger.info("Strategist initialized")
    
    def select_strategy(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze query and select optimal retrieval strategy.
        
        Args:
            query: User's search query
            context: Optional context (previous results, user feedback, etc.)
        
        Returns:
            Dict with 'strategy', 'params', and 'reasoning'
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Analyze query characteristics
        is_question = query.strip().endswith('?')
        has_comparison = any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'between'])
        has_list_intent = any(word in query_lower for word in ['list', 'all', 'types of', 'kinds of', 'examples'])
        has_specific_entity = self._has_specific_entity(query)
        is_exploratory = any(word in query_lower for word in ['overview', 'about', 'general', 'introduction'])
        
        # Decision Logic
        
        # 1. Context-Aware Retrieval: For specific entity queries where surrounding context matters
        if has_specific_entity and not has_list_intent and word_count <= 8:
            return {
                'strategy': 'context_aware',
                'params': {'context_window': 1},
                'reasoning': 'Specific entity query - retrieving with surrounding context'
            }
        
        # 2. MMR (Diversity): For exploratory, comparison, or list queries
        if has_comparison or has_list_intent or is_exploratory:
            diversity_factor = 0.7 if has_comparison else 0.6
            return {
                'strategy': 'mmr',
                'params': {'diversity_factor': diversity_factor},
                'reasoning': f'Query needs diverse results (comparison/list/exploratory)'
            }
        
        # 3. Hybrid Search: For questions that might benefit from keyword matching
        if is_question and word_count >= 5:
            return {
                'strategy': 'hybrid',
                'params': {'semantic_weight': 0.7},
                'reasoning': 'Question query - combining semantic and keyword search'
            }
        
        # 4. Basic Retrieval: Default for precise, short queries
        return {
            'strategy': 'basic',
            'params': {},
            'reasoning': 'Precise query - using standard semantic search'
        }
    
    def _has_specific_entity(self, query: str) -> bool:
        """
        Check if query mentions a specific medical condition, treatment, or entity.
        
        Args:
            query: User's query
        
        Returns:
            bool: True if specific entity detected
        """
        query_lower = query.lower()
        
        # Medical conditions (Ayurvedic and Western)
        conditions = [
            'atisara', 'jwara', 'arsha', 'grahani', 'prameha', 'kamala', 'pakshaghata',
            'diabetes', 'fever', 'diarrhea', 'constipation', 'asthma', 'epilepsy',
            'hemorrhoids', 'jaundice', 'arthritis'
        ]
        
        # Specific treatments or herbs
        treatments = [
            'triphala', 'ashwagandha', 'brahmi', 'guggulu', 'shatavari',
            'panchakarma', 'virechana', 'basti', 'nasya'
        ]
        
        # Check for specific entities
        for entity in conditions + treatments:
            if entity in query_lower:
                return True
        
        # Check for pattern: "what is X" or "treatment for X"
        if 'what is' in query_lower or 'treatment for' in query_lower:
            return True
        
        return False
    
    def should_retry_with_different_strategy(self, 
                                            query: str,
                                            current_strategy: str,
                                            results: List[Dict],
                                            min_results_threshold: int = 2) -> Optional[Dict[str, Any]]:
        """
        Decide if retrieval should be retried with a different strategy.
        
        Args:
            query: Original query
            current_strategy: Strategy that was just used
            results: Results from current strategy
            min_results_threshold: Minimum acceptable number of results
        
        Returns:
            New strategy dict if retry needed, None otherwise
        """
        # Check if results are insufficient
        if len(results) >= min_results_threshold:
            return None
        
        logger.warning(f"Insufficient results ({len(results)}) with {current_strategy} strategy")
        
        # Fallback strategy chain
        fallback_chain = {
            'basic': 'hybrid',
            'hybrid': 'mmr',
            'mmr': 'context_aware',
            'context_aware': None  # No more fallbacks
        }
        
        next_strategy = fallback_chain.get(current_strategy)
        
        if next_strategy:
            logger.info(f"Retrying with {next_strategy} strategy")
            return self.select_strategy_by_name(next_strategy, query)
        
        return None
    
    def select_strategy_by_name(self, strategy_name: str, query: str) -> Dict[str, Any]:
        """
        Get strategy configuration by name.
        
        Args:
            strategy_name: Name of strategy (basic, mmr, hybrid, context_aware)
            query: User's query
        
        Returns:
            Strategy configuration dict
        """
        strategies = {
            'basic': {
                'strategy': 'basic',
                'params': {},
                'reasoning': 'Manual selection: basic retrieval'
            },
            'mmr': {
                'strategy': 'mmr',
                'params': {'diversity_factor': 0.6},
                'reasoning': 'Manual selection: MMR for diversity'
            },
            'hybrid': {
                'strategy': 'hybrid',
                'params': {'semantic_weight': 0.7},
                'reasoning': 'Manual selection: hybrid semantic+keyword'
            },
            'context_aware': {
                'strategy': 'context_aware',
                'params': {'context_window': 1},
                'reasoning': 'Manual selection: context-aware retrieval'
            }
        }
        
        return strategies.get(strategy_name, strategies['basic'])
