from typing import List, Dict, Set, Tuple
import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
from collections import defaultdict
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Comprehensive retrieval evaluation metrics."""
    
    def __init__(self):
        self.results_history = []
    
    def evaluate_retrieval(self,
                          query: str,
                          retrieved_docs: List[Dict],
                          relevant_doc_ids: Set[str],
                          k_values: List[int] = None) -> Dict:
        """
        Comprehensive evaluation of retrieval results.
        
        Args:
            query: Search query
            retrieved_docs: List of retrieved document dictionaries
            relevant_doc_ids: Set of IDs of truly relevant documents
            k_values: List of k values for metrics@k
        
        Returns:
            Dictionary of evaluation metrics
        """
        k_values = k_values or [1, 3, 5, 10]
        retrieved_ids = [doc['id'] for doc in retrieved_docs]
        
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved_docs),
            'num_relevant': len(relevant_doc_ids)
        }
        
        # Calculate metrics for each k
        for k in k_values:
            retrieved_at_k = retrieved_ids[:k]
            relevant_retrieved = set(retrieved_at_k) & relevant_doc_ids
            
            # Precision@k
            precision = len(relevant_retrieved) / k if k > 0 else 0
            metrics[f'precision@{k}'] = precision
            
            # Recall@k
            recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0
            metrics[f'recall@{k}'] = recall
            
            # F1@k
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            metrics[f'f1@{k}'] = f1
            
            # Hit Rate@k (binary: did we retrieve at least one relevant doc?)
            metrics[f'hit_rate@{k}'] = 1.0 if relevant_retrieved else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr = self._calculate_mrr(retrieved_ids, relevant_doc_ids)
        metrics['mrr'] = mrr
        
        # Mean Average Precision (MAP)
        map_score = self._calculate_map(retrieved_ids, relevant_doc_ids)
        metrics['map'] = map_score
        
        # Normalized Discounted Cumulative Gain (NDCG)
        if relevant_doc_ids:
            ndcg = self._calculate_ndcg(retrieved_ids, relevant_doc_ids, max(k_values))
            metrics[f'ndcg@{max(k_values)}'] = ndcg
        
        # Average similarity score of retrieved docs
        if retrieved_docs:
            avg_similarity = np.mean([doc.get('similarity', 0) for doc in retrieved_docs])
            metrics['avg_similarity'] = avg_similarity
        
        # Diversity metrics
        diversity = self._calculate_diversity(retrieved_docs)
        metrics['diversity'] = diversity
        
        self.results_history.append(metrics)
        return metrics
    
    def _calculate_mrr(self, retrieved_ids: List[str], 
                       relevant_ids: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0
    
    def _calculate_map(self, retrieved_ids: List[str],
                       relevant_ids: Set[str]) -> float:
        """Calculate Mean Average Precision."""
        if not relevant_ids:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_ids) if relevant_ids else 0.0
    
    def _calculate_ndcg(self, retrieved_ids: List[str],
                        relevant_ids: Set[str],
                        k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # Create relevance scores (1 for relevant, 0 for not relevant)
        relevance_scores = [1 if doc_id in relevant_ids else 0 
                          for doc_id in retrieved_ids[:k]]
        
        # Ideal ranking (all relevant docs first)
        ideal_scores = sorted(relevance_scores, reverse=True)
        
        if sum(relevance_scores) == 0:
            return 0.0
        
        # Calculate NDCG
        try:
            ndcg = ndcg_score([ideal_scores], [relevance_scores])
            return ndcg
        except:
            return 0.0
    
    def _calculate_diversity(self, retrieved_docs: List[Dict]) -> float:
        """
        Calculate diversity of retrieved documents.
        Based on unique sources/chunks.
        """
        if not retrieved_docs:
            return 0.0
        
        unique_sources = set(doc['metadata'].get('source', '') for doc in retrieved_docs)
        diversity_ratio = len(unique_sources) / len(retrieved_docs)
        
        return diversity_ratio
    
    def batch_evaluate(self,
                      queries_and_relevant: List[Tuple[str, Set[str]]],
                      retriever,
                      k_values: List[int] = None) -> Dict:
        """
        Evaluate retrieval across multiple queries.
        
        Args:
            queries_and_relevant: List of (query, relevant_doc_ids) tuples
            retriever: Retriever instance
            k_values: List of k values for metrics
        
        Returns:
            Aggregated metrics across all queries
        """
        k_values = k_values or [1, 3, 5, 10]
        all_metrics = []
        
        logger.info(f"Evaluating {len(queries_and_relevant)} queries")
        
        for query, relevant_ids in queries_and_relevant:
            # Retrieve documents
            retrieved_docs = retriever.retrieve(query=query, top_k=max(k_values))
            
            # Evaluate
            metrics = self.evaluate_retrieval(
                query=query,
                retrieved_docs=retrieved_docs,
                relevant_doc_ids=relevant_ids,
                k_values=k_values
            )
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics, k_values)
        
        return aggregated
    
    def _aggregate_metrics(self, all_metrics: List[Dict], 
                          k_values: List[int]) -> Dict:
        """Aggregate metrics across queries."""
        aggregated = {
            'num_queries': len(all_metrics),
            'metrics': {}
        }
        
        # Collect all metric names
        metric_names = set()
        for metrics in all_metrics:
            metric_names.update(k for k in metrics.keys() 
                              if k not in ['query', 'num_retrieved', 'num_relevant'])
        
        # Calculate mean and std for each metric
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                aggregated['metrics'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated
    
    def compare_configurations(self,
                             configs: List[Dict],
                             queries_and_relevant: List[Tuple[str, Set[str]]],
                             retriever_class) -> pd.DataFrame:
        """
        Compare different retrieval configurations.
        
        Args:
            configs: List of configuration dictionaries
            queries_and_relevant: Test queries with relevant docs
            retriever_class: Retriever class to instantiate
        
        Returns:
            DataFrame comparing configurations
        """
        results = []
        
        for config in configs:
            logger.info(f"Testing config: {config['name']}")
            
            # Create retriever with this config
            # (Implementation depends on your retriever interface)
            # retriever = retriever_class(**config['params'])
            
            # For now, assuming standard retriever
            from src.retrieval.retriever import Retriever
            from src.vector_store.chroma_manager import ChromaDBManager
            
            chroma_manager = ChromaDBManager()
            retriever = Retriever(chroma_manager, use_reranking=config.get('use_reranking', True))
            
            # Evaluate
            metrics = self.batch_evaluate(queries_and_relevant, retriever)
            
            result = {'config_name': config['name']}
            for metric_name, values in metrics['metrics'].items():
                result[metric_name] = values['mean']
            
            results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def create_confusion_matrix(self,
                               queries_and_relevant: List[Tuple[str, Set[str]]],
                               retriever,
                               k: int = 5) -> Dict:
        """
        Create confusion matrix for retrieval.
        
        Args:
            queries_and_relevant: Test queries with relevant docs
            retriever: Retriever instance
            k: Number of results to consider
        
        Returns:
            Confusion matrix statistics
        """
        tp = 0  # True Positives: relevant docs retrieved
        fp = 0  # False Positives: non-relevant docs retrieved
        fn = 0  # False Negatives: relevant docs not retrieved
        
        for query, relevant_ids in queries_and_relevant:
            retrieved_docs = retriever.retrieve(query=query, top_k=k)
            retrieved_ids = set(doc['id'] for doc in retrieved_docs)
            
            tp += len(retrieved_ids & relevant_ids)
            fp += len(retrieved_ids - relevant_ids)
            fn += len(relevant_ids - retrieved_ids)
        
        # True Negatives are hard to define in retrieval
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def analyze_failure_cases(self,
                             queries_and_relevant: List[Tuple[str, Set[str]]],
                             retriever,
                             threshold: float = 0.5) -> List[Dict]:
        """
        Analyze queries where retrieval performed poorly.
        
        Args:
            queries_and_relevant: Test queries with relevant docs
            retriever: Retriever instance
            threshold: Minimum recall threshold
        
        Returns:
            List of failure case analyses
        """
        failures = []
        
        for query, relevant_ids in queries_and_relevant:
            retrieved_docs = retriever.retrieve(query=query)
            retrieved_ids = set(doc['id'] for doc in retrieved_docs)
            
            recall = len(retrieved_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0
            
            if recall < threshold:
                failure_analysis = {
                    'query': query,
                    'recall': recall,
                    'num_relevant': len(relevant_ids),
                    'num_retrieved': len(retrieved_ids),
                    'retrieved_relevant': len(retrieved_ids & relevant_ids),
                    'avg_similarity': np.mean([d.get('similarity', 0) for d in retrieved_docs]) if retrieved_docs else 0,
                    'retrieved_docs': retrieved_docs[:3]  # Top 3 for analysis
                }
                failures.append(failure_analysis)
        
        return failures
    
    def export_results(self, filepath: Path):
        """Export evaluation results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.results_history, f, indent=2)
        logger.info(f"Exported results to {filepath}")
    
    def generate_report(self, aggregated_metrics: Dict) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            aggregated_metrics: Aggregated metrics from batch_evaluate
        
        Returns:
            Formatted report string
        """
        report = ["=" * 60]
        report.append("RETRIEVAL EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Queries Evaluated: {aggregated_metrics['num_queries']}\n")
        
        report.append("Metrics Summary:")
        report.append("-" * 60)
        
        for metric_name, values in sorted(aggregated_metrics['metrics'].items()):
            report.append(f"\n{metric_name}:")
            report.append(f"  Mean: {values['mean']:.4f}")
            report.append(f"  Std:  {values['std']:.4f}")
            report.append(f"  Min:  {values['min']:.4f}")
            report.append(f"  Max:  {values['max']:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class TestDatasetGenerator:
    """Generate synthetic test datasets for evaluation."""
    
    @staticmethod
    def generate_test_queries(documents: List[Dict], 
                            num_queries: int = 50) -> List[Tuple[str, Set[str]]]:
        """
        Generate test queries from documents.
        
        Args:
            documents: List of document dictionaries with 'id' and 'content'
            num_queries: Number of test queries to generate
        
        Returns:
            List of (query, relevant_doc_ids) tuples
        """
        import random
        
        queries_and_relevant = []
        
        for _ in range(num_queries):
            # Select a random document
            doc = random.choice(documents)
            doc_id = doc['id']
            content = doc['content']
            
            # Extract a sentence or phrase as query
            sentences = content.split('.')
            if len(sentences) > 1:
                query_sentence = random.choice(sentences).strip()
                
                # Create variations
                words = query_sentence.split()
                if len(words) > 5:
                    # Take a subset of words
                    start = random.randint(0, len(words) - 5)
                    query = ' '.join(words[start:start + 5])
                else:
                    query = query_sentence
                
                queries_and_relevant.append((query, {doc_id}))
        
        return queries_and_relevant
    
    @staticmethod
    def create_golden_dataset(filepath: Path,
                            queries_and_relevant: List[Tuple[str, Set[str]]]):
        """Save golden dataset for future evaluation."""
        dataset = [
            {
                'query': query,
                'relevant_doc_ids': list(relevant_ids)
            }
            for query, relevant_ids in queries_and_relevant
        ]
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved golden dataset to {filepath}")
    
    @staticmethod
    def load_golden_dataset(filepath: Path) -> List[Tuple[str, Set[str]]]:
        """Load golden dataset."""
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        
        return [
            (item['query'], set(item['relevant_doc_ids']))
            for item in dataset
        ]