"""
MEMORA - Evaluation Script
Collect embeddings, compute PR curves, tune thresholds for IC classification
Outputs: CSV, JSON report, SQLite evaluation_logs
"""

import numpy as np
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import uuid
import warnings

warnings.filterwarnings("ignore")

from biometric_pipeline import BiometricPipeline
from identity_table import IdentityTable


class EvaluationRunner:
    """
    Runs comprehensive evaluation: data collection → metrics → PR curves → threshold tuning
    """

    def __init__(self, pipeline: BiometricPipeline, num_users: int = 5,
                 samples_per_user: int = 5):
        """
        Initialize evaluation runner

        Args:
            pipeline: BiometricPipeline instance
            num_users: Number of users to collect (5-10)
            samples_per_user: Samples per user (typically 5)
        """
        self.pipeline = pipeline
        self.identity_table = pipeline.identity_table
        self.num_users = num_users
        self.samples_per_user = samples_per_user
        self.collected_users = []
        self.intra_user_pairs = []
        self.inter_user_pairs = []

        print(f"\n[EvaluationRunner] ✓ Initialized")
        print(f"  Target: {num_users} users × {samples_per_user} samples = {num_users * samples_per_user} total")

    def collect_data(self) -> bool:
        """
        Phase 1: Collect embeddings from multiple users

        Returns:
            True if successful
        """
        print(f"\n{'='*60}")
        print(f"PHASE 1: DATA COLLECTION")
        print(f"{'='*60}")

        # Check existing users in DB
        existing_users = self.identity_table.list_all()
        print(f"\nExisting users in DB: {len(existing_users)}")

        users_to_collect = self.num_users - len(existing_users)

        if users_to_collect > 0:
            print(f"\nNeed to collect {users_to_collect} more users\n")

            for i in range(users_to_collect):
                user_num = len(existing_users) + i + 1
                user_name = f"EvalUser{user_num:02d}"

                print(f"\n[{i+1}/{users_to_collect}] Registering {user_name}...")

                # Register with shorter duration for eval (5 seconds per sample)
                for sample_idx in range(self.samples_per_user):
                    print(f"  Sample {sample_idx + 1}/{self.samples_per_user}")
                    success = self.pipeline.register_user(user_name, duration_sec=5)

                    if not success:
                        print(f"  ✗ Failed to register {user_name} sample {sample_idx + 1}")
                        return False

                    time.sleep(1)  # Brief pause between samples

        else:
            print(f"✓ Already have {len(existing_users)} users, no new collection needed\n")

        # Build list of all registered users with their embeddings
        all_users = self.identity_table.list_all()
        print(f"\n✓ Total users in DB: {len(all_users)}")

        for user in all_users:
            user_data = self.identity_table.get_identity(user['user_id'])
            self.collected_users.append({
                'user_id': user['user_id'],
                'name': user['name'],
                'face_vector': user_data['face_vector'],
                'voice_vector': user_data['voice_vector']
            })

        return True

    def compute_metrics(self) -> bool:
        """
        Phase 2: Compute intra-user and inter-user similarity metrics

        Returns:
            True if successful
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2: METRICS COMPUTATION")
        print(f"{'='*60}")

        # Group users by name
        users_by_name = {}
        for user_data in self.collected_users:
            name = user_data['name']
            if name not in users_by_name:
                users_by_name[name] = []
            users_by_name[name].append(user_data)

        print(f"\nFound {len(users_by_name)} unique users")

        # Intra-user pairs: compare samples from same user
        print(f"\nComputing intra-user similarities...")
        intra_count = 0
        for user_name, samples in users_by_name.items():
            # Compare all pairs within this user's samples
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    face_sim = float(np.dot(samples[i]['face_vector'], samples[j]['face_vector']))
                    voice_sim = float(np.dot(samples[i]['voice_vector'], samples[j]['voice_vector']))

                    # Use IC fuser to get combined score
                    ic_score = self.pipeline.ic_fuser.fuse(
                        samples[i]['face_vector'], samples[i]['voice_vector'],
                        samples[j]['face_vector'], samples[j]['voice_vector']
                    )

                    self.intra_user_pairs.append({
                        'user_1': samples[i]['user_id'],
                        'user_2': samples[j]['user_id'],
                        'face_sim': face_sim,
                        'voice_sim': voice_sim,
                        'ic_score': ic_score,
                        'label': 1  # Match (same user)
                    })
                    intra_count += 1

        print(f"  ✓ Generated {intra_count} intra-user pairs")

        # Inter-user pairs: compare samples from different users
        print(f"\nComputing inter-user similarities...")
        users_list = list(users_by_name.keys())
        inter_count = 0

        # Sample inter-user pairs (don't need all O(n²) comparisons)
        for i in range(len(users_list)):
            for j in range(i + 1, min(i + 3, len(users_list))):  # Compare each user with 2-3 others
                user_1_name = users_list[i]
                user_2_name = users_list[j]

                sample_1 = users_by_name[user_1_name][0]  # First sample from each user
                sample_2 = users_by_name[user_2_name][0]

                face_sim = float(np.dot(sample_1['face_vector'], sample_2['face_vector']))
                voice_sim = float(np.dot(sample_1['voice_vector'], sample_2['voice_vector']))

                ic_score = self.pipeline.ic_fuser.fuse(
                    sample_1['face_vector'], sample_1['voice_vector'],
                    sample_2['face_vector'], sample_2['voice_vector']
                )

                self.inter_user_pairs.append({
                    'user_1': sample_1['user_id'],
                    'user_2': sample_2['user_id'],
                    'face_sim': face_sim,
                    'voice_sim': voice_sim,
                    'ic_score': ic_score,
                    'label': 0  # Non-match (different users)
                })
                inter_count += 1

        print(f"  ✓ Generated {inter_count} inter-user pairs")

        print(f"\n✓ Total pairs: {len(self.intra_user_pairs) + len(self.inter_user_pairs)}")
        print(f"  Intra-user (matches): {len(self.intra_user_pairs)}")
        print(f"  Inter-user (non-matches): {len(self.inter_user_pairs)}")

        return True

    def optimize_threshold(self) -> Dict:
        """
        Phase 3: Scan thresholds and compute precision-recall curve

        Returns:
            Dict with optimization results
        """
        print(f"\n{'='*60}")
        print(f"PHASE 3: THRESHOLD OPTIMIZATION")
        print(f"{'='*60}\n")

        # Combine all pairs
        all_pairs = self.intra_user_pairs + self.inter_user_pairs

        # Scan thresholds
        thresholds = np.arange(0.50, 1.01, 0.01)
        results = []

        for threshold in thresholds:
            tp = 0  # True positives (intra-user pairs above threshold)
            fp = 0  # False positives (inter-user pairs above threshold)
            fn = 0  # False negatives (intra-user pairs below threshold)

            for pair in all_pairs:
                is_match = pair['label'] == 1

                if pair['ic_score'] >= threshold:
                    if is_match:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if is_match:
                        fn += 1

            # Compute metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + len(self.inter_user_pairs) - fp) if len(self.inter_user_pairs) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'threshold': threshold,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            if threshold % 0.1 < 0.01:  # Print every 0.1
                print(f"  θ={threshold:.2f}: TPR={tpr:.2f}, FPR={fpr:.2f}, Prec={precision:.2f}, "
                      f"Rec={recall:.2f}, F1={f1:.3f}")

        # Find optimal threshold (max F1-score)
        optimal_result = max(results, key=lambda x: x['f1_score'])
        optimal_theta = optimal_result['threshold']

        print(f"\n✓ Optimal threshold (max F1): {optimal_theta:.2f}")
        print(f"  F1-score: {optimal_result['f1_score']:.4f}")
        print(f"  Precision: {optimal_result['precision']:.4f}")
        print(f"  Recall: {optimal_result['recall']:.4f}")

        return {
            'results': results,
            'optimal_threshold': optimal_theta,
            'optimal_metric': optimal_result
        }

    def generate_report(self, threshold_results: Dict, output_prefix: str = "results_evaluation") -> Tuple[str, str, str]:
        """
        Phase 4: Generate CSV, JSON, and SQLite outputs

        Args:
            threshold_results: Results from optimize_threshold()
            output_prefix: Prefix for output files

        Returns:
            (csv_path, json_path, timestamp)
        """
        print(f"\n{'='*60}")
        print(f"PHASE 4: REPORT GENERATION")
        print(f"{'='*60}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{output_prefix}_{timestamp}.csv"
        json_path = f"{output_prefix}_{timestamp}.json"

        # Generate CSV
        print(f"Writing CSV: {csv_path}")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['threshold', 'tpr', 'fpr', 'precision', 'recall', 'f1_score'])
            writer.writeheader()
            for result in threshold_results['results']:
                writer.writerow({
                    'threshold': f"{result['threshold']:.2f}",
                    'tpr': f"{result['tpr']:.4f}",
                    'fpr': f"{result['fpr']:.4f}",
                    'precision': f"{result['precision']:.4f}",
                    'recall': f"{result['recall']:.4f}",
                    'f1_score': f"{result['f1_score']:.4f}"
                })

        # Generate JSON
        print(f"Writing JSON: {json_path}")

        json_output = {
            'metadata': {
                'num_users': len(self.collected_users),
                'samples_per_user': self.samples_per_user,
                'intra_pairs': len(self.intra_user_pairs),
                'inter_pairs': len(self.inter_user_pairs),
                'timestamp': timestamp,
                'model_versions': {
                    'face': 'buffalo_s',
                    'speaker': 'ECAPA-TDNN'
                }
            },
            'results': threshold_results['results'],
            'recommended': {
                'theta_known': float(threshold_results['optimal_threshold']),
                'theta_high': float(threshold_results['optimal_threshold'] + 0.10),
                'theta_medium': float(threshold_results['optimal_threshold']),
                'optimal_f1_threshold': float(threshold_results['optimal_threshold']),
                'reasoning': f"Optimal at F1={threshold_results['optimal_metric']['f1_score']:.4f}"
            }
        }

        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)

        # Log to SQLite
        print(f"Logging to SQLite...")
        eval_count = 0
        for pair in self.intra_user_pairs + self.inter_user_pairs:
            eval_id = str(uuid.uuid4())
            test_type = "intra_user" if pair['label'] == 1 else "inter_user"

            # Use optimal threshold for decision
            decision = "MATCH" if pair['ic_score'] >= threshold_results['optimal_threshold'] else "REJECT"

            self.identity_table.log_evaluation(
                eval_id=eval_id,
                test_type=test_type,
                user_id_1=pair['user_1'],
                user_id_2=pair['user_2'],
                ic_score=pair['ic_score'],
                face_sim=pair['face_sim'],
                voice_sim=pair['voice_sim'],
                threshold_applied=threshold_results['optimal_threshold'],
                decision=decision,
                metadata={'expected_label': int(pair['label'])}
            )
            eval_count += 1

        print(f"  ✓ Logged {eval_count} evaluation entries")

        print(f"\n✓ Report generation complete")
        return csv_path, json_path, timestamp

    def run(self) -> bool:
        """Run full evaluation pipeline"""
        try:
            # Phase 1: Data collection
            if not self.collect_data():
                return False

            # Phase 2: Metrics
            if not self.compute_metrics():
                return False

            # Phase 3: Threshold optimization
            threshold_results = self.optimize_threshold()

            # Phase 4: Report generation
            csv_path, json_path, timestamp = self.generate_report(threshold_results)

            # Summary
            print(f"\n{'='*60}")
            print(f"✓ EVALUATION COMPLETE")
            print(f"{'='*60}")
            print(f"\nOutputs:")
            print(f"  CSV:     {csv_path}")
            print(f"  JSON:    {json_path}")
            print(f"  SQLite:  memora_identity.db (evaluation_logs table)")
            print(f"\nRecommended Thresholds:")
            print(f"  θ_known:  {threshold_results['optimal_metric']['threshold']:.2f}")
            print(f"  θ_high:   {threshold_results['optimal_metric']['threshold'] + 0.10:.2f}")
            print(f"  θ_medium: {threshold_results['optimal_metric']['threshold']:.2f}")
            print(f"\nMetrics @ Optimal Threshold:")
            print(f"  Precision: {threshold_results['optimal_metric']['precision']:.4f}")
            print(f"  Recall:    {threshold_results['optimal_metric']['recall']:.4f}")
            print(f"  F1-score:  {threshold_results['optimal_metric']['f1_score']:.4f}")
            print(f"{'='*60}\n")

            return True

        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for evaluation"""
    import torch

    print("\n" + "="*60)
    print("MEMORA - Evaluation Framework")
    print("="*60)
    print("Collect embeddings → Compute metrics → Tune thresholds")
    print("="*60 + "\n")

    # Initialize pipeline
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Evaluation] Using device: {device}\n")
        pipeline = BiometricPipeline(device=device)
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return

    # Get user input
    print("\nEvaluation Configuration:")
    num_users_input = input("Number of users (5-10, default 5): ").strip()
    try:
        num_users = int(num_users_input) if num_users_input else 5
        num_users = max(3, min(num_users, 10))  # Clamp to [3, 10]
    except ValueError:
        num_users = 5

    samples_input = input("Samples per user (1-10, default 5): ").strip()
    try:
        samples_per_user = int(samples_input) if samples_input else 5
        samples_per_user = max(1, min(samples_per_user, 10))
    except ValueError:
        samples_per_user = 5

    print(f"\n✓ Configuration: {num_users} users × {samples_per_user} samples per user\n")

    # Run evaluation
    runner = EvaluationRunner(pipeline, num_users=num_users, samples_per_user=samples_per_user)
    success = runner.run()

    if success:
        print("✓ Evaluation successful!")
    else:
        print("✗ Evaluation failed!")


if __name__ == "__main__":
    main()
