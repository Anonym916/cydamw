# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import csv
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
import scipy.stats


class PseudoLabelMonitor:
    """
    Monitor for pseudo-label quality and gradient impact analysis.
    
    Tracks:
    1. Wrong pseudo-label frequency across different confidence thresholds
    2. Gradient impact of wrong pseudo-labels
    3. Downstream accuracy impact correlation
    """

    def __init__(self, args, save_dir: str, rank: int = 0, world_size: int = 1):
        self.args = args
        self.save_dir = save_dir
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Configuration
        self.enable = getattr(args, 'monitor_enable', False)
        if not self.enable:
            return

        self.tau_list = getattr(args, 'monitor_tau_list', [0.95])
        self.log_interval = getattr(args, 'monitor_log_interval', 10)
        self.grad_probe_interval = getattr(args, 'monitor_grad_probe_interval', 20)
        self.acc_probe_interval = getattr(args, 'monitor_acc_probe_interval', 200)
        self.acc_probe_horizon_steps = getattr(args, 'monitor_acc_probe_horizon_steps', [50, 100, 200, 500])
        self.output_subdir = getattr(args, 'monitor_output_subdir', 'monitor_stats')
        self.save_step_jsonl = getattr(args, 'monitor_save_step_jsonl', True)
        self.save_summary_csv = getattr(args, 'monitor_save_summary_csv', True)
        self.save_per_tau_csv = getattr(args, 'monitor_save_per_tau_csv', True)
        self.save_plots = getattr(args, 'monitor_save_plots', False)
        self.max_eval_batches_for_probe = getattr(args, 'monitor_max_eval_batches_for_probe', 0)
        self.use_true_labels = getattr(args, 'monitor_use_unlabeled_true_labels_for_monitor_only', True)
        self.enabled_metrics = set(getattr(args, 'monitor_enabled_metrics', []))

        # Training metadata
        self.optimizer_name = getattr(args, 'optim', 'unknown')
        self.algorithm = getattr(args, 'algorithm', 'unknown')
        self.dataset = getattr(args, 'dataset', 'unknown')
        self.backbone = getattr(args, 'net', 'unknown')
        self.seed = getattr(args, 'seed', 0)
        self.train_tau = getattr(args, 'p_cutoff', 0.95)

        # State
        self.global_step = 0
        self.epoch = 0

        # Per-tau statistics accumulators
        self.tau_stats = {tau: {
            'accepted_count': 0,
            'accepted_wrong_count': 0,
            'accepted_correct_count': 0,
        } for tau in self.tau_list}

        # Train tau stats
        self.train_tau_stats = {
            'accepted_count': 0,
            'accepted_wrong_count': 0,
        }

        # Gradient event history
        self.gradient_events = []

        # Evaluation history
        self.eval_history = []

        # Output paths
        self.output_dir = os.path.join(save_dir, self.output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        # File handles
        self.step_jsonl_file = None
        if self.save_step_jsonl and self.is_main_process:
            self.step_jsonl_path = os.path.join(self.output_dir, 'step_stats.jsonl')
            self.step_jsonl_file = open(self.step_jsonl_path, 'w')

        # Initialize gradient impact probe
        self.grad_probe = GradientImpactProbe() if 'gradient_impact' in self.enabled_metrics else None

        # Initialize accuracy impact tracker
        self.acc_tracker = AccuracyImpactTracker(self.acc_probe_horizon_steps) if 'downstream_accuracy_impact' in self.enabled_metrics else None

    def update_pseudo_label_stats(self, logits_ulb_w: torch.Tensor, targets_ulb: torch.Tensor):
        """Update wrong pseudo-label frequency statistics"""
        if not self.enable or 'wrong_pseudo_label_frequency' not in self.enabled_metrics:
            return

        probs = torch.softmax(logits_ulb_w, dim=-1)
        max_probs, pseudo_labels = probs.max(dim=-1)

        batch_size = logits_ulb_w.shape[0]

        # Update per-tau stats
        for tau in self.tau_list:
            mask = max_probs >= tau
            accepted_count = mask.sum().item()
            accepted_indices = mask.nonzero(as_tuple=True)[0]

            if accepted_count > 0:
                accepted_targets = targets_ulb[accepted_indices]
                accepted_pseudo = pseudo_labels[accepted_indices]
                wrong_count = (accepted_targets != accepted_pseudo).sum().item()
                correct_count = accepted_count - wrong_count
            else:
                wrong_count = 0
                correct_count = 0

            self.tau_stats[tau]['accepted_count'] += accepted_count
            self.tau_stats[tau]['accepted_wrong_count'] += wrong_count
            self.tau_stats[tau]['accepted_correct_count'] += correct_count

        # Update train tau stats
        mask = max_probs >= self.train_tau
        accepted_count = mask.sum().item()
        accepted_indices = mask.nonzero(as_tuple=True)[0]

        if accepted_count > 0:
            accepted_targets = targets_ulb[accepted_indices]
            accepted_pseudo = pseudo_labels[accepted_indices]
            wrong_count = (accepted_targets != accepted_pseudo).sum().item()
        else:
            wrong_count = 0

        self.train_tau_stats['accepted_count'] += accepted_count
        self.train_tau_stats['accepted_wrong_count'] += wrong_count

    def probe_gradient_impact(self, model: nn.Module, x_ulb_w: torch.Tensor, x_ulb_s: torch.Tensor,
                            targets_ulb: torch.Tensor, logits_ulb_w: torch.Tensor,
                            consistency_loss_fn, mask: torch.Tensor):
        """Probe gradient impact of wrong pseudo-labels"""
        if not self.enable or self.grad_probe is None:
            return {}

        stats = self.grad_probe.probe(
            model, x_ulb_w, x_ulb_s, targets_ulb, logits_ulb_w,
            consistency_loss_fn, mask, self.train_tau
        )

        # Add event for accuracy tracking
        if self.acc_tracker and stats:
            event = {
                'global_step': self.global_step,
                'train_tau': self.train_tau,
                'accepted_count': int((torch.softmax(logits_ulb_w, dim=-1).max(dim=-1)[0] >= self.train_tau).sum().item()),
                'accepted_wrong_count': 0,  # Would need to compute
                'accepted_wrong_rate': 0.0,  # Would need to compute
                'L_u_all': stats.get('L_u_all', 0.0),
                'L_u_wrong': stats.get('L_u_wrong', 0.0),
                'grad_norm_u_all': stats.get('grad_norm_u_all', 0.0),
                'grad_norm_u_wrong': stats.get('grad_norm_u_wrong', 0.0),
                'wrong_grad_norm_ratio': stats.get('wrong_grad_norm_ratio', 0.0),
            }
            # Compute accepted_wrong_count
            probs = torch.softmax(logits_ulb_w, dim=-1)
            max_probs, pseudo_labels = probs.max(dim=-1)
            accepted_mask = (max_probs >= self.train_tau) & mask.bool()
            if accepted_mask.any():
                accepted_targets = targets_ulb[accepted_mask]
                accepted_pseudo = pseudo_labels[accepted_mask]
                event['accepted_wrong_count'] = int((accepted_targets != accepted_pseudo).sum().item())
                event['accepted_wrong_rate'] = event['accepted_wrong_count'] / accepted_mask.sum().item() if accepted_mask.sum() > 0 else 0.0
            
            self.acc_tracker.add_event(event)
            self.gradient_events.append(event)

        return stats

    def record_evaluation(self, eval_results: Dict[str, float]):
        """Record evaluation results"""
        if not self.enable or 'downstream_accuracy_impact' not in self.enabled_metrics:
            return

        eval_record = {
            'global_step': self.global_step,
            'eval_top1': eval_results.get('eval/top-1-acc', 0.0),
            'eval_top5': eval_results.get('eval/top-5-acc', 0.0),
            'eval_loss': eval_results.get('eval/loss', 0.0),
        }
        self.eval_history.append(eval_record)
        if self.acc_tracker:
            self.acc_tracker.add_evaluation(eval_record)

    def log_step(self, grad_stats: Dict = None):
        """Log step-level statistics"""
        if not self.enable or not self.is_main_process:
            return

        # Prepare log data
        log_data = {
            'step': self.global_step,
            'epoch': self.epoch,
            'optimizer_name': self.optimizer_name,
            'train_tau': self.train_tau,
        }

        # Add tau stats
        for tau in self.tau_list:
            stats = self.tau_stats[tau]
            total_accepted = stats['accepted_count']
            if total_accepted > 0:
                wrong_rate = stats['accepted_wrong_count'] / total_accepted
                correct_rate = stats['accepted_correct_count'] / total_accepted
            else:
                wrong_rate = 0.0
                correct_rate = 0.0

            log_data[f'tau_{tau}_accepted_count'] = total_accepted
            log_data[f'tau_{tau}_wrong_rate'] = wrong_rate
            log_data[f'tau_{tau}_correct_rate'] = correct_rate

        # Add train tau stats
        train_stats = self.train_tau_stats
        total_accepted = train_stats['accepted_count']
        if total_accepted > 0:
            wrong_rate = train_stats['accepted_wrong_count'] / total_accepted
        else:
            wrong_rate = 0.0
        log_data['train_tau_accepted_count'] = total_accepted
        log_data['train_tau_wrong_rate'] = wrong_rate

        # Add gradient stats
        if grad_stats:
            log_data.update(grad_stats)

        # Write to JSONL
        if self.step_jsonl_file:
            json.dump(log_data, self.step_jsonl_file)
            self.step_jsonl_file.write('\n')
            self.step_jsonl_file.flush()

    def finalize(self):
        """Finalize monitoring and save summaries"""
        if not self.enable or not self.is_main_process:
            return

        # Close files
        if self.step_jsonl_file:
            self.step_jsonl_file.close()

        # Save summaries
        if self.save_summary_csv:
            self._save_summary_csv()

        if self.save_per_tau_csv:
            self._save_per_tau_csv()

        if self.acc_tracker:
            self.acc_tracker.finalize(self.output_dir, self.optimizer_name)

    def _save_summary_csv(self):
        """Save final summary CSV"""
        summary_path = os.path.join(self.output_dir, 'final_summary.csv')

        # Calculate aggregate stats
        total_steps = self.global_step

        # Tau stats
        tau_summary = {}
        for tau in self.tau_list:
            stats = self.tau_stats[tau]
            total_accepted = stats['accepted_count']
            if total_accepted > 0:
                wrong_rate = stats['accepted_wrong_count'] / total_accepted
            else:
                wrong_rate = 0.0
            tau_summary[f'tau_{tau}_wrong_rate'] = wrong_rate

        # Train tau stats
        train_total = self.train_tau_stats['accepted_count']
        train_wrong_rate = (self.train_tau_stats['accepted_wrong_count'] / train_total) if train_total > 0 else 0.0

        # Gradient stats (aggregate from events)
        grad_summary = {}
        if self.gradient_events:
            events = self.gradient_events
            grad_summary['mean_wrong_grad_norm_ratio'] = np.mean([e['wrong_grad_norm_ratio'] for e in events])
            grad_summary['mean_grad_norm_u_wrong'] = np.mean([e['grad_norm_u_wrong'] for e in events])

        # Accuracy correlation (if available)
        acc_summary = {}
        if self.acc_tracker and self.acc_tracker.correlations:
            for horizon, corr in self.acc_tracker.correlations.items():
                acc_summary[f'corr_wrong_strength_vs_acc_delta_h{horizon}'] = corr

        # Final eval
        final_eval = self.eval_history[-1] if self.eval_history else {}
        final_top1 = final_eval.get('eval_top1', 0.0)

        # Write summary
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['optimizer_name', 'mean_train_tau_wrong_rate', 'mean_wrong_grad_norm_ratio',
                           'mean_grad_norm_u_wrong', 'final_eval_top1'] +
                          [f'corr_wrong_strength_vs_acc_delta_h{h}' for h in self.acc_probe_horizon_steps])

            row = [self.optimizer_name, train_wrong_rate,
                   grad_summary.get('mean_wrong_grad_norm_ratio', 0.0),
                   grad_summary.get('mean_grad_norm_u_wrong', 0.0),
                   final_top1]
            for horizon in self.acc_probe_horizon_steps:
                row.append(acc_summary.get(f'corr_wrong_strength_vs_acc_delta_h{horizon}', 0.0))
            writer.writerow(row)

    def _save_per_tau_csv(self):
        """Save per-tau statistics CSV"""
        tau_path = os.path.join(self.output_dir, 'tau_summary.csv')

        with open(tau_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tau', 'accepted_count_total', 'accepted_wrong_count_total',
                           'accepted_correct_count_total', 'accepted_rate_mean',
                           'accepted_wrong_rate_mean', 'accepted_correct_rate_mean'])

            for tau in self.tau_list:
                stats = self.tau_stats[tau]
                total_accepted = stats['accepted_count']
                total_wrong = stats['accepted_wrong_count']
                total_correct = stats['accepted_correct_count']

                # Note: accepted_rate_mean would require per-step tracking, using total for now
                if total_accepted > 0:
                    wrong_rate = total_wrong / total_accepted
                    correct_rate = total_correct / total_accepted
                else:
                    wrong_rate = 0.0
                    correct_rate = 0.0

                writer.writerow([tau, total_accepted, total_wrong, total_correct,
                               0.0, wrong_rate, correct_rate])  # accepted_rate_mean placeholder


class GradientImpactProbe:
    """Probe for measuring gradient impact of wrong pseudo-labels"""

    # def __init__(self, model):
    #     self.model = model

    def probe(self, model: nn.Module, x_ulb_w: torch.Tensor, x_ulb_s: torch.Tensor,
              targets_ulb: torch.Tensor, logits_ulb_w: torch.Tensor,
              consistency_loss_fn, mask: torch.Tensor, train_tau: float) -> Dict[str, float]:
        """Probe gradient impact"""

        batch_size = x_ulb_w.shape[0]

        # Get accepted samples
        probs = torch.softmax(logits_ulb_w, dim=-1)
        max_probs, pseudo_labels = probs.max(dim=-1)
        accepted_mask = (max_probs >= train_tau) & mask.bool()

        if accepted_mask.sum() == 0:
            return {
                'grad_norm_u_all': 0.0,
                'grad_norm_u_correct': 0.0,
                'grad_norm_u_wrong': 0.0,
                'wrong_grad_norm_ratio': 0.0,
                'correct_grad_norm_ratio': 0.0,
                'L_u_all': 0.0,
                'L_u_correct': 0.0,
                'L_u_wrong': 0.0,
                'wrong_loss_fraction': 0.0,
                'correct_loss_fraction': 0.0,
            }

        # Partition into correct and wrong
        accepted_indices = accepted_mask.nonzero(as_tuple=True)[0]
        accepted_targets = targets_ulb[accepted_indices]
        accepted_pseudo = pseudo_labels[accepted_indices]

        correct_mask = (accepted_targets == accepted_pseudo)
        wrong_mask = ~correct_mask

        correct_indices = accepted_indices[correct_mask]
        wrong_indices = accepted_indices[wrong_mask]

        results = {}

        # Compute L_u_all
        with torch.enable_grad():
            model.zero_grad()
            #logits_s = model(x_ulb_s).detach().requires_grad_(True)
            output_dict = model(x_ulb_s)
            logits_s = output_dict['logits'].detach().requires_grad_(True)
            L_u_all = consistency_loss_fn(logits_s, pseudo_labels, 'ce', mask=accepted_mask.float())
            L_u_all.backward()
            grad_norm_all = self._compute_global_grad_norm(model)
            results['grad_norm_u_all'] = grad_norm_all
            results['L_u_all'] = L_u_all.item()

        # Compute L_u_correct
        if correct_indices.numel() > 0:
            with torch.enable_grad():
                model.zero_grad()
                correct_mask_full = torch.zeros(batch_size, dtype=torch.float, device=mask.device)
                correct_mask_full[correct_indices] = 1.0
                L_u_correct = consistency_loss_fn(logits_s, pseudo_labels, 'ce', mask=correct_mask_full)
                # L_u_correct.backward()
                params = [p for p in model.parameters() if p.requires_grad]
                grads = torch.autograd.grad(
                    L_u_correct,
                    params,  # model.parameters(),
                    retain_graph=True,
                    allow_unused=True
                )
                grad_norm_correct = self._compute_global_grad_norm(model)
                results['grad_norm_u_correct'] = grad_norm_correct
                results['L_u_correct'] = L_u_correct.item()
        else:
            results['grad_norm_u_correct'] = 0.0
            results['L_u_correct'] = 0.0

        # Compute L_u_wrong
        if wrong_indices.numel() > 0:
            with torch.enable_grad():
                model.zero_grad()
                wrong_mask_full = torch.zeros(batch_size, dtype=torch.float, device=mask.device)
                wrong_mask_full[wrong_indices] = 1.0
                L_u_wrong = consistency_loss_fn(logits_s, pseudo_labels, 'ce', mask=wrong_mask_full)
                L_u_wrong.backward()
                grad_norm_wrong = self._compute_global_grad_norm(model)
                results['grad_norm_u_wrong'] = grad_norm_wrong
                results['L_u_wrong'] = L_u_wrong.item()
        else:
            results['grad_norm_u_wrong'] = 0.0
            results['L_u_wrong'] = 0.0

        # Compute ratios
        eps = 1e-8
        results['wrong_grad_norm_ratio'] = results['grad_norm_u_wrong'] / max(results['grad_norm_u_all'], eps)
        results['correct_grad_norm_ratio'] = results['grad_norm_u_correct'] / max(results['grad_norm_u_all'], eps)
        results['wrong_loss_fraction'] = results['L_u_wrong'] / max(results['L_u_all'], eps)
        results['correct_loss_fraction'] = results['L_u_correct'] / max(results['L_u_all'], eps)

        return results

    def _compute_global_grad_norm(self, model: nn.Module) -> float:
        """Compute global L2 norm of gradients"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5


class AccuracyImpactTracker:
    """Track accuracy impact of wrong pseudo-label gradient events"""

    def __init__(self, horizon_steps: List[int]):
        self.horizon_steps = horizon_steps
        self.events = []
        self.eval_history = []
        self.correlations = {}

    def add_event(self, event: Dict[str, Any]):
        """Add a gradient event"""
        self.events.append(event)

    def add_evaluation(self, eval_record: Dict[str, Any]):
        """Add evaluation record"""
        self.eval_history.append(eval_record)

    def finalize(self, output_dir: str, optimizer_name: str):
        """Compute final correlations and save results"""
        if not self.events or not self.eval_history:
            return

        # Save gradient events
        events_path = os.path.join(output_dir, 'gradient_events.csv')
        with open(events_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['global_step', 'optimizer_name', 'train_tau', 'accepted_count',
                           'accepted_wrong_count', 'accepted_wrong_rate', 'L_u_all', 'L_u_wrong',
                           'grad_norm_u_all', 'grad_norm_u_wrong', 'wrong_grad_norm_ratio',
                           'last_eval_acc_before_event'])
            for event in self.events:
                writer.writerow([
                    event['global_step'], optimizer_name, event['train_tau'],
                    event['accepted_count'], event['accepted_wrong_count'],
                    event['accepted_wrong_rate'], event['L_u_all'], event['L_u_wrong'],
                    event['grad_norm_u_all'], event['grad_norm_u_wrong'],
                    event['wrong_grad_norm_ratio'], event.get('last_eval_acc_before', 0.0)
                ])

        # Save eval history
        eval_path = os.path.join(output_dir, 'eval_history.csv')
        with open(eval_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['global_step', 'eval_top1', 'eval_top5', 'eval_loss'])
            for eval_record in self.eval_history:
                writer.writerow([
                    eval_record['global_step'], eval_record['eval_top1'],
                    eval_record['eval_top5'], eval_record['eval_loss']
                ])

        # Compute correlations for each horizon
        for horizon in self.horizon_steps:
            self._compute_correlation(horizon)

    def _compute_correlation(self, horizon: int):
        """Compute correlation between wrong event strength and future accuracy delta"""
        event_strengths = []
        acc_deltas = []

        for event in self.events:
            event_step = event['global_step']
            strength = event['wrong_grad_norm_ratio']  # or other strength metric

            # Find future eval at event_step + horizon
            future_eval = None
            for eval_record in self.eval_history:
                if eval_record['global_step'] >= event_step + horizon:
                    future_eval = eval_record
                    break

            if future_eval is None:
                continue

            # Find eval before event
            prev_eval = None
            for eval_record in reversed(self.eval_history):
                if eval_record['global_step'] <= event_step:
                    prev_eval = eval_record
                    break

            if prev_eval is None:
                continue

            acc_delta = future_eval['eval_top1'] - prev_eval['eval_top1']
            event_strengths.append(strength)
            acc_deltas.append(acc_delta)

        if len(event_strengths) >= 2:
            try:
                corr, _ = scipy.stats.pearsonr(event_strengths, acc_deltas)
                self.correlations[horizon] = corr
            except:
                self.correlations[horizon] = 0.0
        else:
            self.correlations[horizon] = 0.0
