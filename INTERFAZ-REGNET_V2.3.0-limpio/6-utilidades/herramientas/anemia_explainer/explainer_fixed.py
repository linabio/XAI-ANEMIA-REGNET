import numpy as np
import torch
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import time
import psutil
import os
import gc
from contextlib import contextmanager
from .model import BaseModel, RegNetBinaryClassifier
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class ExplanationError(Exception):
    pass

@dataclass
class PerformanceMetrics:
    """Dataclass to store performance metrics."""
    processing_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    batch_size: Optional[int] = None
    image_size: Optional[Tuple[int, int]] = None
    num_samples: Optional[int] = None
    num_features: Optional[int] = None

@dataclass
class ExplanationConfig:
    num_samples: int = 1000
    num_features: int = 8
    hide_color: int = 0
    positive_only: bool = True
    hide_rest: bool = False
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 300
    save_bbox: str = 'tight'
    batch_size: int = 32
    max_workers: int = 4
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True

class PerformanceMonitor:
    """Monitor for tracking performance metrics during LIME explanations."""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.process = psutil.Process(os.getpid())
        self._initial_memory = None
        self._peak_memory = 0
        
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except:
            return 0.0
    
    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Get GPU memory usage in MB if available."""
        if not self.enable_gpu_monitoring:
            return None
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return None
    
    def start_monitoring(self):
        """Start monitoring and record initial state."""
        self._initial_memory = self._get_memory_usage_mb()
        self._peak_memory = self._initial_memory
        self._start_time = time.time()
        self._start_cpu = self._get_cpu_usage_percent()
        
        # Force garbage collection before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self._get_memory_usage_mb()
        if current_memory > self._peak_memory:
            self._peak_memory = current_memory
    
    def get_metrics(self, **kwargs) -> PerformanceMetrics:
        """Get current performance metrics."""
        self.update_peak_memory()
        
        end_time = time.time()
        processing_time = end_time - self._start_time
        
        current_memory = self._get_memory_usage_mb()
        memory_usage = current_memory - self._initial_memory if self._initial_memory else current_memory
        
        # Calculate average CPU usage (simplified)
        current_cpu = self._get_cpu_usage_percent()
        cpu_usage = (self._start_cpu + current_cpu) / 2
        
        gpu_memory = self._get_gpu_memory_mb()
        
        return PerformanceMetrics(
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_memory_mb=gpu_memory,
            peak_memory_mb=self._peak_memory,
            **kwargs
        )
    
    def log_metrics(self, metrics: PerformanceMetrics, operation: str = "LIME Explanation"):
        """Log performance metrics."""
        logger.info(f"=== {operation} Performance Metrics ===")
        logger.info(f"Processing Time: {metrics.processing_time:.3f} seconds")
        logger.info(f"Memory Usage: {metrics.memory_usage_mb:.2f} MB")
        logger.info(f"Peak Memory: {metrics.peak_memory_mb:.2f} MB")
        logger.info(f"CPU Usage: {metrics.cpu_usage_percent:.2f}%")
        if metrics.gpu_memory_mb is not None:
            logger.info(f"GPU Memory: {metrics.gpu_memory_mb:.2f} MB")
        if metrics.batch_size:
            logger.info(f"Batch Size: {metrics.batch_size}")
        if metrics.image_size:
            logger.info(f"Image Size: {metrics.image_size}")
        if metrics.num_samples:
            logger.info(f"LIME Samples: {metrics.num_samples}")
        if metrics.num_features:
            logger.info(f"LIME Features: {metrics.num_features}")
        logger.info("=" * 40)

@contextmanager
def performance_monitor(monitor: PerformanceMonitor, operation: str = "Operation"):
    """Context manager for performance monitoring."""
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        metrics = monitor.get_metrics()
        monitor.log_metrics(metrics, operation)

class LimeExplainer:
    def __init__(self,
                 model: BaseModel,
                 device: str = 'cpu',
                 config: Optional[ExplanationConfig] = None,
                 class_names: Optional[List[str]] = None):
        """Initialize the LimeExplainer with a model and configuration."""
        self.model = model
        self.device = device
        self.config = config or ExplanationConfig()
        self.class_names = class_names or ['Negative', 'Positive']
        self.processor = ImageProcessor()
        self.explainer = lime_image.LimeImageExplainer()
        self.model.eval()
        self._prediction_cache = {}
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(enable_gpu_monitoring='cuda' in device)
        self.performance_history = []
        
        logger.info("LimeExplainer initialized")

    def _batch_predict(self, images: np.ndarray) -> np.ndarray:
        """Predict probabilities for a batch of images."""
        try:
            with performance_monitor(self.performance_monitor, "Batch Prediction") as monitor:
                batch_tensors = [self.processor.preprocess_for_model(img) for img in images]
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.inference_mode():
                    probs = self.model.predict_proba(batch).cpu().numpy()
                
                del batch, batch_tensors
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()
                
                # Record metrics
                metrics = monitor.get_metrics(batch_size=len(images))
                if self.config.log_performance_metrics:
                    monitor.log_metrics(metrics, "Batch Prediction")
                self.performance_history.append(metrics)
                
                return probs  # Shape: (batch_size, num_classes)
        except Exception as e:
            logger.exception("Batch predict error")
            raise ExplanationError("Prediction failed") from e

    def _cached_predict(self, images: np.ndarray, image_ids: Optional[List[str]] = None) -> np.ndarray:
        """Predict with caching using image IDs or MD5 hashes."""
        if image_ids is None:
            hashes = [hashlib.md5(img.tobytes()).hexdigest() for img in images]
        else:
            hashes = image_ids

        results = []
        to_compute = []

        for i, h in enumerate(hashes):
            if h in self._prediction_cache:
                results.append((i, self._prediction_cache[h]))
            else:
                to_compute.append(i)

        if to_compute:
            images_to_compute = np.array([images[i] for i in to_compute])
            computed_probs = self._batch_predict(images_to_compute)
            for idx, prob in zip(to_compute, computed_probs):
                h = hashes[idx]
                self._prediction_cache[h] = prob
                results.append((idx, prob))

        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])

    def explain_image(self,
                      image: Union[str, np.ndarray],
                      save_path: Optional[str] = None,
                      show_plot: bool = True,
                      return_explanation: bool = False,
                      num_samples: Optional[int] = None,
                      num_features: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Explain an image using LIME with customizable parameters."""
        try:
            num_samples = num_samples or self.config.num_samples
            num_features = num_features or self.config.num_features

            if isinstance(image, str):
                image_path = image
                image = self.processor.load_image(image_path)
            else:
                image_path = None

            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            if image_hash in self._prediction_cache:
                logger.debug("Using cached explanation")
                return self._prediction_cache[image_hash + '_explanation']

            logger.info("Generating LIME explanation...")
            image_uint8 = (image / 255).astype(np.uint8) if image.max() > 1 else image

            # Monitor performance during LIME explanation
            with performance_monitor(self.performance_monitor, "LIME Explanation") as monitor:
                explanation = self.explainer.explain_instance(
                    image=image_uint8,
                    classifier_fn=self._cached_predict,
                    top_labels=1,  # Fixed: use integer instead of model.num_classes
                    hide_color=self.config.hide_color,
                    num_samples=num_samples
                )

                # Get the top label safely
                top_label = getattr(explanation, 'top_labels', [0])[0] if hasattr(explanation, 'top_labels') else 0
                
                temp, mask = explanation.get_image_and_mask(
                    top_label,
                    positive_only=False,
                    num_features=num_features,
                    hide_rest=self.config.hide_rest
                )

                if show_plot or save_path:
                    self._visualize_explanation(temp, mask, explanation, save_path, show_plot)

                # Get performance metrics
                metrics = monitor.get_metrics(
                    image_size=image.shape[:2],
                    num_samples=num_samples,
                    num_features=num_features
                )
                
                if self.config.log_performance_metrics:
                    monitor.log_metrics(metrics, "LIME Explanation")
                self.performance_history.append(metrics)

                # Get the top label safely for result
                predicted_class = getattr(explanation, 'top_labels', [0])[0] if hasattr(explanation, 'top_labels') else 0

            result = {
                'explanation': explanation,
                'image': temp,
                'mask': mask,
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'image_path': image_path,
                'performance_metrics': metrics
            }

            self._prediction_cache[image_hash + '_explanation'] = result
            return result if return_explanation else None

        except Exception as e:
            logger.error(f"Explanation error: {e}")
            image_array = self.processor.load_image(image) if isinstance(image, str) else image
            return self._generate_fallback_explanation(image_array, str(e), image if isinstance(image, str) else None)

    def _generate_fallback_explanation(self,
                                       image_array: np.ndarray,
                                       error_msg: str,
                                       image_path: Optional[str] = None) -> dict:
        """Generate a fallback explanation with detailed error info."""
        logger.warning(f"Fallback triggered: {error_msg}")
        try:
            if image_array is None:
                image_array = np.zeros((*self.processor.target_size, 3), dtype=np.float32)
            return {
                'error': error_msg,
                'fallback': True,
                'image': image_array,
                'mask': np.zeros(image_array.shape[:2], dtype=bool),
                'image_path': image_path,
                'partial_analysis': 'Unable to generate full explanation due to error'
            }
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return {
                'error': f"{error_msg} (additional fallback error: {str(e)})",
                'fallback': True,
                'image_path': image_path
            }

    def _visualize_explanation(self,
                               temp: np.ndarray,
                               mask: np.ndarray,
                               explanation,
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
        """Visualize explanation with colored masks."""
        try:
            plt.figure(figsize=self.config.figure_size)
            image_uint8 = (temp / 255).astype(np.uint8) if temp.max() > 1 else temp
            plt.imshow(image_uint8)

            # Get the top label safely for visualization
            top_label = getattr(explanation, 'top_labels', [0])[0] if hasattr(explanation, 'top_labels') else 0

            mask_pos = explanation.get_image_and_mask(
                top_label,
                positive_only=True,
                num_features=self.config.num_features,
                hide_rest=False
            )[1]
            mask_neg = explanation.get_image_and_mask(
                top_label,
                positive_only=False,
                num_features=self.config.num_features,
                hide_rest=False
            )[1]
            mask_neg_only = (mask_neg & ~mask_pos).astype(bool)

            plt.imshow(mask_pos, cmap='Reds', alpha=0.4)
            plt.imshow(mask_neg_only, cmap='Blues', alpha=0.4)

            predicted_class = getattr(explanation, 'top_labels', [0])[0] if hasattr(explanation, 'top_labels') else 0
            class_name = self.class_names[predicted_class]
            plt.title(f"LIME - Predicted: {class_name}", fontsize=14, fontweight='bold')
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, bbox_inches=self.config.save_bbox, dpi=self.config.dpi)
                logger.info(f"Explanation saved: {save_path}")

            if show_plot:
                plt.show()
        finally:
            plt.close('all')

    def explain_batch(self,
                      images: List[Union[str, np.ndarray]],
                      save_dir: Optional[str] = None,
                      show_plots: bool = False) -> List[Dict[str, Any]]:
        """Explain a batch of images in parallel."""
        results = []
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._safe_explain_image, img, save_dir, show_plots, i)
                for i, img in enumerate(images)
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Parallel task error: {e}")
                    results.append({'error': str(e), 'fallback': True})
        return results

    def _safe_explain_image(self, image, save_dir, show_plots, idx):
        """Safely explain a single image for batch processing."""
        save_path = Path(save_dir) / f"explanation_{idx:04d}.png" if save_dir else None
        return self.explain_image(
            image=image,
            save_path=str(save_path) if save_path else None,
            show_plot=show_plots,
            return_explanation=True
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of performance metrics."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Calculate statistics
        processing_times = [m.processing_time for m in self.performance_history]
        memory_usages = [m.memory_usage_mb for m in self.performance_history]
        cpu_usages = [m.cpu_usage_percent for m in self.performance_history]
        gpu_memories = [m.gpu_memory_mb for m in self.performance_history if m.gpu_memory_mb is not None]
        
        summary = {
            "total_operations": len(self.performance_history),
            "processing_time": {
                "mean": np.mean(processing_times),
                "std": np.std(processing_times),
                "min": np.min(processing_times),
                "max": np.max(processing_times),
                "total": np.sum(processing_times)
            },
            "memory_usage": {
                "mean_mb": np.mean(memory_usages),
                "std_mb": np.std(memory_usages),
                "min_mb": np.min(memory_usages),
                "max_mb": np.max(memory_usages),
                "peak_mb": max([m.peak_memory_mb for m in self.performance_history])
            },
            "cpu_usage": {
                "mean_percent": np.mean(cpu_usages),
                "std_percent": np.std(cpu_usages),
                "min_percent": np.min(cpu_usages),
                "max_percent": np.max(cpu_usages)
            }
        }
        
        if gpu_memories:
            summary["gpu_memory"] = {
                "mean_mb": np.mean(gpu_memories),
                "std_mb": np.std(gpu_memories),
                "min_mb": np.min(gpu_memories),
                "max_mb": np.max(gpu_memories)
            }
        
        # Add operation breakdown
        operation_counts = {}
        for metrics in self.performance_history:
            # Use a default operation type since we don't track this specifically
            op_type = 'lime_operation'
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        summary["operation_breakdown"] = operation_counts
        
        return summary

    def export_performance_report(self, output_path: str) -> None:
        """Export detailed performance metrics to a JSON file."""
        import json
        from datetime import datetime
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "config": {
                "num_samples": self.config.num_samples,
                "num_features": self.config.num_features,
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers
            },
            "summary": self.get_performance_summary(),
            "detailed_metrics": [
                {
                    "processing_time": m.processing_time,
                    "memory_usage_mb": m.memory_usage_mb,
                    "cpu_usage_percent": m.cpu_usage_percent,
                    "gpu_memory_mb": m.gpu_memory_mb,
                    "peak_memory_mb": m.peak_memory_mb,
                    "batch_size": m.batch_size,
                    "image_size": m.image_size,
                    "num_samples": m.num_samples,
                    "num_features": m.num_features
                }
                for m in self.performance_history
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Performance report exported to: {output_path}")

    def clear_performance_history(self) -> None:
        """Clear the performance history."""
        self.performance_history.clear()
        logger.info("Performance history cleared")

    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        return self.performance_history[-1] if self.performance_history else None

def create_anemia_explainer(weights_path: str,
                            device: Optional[str] = None,
                            num_samples: int = 1000,
                            num_features: int = 8):
    """Create an anemia explainer wrapper."""
    config = ExplanationConfig(num_samples=num_samples, num_features=num_features)
    model = RegNetBinaryClassifier(weights_path=weights_path, device=device or 'cpu')
    lime_explainer = LimeExplainer(model=model, device=device or 'cpu', config=config)

    class AnemiaLimeExplainerWrapper:
        def __init__(self, lime_explainer):
            self.explainer = lime_explainer

        def explain_single_image(self, image_path, save_path=None, show_plot=True):
            result = self.explainer.explain_image(
                image=image_path,
                save_path=save_path,
                show_plot=show_plot,
                return_explanation=True
            )
            # Asegurar que devolvemos un diccionario con la información estructurada
            if result is None:
                return {
                    'error': 'No se pudo generar explicación',
                    'class_name': 'Error',
                    'explanation': None
                }
            return result

        def explain_multiple_images(self, image_paths, save_dir=None):
            return self.explainer.explain_batch(
                images=image_paths,
                save_dir=save_dir,
                show_plots=False
            )

        def cleanup(self):
            pass

        def get_performance_summary(self) -> Dict[str, Any]:
            """Get a summary of all performance metrics."""
            return self.explainer.get_performance_summary()

        def export_performance_report(self, output_path: str) -> None:
            """Export performance metrics to a JSON file."""
            return self.explainer.export_performance_report(output_path)

        def get_latest_metrics(self):
            """Get the most recent performance metrics."""
            return self.explainer.get_latest_metrics()

        def clear_performance_history(self) -> None:
            """Clear the performance history."""
            return self.explainer.clear_performance_history()

    return AnemiaLimeExplainerWrapper(lime_explainer) 