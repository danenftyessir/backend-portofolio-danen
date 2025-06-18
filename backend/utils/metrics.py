import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """data point untuk metric"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """collector untuk aplikasi metrics"""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
    def record_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """record counter metric"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        
        # juga simpan sebagai time series
        self.metrics[key].append(MetricPoint(
            timestamp=datetime.utcnow(),
            value=self.counters[key],
            labels=labels or {}
        ))
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """record gauge metric"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        self.metrics[key].append(MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        ))
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """record histogram metric"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        
        # keep only last 1000 values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        self.metrics[key].append(MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        ))
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """create key dari name dan labels"""
        if not labels:
            return name
        
        label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{name}{{{label_str}}}"
    
    def get_metric_summary(self, name: str, labels: Dict[str, str] = None) -> Dict[str, Any]:
        """get summary untuk metric"""
        key = self._make_key(name, labels)
        
        if key in self.counters:
            return {
                "type": "counter",
                "current_value": self.counters[key],
                "total_points": len(self.metrics[key])
            }
        
        if key in self.gauges:
            return {
                "type": "gauge", 
                "current_value": self.gauges[key],
                "total_points": len(self.metrics[key])
            }
        
        if key in self.histograms:
            values = self.histograms[key]
            if values:
                return {
                    "type": "histogram",
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": self._percentile(values, 50),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99)
                }
        
        return {"type": "unknown"}
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """get all metrics summary"""
        result = {}
        
        # counters
        for key in self.counters:
            result[key] = self.get_metric_summary(key.split("{")[0])
        
        # gauges
        for key in self.gauges:
            if key not in result:
                result[key] = self.get_metric_summary(key.split("{")[0])
        
        # histograms
        for key in self.histograms:
            if key not in result:
                result[key] = self.get_metric_summary(key.split("{")[0])
        
        return result
    
    def get_time_series(
        self, 
        name: str, 
        labels: Dict[str, str] = None,
        since: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """get time series data"""
        key = self._make_key(name, labels)
        points = list(self.metrics[key])
        
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        return points
    
    def clear_old_metrics(self, older_than: timedelta):
        """clear metrics older than specified time"""
        cutoff = datetime.utcnow() - older_than
        
        for key in self.metrics:
            points = self.metrics[key]
            # remove old points
            while points and points[0].timestamp < cutoff:
                points.popleft()

# global metrics collector instance
_metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """get global metrics collector"""
    return _metrics_collector

class TimingContext:
    """context manager untuk timing operations"""
    
    def __init__(self, metric_name: str, labels: Dict[str, str] = None):
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
        self.collector = get_metrics_collector()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_histogram(
                self.metric_name,
                duration_ms,
                self.labels
            )

def time_operation(metric_name: str, labels: Dict[str, str] = None):
    """decorator untuk timing function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(metric_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def record_request_metrics(
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float
):
    """record metrics untuk http request"""
    collector = get_metrics_collector()
    
    labels = {
        "endpoint": endpoint,
        "method": method,
        "status": str(status_code)
    }
    
    # counter untuk total requests
    collector.record_counter("http_requests_total", 1, labels)
    
    # histogram untuk response time
    collector.record_histogram("http_request_duration_ms", response_time_ms, labels)
    
    # counter untuk error rates
    if status_code >= 400:
        collector.record_counter("http_errors_total", 1, labels)

def record_ai_metrics(
    model: str,
    tokens_used: int,
    response_time_ms: float,
    success: bool
):
    """record metrics untuk ai operations"""
    collector = get_metrics_collector()
    
    labels = {
        "model": model,
        "success": str(success)
    }
    
    collector.record_counter("ai_requests_total", 1, labels)
    collector.record_histogram("ai_response_time_ms", response_time_ms, labels)
    collector.record_histogram("ai_tokens_used", tokens_used, labels)

def record_rag_metrics(
    query: str,
    documents_retrieved: int,
    retrieval_time_ms: float,
    similarity_scores: List[float]
):
    """record metrics untuk rag operations"""
    collector = get_metrics_collector()
    
    collector.record_histogram("rag_retrieval_time_ms", retrieval_time_ms)
    collector.record_gauge("rag_documents_retrieved", documents_retrieved)
    
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        collector.record_gauge("rag_avg_similarity", avg_similarity)

def get_system_metrics() -> Dict[str, Any]:
    """get system-level metrics"""
    try:
        import psutil
        
        # memory usage
        memory = psutil.virtual_memory()
        
        # cpu usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # disk usage
        disk = psutil.disk_usage('/')
        
        return {
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        }
        
    except ImportError:
        logger.warning("psutil not available, cannot collect system metrics")
        return {}
    except Exception as e:
        logger.error(f"error collecting system metrics: {e}")
        return {}