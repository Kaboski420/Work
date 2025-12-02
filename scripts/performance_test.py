#!/usr/bin/env python3
"""
Performance Testing Script

Tests latency and throughput requirements:
- Latency: < 2.0 seconds at 95th percentile (or ≤ 5 minutes E2E)
- Throughput: ≥ 100 scoring operations per second (or 100 requests/minute)
"""

import asyncio
import time
import statistics
import httpx
import json
from typing import List, Dict, Any
from datetime import datetime
import argparse

API_BASE_URL = "http://localhost:8000"
ENDPOINTS = {
    "health": "/health",
    "score": "/api/v1/score",
    "ingest": "/api/v1/ingest",
    "analyze": "/api/v1/analyze"
}


class PerformanceTest:
    """Performance testing suite."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.results = {
            "latency": [],
            "throughput": [],
            "errors": []
        }
    
    async def test_latency(
        self,
        endpoint: str,
        num_requests: int = 100,
        concurrent: int = 10
    ) -> Dict[str, Any]:
        """Test endpoint latency."""
        print(f"\n📊 Testing latency for {endpoint}")
        print(f"   Requests: {num_requests}, Concurrent: {concurrent}")
        
        latencies = []
        errors = []
        test_data = self._prepare_test_data(endpoint)
        
        async def make_request(client: httpx.AsyncClient, data: Dict[str, Any]) -> float:
            """Make a single request and measure latency."""
            start_time = time.time()
            try:
                if endpoint == "/api/v1/score":
                    response = await client.post(
                        f"{self.base_url}{endpoint}",
                        json=data,
                        timeout=30.0
                    )
                elif endpoint == "/api/v1/ingest":
                    files = {"file": ("test.mp4", b"fake video content", "video/mp4")}
                    form_data = {
                        "platform": data.get("platform", "tiktok"),
                        "caption": data.get("caption", ""),
                    }
                    response = await client.post(
                        f"{self.base_url}{endpoint}",
                        files=files,
                        data=form_data,
                        timeout=60.0
                    )
                else:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        timeout=10.0
                    )
                
                response.raise_for_status()
                return time.time() - start_time
            except httpx.ConnectError as e:
                error_msg = f"Connection error: API not accessible at {self.base_url}. Is the API running?"
                errors.append(error_msg)
                return None
            except Exception as e:
                errors.append(f"{type(e).__name__}: {str(e)}")
                return None
        
        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(concurrent)
            
            async def bounded_request(data):
                async with semaphore:
                    return await make_request(client, test_data)
            
            tasks = [bounded_request(test_data) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        latencies = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        if not latencies:
            error_msg = "All requests failed"
            if errors:
                first_error = errors[0]
                if "Connection error" in first_error:
                    error_msg = f"API not accessible at {self.base_url}. Please ensure the API is running."
            return {
                "endpoint": endpoint,
                "success": False,
                "error": error_msg,
                "errors": errors[:5]
            }
        
        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
        
        result = {
            "endpoint": endpoint,
            "success": True,
            "total_requests": num_requests,
            "successful_requests": len(latencies),
            "failed_requests": num_requests - len(latencies),
            "latency": {
                "min": min(latencies),
                "max": max(latencies),
                "mean": statistics.mean(latencies),
                "median": p50,
                "p95": p95,
                "p99": p99,
                "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "errors": errors[:5] if errors else None
        }
        
        if endpoint in ["/api/v1/score", "/api/v1/ingest", "/api/v1/analyze"]:
            requirement_met = p95 <= 300.0
            result["requirement"] = "≤ 5 minutes E2E"
            result["requirement_met"] = requirement_met
        else:
            requirement_met = p95 <= 2.0
            result["requirement"] = "< 2.0 seconds (P95)"
            result["requirement_met"] = requirement_met
        
        return result
    
    async def test_throughput(
        self,
        endpoint: str,
        duration_seconds: int = 60,
        target_rps: int = 100
    ) -> Dict[str, Any]:
        """Test endpoint throughput."""
        print(f"\n⚡ Testing throughput for {endpoint}")
        print(f"   Duration: {duration_seconds}s, Target: {target_rps} req/s")
        
        requests_sent = 0
        requests_successful = 0
        errors = []
        test_data = self._prepare_test_data(endpoint)
        
        async def make_request(client: httpx.AsyncClient, data: Dict[str, Any]) -> bool:
            """Make a single request."""
            try:
                if endpoint == "/api/v1/score":
                    response = await client.post(
                        f"{self.base_url}{endpoint}",
                        json=data,
                        timeout=30.0
                    )
                elif endpoint == "/api/v1/ingest":
                    files = {"file": ("test.mp4", b"fake video content", "video/mp4")}
                    form_data = {
                        "platform": data.get("platform", "tiktok"),
                        "caption": data.get("caption", ""),
                    }
                    response = await client.post(
                        f"{self.base_url}{endpoint}",
                        files=files,
                        data=form_data,
                        timeout=60.0
                    )
                else:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        timeout=10.0
                    )
                
                response.raise_for_status()
                return True
            except httpx.ConnectError as e:
                error_msg = f"Connection error: API not accessible at {self.base_url}"
                errors.append(error_msg)
                return False
            except Exception as e:
                errors.append(f"{type(e).__name__}: {str(e)}")
                return False
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        async with httpx.AsyncClient() as client:
            while time.time() < end_time:
                tasks = []
                for _ in range(target_rps):
                    tasks.append(make_request(client, test_data))
                    requests_sent += 1
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                requests_successful += sum(1 for r in results if r is True)
                await asyncio.sleep(0.1)
        
        actual_duration = time.time() - start_time
        actual_rps = requests_successful / actual_duration if actual_duration > 0 else 0
        target_rpm = target_rps * 60
        actual_rpm = requests_successful / (actual_duration / 60) if actual_duration > 0 else 0
        
        result = {
            "endpoint": endpoint,
            "success": True,
            "duration_seconds": actual_duration,
            "requests_sent": requests_sent,
            "requests_successful": requests_successful,
            "requests_failed": requests_sent - requests_successful,
            "throughput": {
                "rps": actual_rps,
                "rpm": actual_rpm,
                "target_rps": target_rps,
                "target_rpm": target_rpm
            },
            "requirement": "≥ 100 requests/minute (Sprint 25)",
            "requirement_met": actual_rpm >= 100,
            "errors": errors[:5] if errors else None
        }
        
        return result
    
    def _prepare_test_data(self, endpoint: str) -> Dict[str, Any]:
        """Prepare test data for endpoint."""
        if endpoint == "/api/v1/score":
            return {
                "content_id": "test-content-123",
                "features": {
                    "visual": {"entropy": 0.7},
                    "audio": {"bpm": 120.0},
                    "text": {"trend_score": 0.6}
                },
                "embeddings": {
                    "visual": [0.1] * 512,
                    "audio": [0.2] * 128,
                    "text": [0.3] * 384
                },
                "metadata": {
                    "platform": "tiktok",
                    "engagement_metrics": {
                        "views": 1000.0,
                        "likes": 100.0,
                        "shares": 50.0,
                        "comments": 25.0
                    }
                }
            }
        elif endpoint == "/api/v1/ingest":
            return {
                "platform": "tiktok",
                "caption": "Test content for performance testing"
            }
        else:
            return {}
    
    async def run_full_test(self) -> Dict[str, Any]:
        """Run full performance test suite."""
        print("🚀 Starting Performance Test Suite")
        print("=" * 60)
        
        # Check if API is accessible
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    print(f"✅ API is accessible at {self.base_url}")
                else:
                    print(f"⚠️  API responded with status {response.status_code}")
        except Exception as e:
            print(f"❌ ERROR: Cannot connect to API at {self.base_url}")
            print(f"   Please ensure the API is running:")
            print(f"   - Docker: docker-compose -f docker-compose.test.yml up -d")
            print(f"   - Local: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
            print(f"   Error: {str(e)}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "base_url": self.base_url,
                "error": "API not accessible",
                "tests": {}
            }
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "base_url": self.base_url,
            "tests": {}
        }
        
        results["tests"]["health_latency"] = await self.test_latency(
            "/health",
            num_requests=100,
            concurrent=10
        )
        
        results["tests"]["score_latency"] = await self.test_latency(
            "/api/v1/score",
            num_requests=50,
            concurrent=5
        )
        
        results["tests"]["score_throughput"] = await self.test_throughput(
            "/api/v1/score",
            duration_seconds=60,
            target_rps=2
        )
        
        results["summary"] = {
            "total_tests": len(results["tests"]),
            "passed": sum(1 for t in results["tests"].values() if t.get("requirement_met", False)),
            "failed": sum(1 for t in results["tests"].values() if not t.get("requirement_met", True))
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TEST RESULTS")
        print("=" * 60)
        
        for test_name, test_result in results["tests"].items():
            print(f"\n{test_name.upper()}:")
            if not test_result.get("success"):
                print(f"  ❌ FAILED: {test_result.get('error', 'Unknown error')}")
                continue
            
            if "latency" in test_result:
                latency = test_result["latency"]
                print(f"  Latency (P95): {latency['p95']:.3f}s")
                print(f"  Latency (P99): {latency['p99']:.3f}s")
                print(f"  Mean: {latency['mean']:.3f}s")
                print(f"  Requirement: {test_result.get('requirement', 'N/A')}")
                print(f"  Status: {'✅ PASS' if test_result.get('requirement_met') else '❌ FAIL'}")
            
            if "throughput" in test_result:
                throughput = test_result["throughput"]
                print(f"  Throughput: {throughput['rpm']:.1f} requests/minute")
                print(f"  Target: {throughput['target_rpm']:.1f} requests/minute")
                print(f"  Requirement: {test_result.get('requirement', 'N/A')}")
                print(f"  Status: {'✅ PASS' if test_result.get('requirement_met') else '❌ FAIL'}")
        
        print("\n" + "=" * 60)
        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print("=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Performance testing for Virality Engine")
    parser.add_argument(
        "--url",
        default=API_BASE_URL,
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output",
        help="Output file for JSON results"
    )
    
    args = parser.parse_args()
    
    tester = PerformanceTest(base_url=args.url)
    results = await tester.run_full_test()
    
    tester.print_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
