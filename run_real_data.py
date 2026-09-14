"""
UrbanTrack AI — Real Observation Evaluation Compatibility Entry Point.

DEPRECATION NOTICE:
This entry point wraps the canonical runner in run_real_member1.py.
Legacy feeds previously located in data/observations/kanishka_traffic.json
have been quarantined under data/legacy/ to prevent accidental execution
against un-embedded mock data.
"""

from run_real_member1 import execute_canonical_member1_pipeline


if __name__ == "__main__":
    print("[NOTICE] Redirecting to canonical Member 1 perception runner (run_real_member1.py)...")
    execute_canonical_member1_pipeline()
