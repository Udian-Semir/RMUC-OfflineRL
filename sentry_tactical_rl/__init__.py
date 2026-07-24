"""Single-sentry tactical reinforcement-learning demo.

The package intentionally owns only high-level tactical decisions.  A chosen
goal is always validated and executed by a navigation backend.
"""

from .env import SentryTacticalEnv

__all__ = ["SentryTacticalEnv"]
