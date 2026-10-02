"""Numerical schemes for Itô equations."""

from .base import Scheme
from .euler_maruyama import Euler, EulerMaruyama

__all__ = ["Euler", "EulerMaruyama", "Scheme"]
