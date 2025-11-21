from dataclasses import dataclass


@dataclass
class Environment:
    """Хранит переменные, которые используют функции обнаружения"""
    detector = None
    cap = None