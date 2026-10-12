# -*- coding: utf-8 -*-
"""Tiny helper for aligning terminal tables that mix Chinese and ASCII text."""
from __future__ import annotations

import unicodedata


def display_width(text):
    """Display width in terminal columns (CJK characters occupy two columns)."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, width):
    """Left-align `text` inside a field of `width` display columns."""
    return text + " " * max(width - display_width(text), 0)
