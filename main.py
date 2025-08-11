#!/usr/bin/env python3
"""
Main entry point wrapper for the live camera learning system.
Calls the actual implementation in python/main.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

if __name__ == "__main__":
    from main import main
    main()
