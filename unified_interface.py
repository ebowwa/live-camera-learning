#!/usr/bin/env python3
"""
Unified interface entry point wrapper.
Calls the actual implementation in python/unified_interface.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

if __name__ == "__main__":
    from unified_interface import main
    main()
