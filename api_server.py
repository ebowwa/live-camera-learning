#!/usr/bin/env python3
"""
API server entry point wrapper.
Calls the actual implementation in python/api_server.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

if __name__ == "__main__":
    from api_server import main
    main()
