#!/usr/bin/env python3
"""
EdaxShifu - The Complete Intelligent Camera System
One command, one interface, everything you need.
"""

import sys
import os

print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║                    🎯 EdaxShifu                          ║
║         Intelligent Camera Learning System                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

ONE interface with everything:
✓ Live RTSP/webcam stream with YOLO detection
✓ Capture objects with one click
✓ Annotate unknown objects
✓ Teach new objects by example
✓ Real-time learning and improvement

Starting unified interface...
""")

# Import and run the unified interface
from unified_interface import main

if __name__ == "__main__":
    main()