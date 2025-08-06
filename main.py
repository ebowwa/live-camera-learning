#!/usr/bin/env python3

import argparse
import sys
from src.rtsp_stream import RTSPStream, RTSPViewer


def main():
    parser = argparse.ArgumentParser(description='RTSP Stream Viewer')
    parser.add_argument(
        '--url',
        type=str,
        default='rtsp://admin:admin@192.168.42.1:554/live',
        help='RTSP stream URL (default: rtsp://admin:admin@192.168.42.1:554/live)'
    )
    parser.add_argument(
        '--window-name',
        type=str,
        default='RTSP Stream',
        help='Window name for display (default: RTSP Stream)'
    )
    
    args = parser.parse_args()
    
    stream = RTSPStream(args.url)
    viewer = RTSPViewer(stream, args.window_name)
    
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()