# demo.py

import argparse
from smart_camera import teach_object, detect_once, reset_model, list_known_objects

def main():
    parser = argparse.ArgumentParser(description="📷 Smart Camera Teaching Demo")
    parser.add_argument('--mode', type=str, choices=['teach', 'detect', 'reset', 'list'], required=True,
                        help="操作模式：teach（教学）、detect（识别）、reset（重置模型）、list（列出已学标签）")
    args = parser.parse_args()

    if args.mode == 'teach':
        teach_object()
    elif args.mode == 'detect':
        detect_once()
    elif args.mode == 'reset':
        reset_model()
    elif args.mode == 'list':
        objects = list_known_objects()
        print("🎓 已学物体：", objects)

if __name__ == '__main__':
    main()

