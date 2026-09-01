# Compatibility bridge for HyperPyYAML referencing src.classifier_head
import sys
import os

sb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sb_dir not in sys.path:
    sys.path.insert(0, sb_dir)

from classifier_head import SuperbBaseModel

__all__ = ["SuperbBaseModel"]
