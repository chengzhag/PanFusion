import sys
import os
import argparse

from external import PanoAnnotator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    labelPath = args.i
    outputPath = 'debug' if args.debug else os.path.dirname(args.i)

    scene = PanoAnnotator.data.Scene(None)
    scene.initEmptyScene()

    PanoAnnotator.utils.loadLabelByJson(labelPath, scene)
    PanoAnnotator.utils.saveSceneAsMaps(outputPath, scene)
