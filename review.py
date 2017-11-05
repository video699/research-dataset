"""
    Displays the individual frames of the annotated videos along with screens
    and the associated document pages.
"""
from xml.etree import ElementTree as ET, ElementInclude as EI

import cv2
import numpy as np

def crop(image, (top_left, top_right, bottom_left, bottom_right)):
    """ Crops out a quadrilinear out of the input image. """
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2)
                      + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2)
                      + ((top_right[1] - top_left[1]) ** 2))
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2)
                       + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2)
                       + ((top_left[1] - bottom_left[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    src = np.array([top_left, top_right,
                    bottom_right, bottom_left], dtype="float32")
    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, transform, (max_width, max_height))

def main():
    """
        Displays the individual frames of the annotated videos along with screens
        and the associated document pages.
    """
    dataset = ET.parse("dataset.xml")
    root = dataset.getroot()
    EI.include(root)
    for video in root.findall("video"):
        video_dirname = video.attrib["dirname"]
        for frame in video.findall("frames/frame"):
            frame_filename = "%s/%s" % (video_dirname, frame.attrib["filename"])
            frame_image = cv2.imread(frame_filename)
            assert frame_image is not None
            screens = frame.find("screens")
            # If we detected no screens, just show the original frame.
            if screens.find("screen") is None:
                cv2.imshow(frame_filename, frame_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # Otherwise, display all screens along with the associated document pages.
            else:
                for screen_num, screen in enumerate(screens.findall("screen")):
                    condition = screen.attrib["condition"]
                    top_left = (float(screen.attrib["x0"]), float(screen.attrib["y0"]))
                    top_right = (float(screen.attrib["x1"]), float(screen.attrib["y1"]))
                    bottom_left = (float(screen.attrib["x2"]), float(screen.attrib["y2"]))
                    bottom_right = (float(screen.attrib["x3"]), float(screen.attrib["y3"]))
                    screen_image = crop(frame_image, (top_left, top_right,
                                                      bottom_left, bottom_right))
                    cv2.imshow(frame_filename, frame_image)
                    cv2.imshow("screen %d (%s)" % (screen_num, condition), screen_image)
                    for keyref in screen.findall("keyrefs/keyref"):
                        similarity = keyref.attrib["similarity"]
                        key = keyref.text
                        page = video.find(".//page[@key='%s']" % key)
                        page_filename = "%s/%s" % (video_dirname, page.attrib["filename"])
                        page_image = cv2.imread(page_filename)
                        assert page_image is not None
                        page_image = cv2.resize(page_image, (720, 576), cv2.INTER_CUBIC)
                        cv2.imshow("keyref (%s, %s)" % (page_filename, similarity), page_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
