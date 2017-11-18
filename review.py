"""
    Displays the individual frames of the annotated videos along with screens
    and the associated document pages.
"""
import cv2
import numpy as np

from dataset import Dataset

def crop(image, quadrilinear):
    """ Crops out a quadrilinear out of the input image. """
    width_a = np.sqrt(((quadrilinear.bottom_right.x - quadrilinear.bottom_left.x) ** 2)
                      + ((quadrilinear.bottom_right.y - quadrilinear.bottom_left.y) ** 2))
    width_b = np.sqrt(((quadrilinear.top_right.x - quadrilinear.top_left.x) ** 2)
                      + ((quadrilinear.top_right.y - quadrilinear.top_left.y) ** 2))
    height_a = np.sqrt(((quadrilinear.top_right.x - quadrilinear.bottom_right.x) ** 2)
                       + ((quadrilinear.top_right.y - quadrilinear.bottom_right.y) ** 2))
    height_b = np.sqrt(((quadrilinear.top_left.x - quadrilinear.bottom_left.x) ** 2)
                       + ((quadrilinear.top_left.y - quadrilinear.bottom_left.y) ** 2))
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    src = np.array([quadrilinear.top_left, quadrilinear.top_right,
                    quadrilinear.bottom_right, quadrilinear.bottom_left], dtype="float32")
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
    dataset = Dataset(".")
    for video in dataset.videos:
        for frame in video.frames:
            frame_image = cv2.imread(frame.filename)
            assert frame_image is not None
            # If we detected no screens, just show the original frame.
            if not frame.screens:
                cv2.imshow(frame.filename, frame_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # Otherwise, display all screens along with the associated document pages.
            else:
                for screen_num, screen in enumerate(frame.screens):
                    screen_image = crop(frame_image, screen.bounds)
                    assert screen_image is not None
                    cv2.imshow(frame.filename, frame_image)
                    cv2.imshow("screen %d (%s)" % (screen_num, screen.condition), screen_image)
                    for keyref in screen.keyrefs:
                        page_image = cv2.imread(keyref.page.filename)
                        assert page_image is not None
                        page_image = cv2.resize(page_image, (video.width, video.height),
                                                cv2.INTER_CUBIC)
                        cv2.imshow("keyref (%s, %s)" % (keyref.page.filename, keyref.similarity),
                                   page_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
