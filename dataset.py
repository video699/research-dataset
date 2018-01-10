"""
    This module defines an object model of the dataset.
"""
from collections import namedtuple
import json
import logging
import random

from lxml import etree
from lxml.etree import XML, XMLSchema, XMLParser
from numpy import array

DATASET_FILENAME = "dataset.xml"
SCHEMA_FILENAME = "schema.xsd"
LOGGER = logging.getLogger(__name__)
FOLDS_NUM = 17
RANDOM_STATE = 12345

class Dataset(object):
    """ This class represents the entire dataset. """
    def __init__(self, dirname):
        """Constructs an object representation of the entire dataset.

        Parameters:
            dirname The name of the directory where the dataset is located.
        """
        self.dirname = dirname
        self.filename = "%s/%s" % (self.dirname, DATASET_FILENAME)

        LOGGER.info("Validating the dataset ...")
        tree = etree.parse(self.filename)
        tree.xinclude()
        with open("%s/%s" % (self.dirname, SCHEMA_FILENAME), "rb") as f:
            schema = XMLSchema(file=f)
        xmlparser = XMLParser(schema=schema, huge_tree=True)
        element = etree.fromstring(etree.tostring(tree), xmlparser)
        LOGGER.info("Done validating the dataset.")

        LOGGER.info("Processing the dataset ...")
        self.videos = []
        self.documents = []
        self.pages = []
        self.frames = []
        self.screens = []
        self.keyrefs = []
        for descendant in element.findall(".//video"):
            video = Video(self, descendant)
            self.videos.append(video)
            self.documents.extend(video.documents)
            self.pages.extend(video.pages)
            self.frames.extend(video.frames)
            self.screens.extend(video.screens)
            self.keyrefs.extend(video.keyrefs)
        LOGGER.info("Done processing the dataset, which contains:")
        LOGGER.info("- %d videos containing %d frames with %d screens (%d non-matched)" + \
                    " and %d keyrefs, and", len(self.videos), len(self.frames), len(self.screens), \
                    len([screen for screen in self.screens if not screen.matching_pages]), \
                    len(self.keyrefs))
        LOGGER.info("- %d documents containing %d pages.", len(self.documents), len(self.pages))

    def task1_evaluation_dataset(self, k_folds=FOLDS_NUM):
        """Produces an evaluation dataset for task1, subtask A (screen-based document page
        retrieval) and subtask B (no-match screen detection). The method returns a list of videos
        with a number of elements divisible by the provided integer k (parameters `k_folds`),
        allowing for k-fold cross-validation.

        Parameters:
            k_folds         The number of folds that will be produced from the dataset.
        """
        sample = self.videos[:]
        random.seed(RANDOM_STATE)
        random.shuffle(sample)
        return array(sample[:len(sample)-len(sample)%k_folds])

    def __repr__(self):
        return "Dataset %s" % self.filename

class Video(object):
    """ This class represents a single video. """
    def __init__(self, dataset, element):
        """Constructs the object representation of a single video.

        Parameters:
            dataset The dataset to which the video belongs.
            element The XML element that represents the video.
        """
        self.dataset = dataset

        # Set own attributes.
        self.dirname = "%s/%s" % (dataset.dirname, element.attrib["dirname"])
        self.fps = int(element.attrib["fps"])
        self.frames_num = int(element.attrib["frames"])
        self.width = int(element.attrib["width"])
        self.height = int(element.attrib["height"])
        self.uri = element.attrib["uri"]
        self.documents = []

        # Process descendant elements.
        ## Process documents.
        for document in element.findall(".//document"):
            self.documents.append(Document(dataset, self, document))
        self.pages = []
        self.page_dict = {}
        for document in self.documents:
            self.pages.extend(document.pages)
            for page in document.pages:
                self.page_dict[page.key] = page

        ## Process frames.
        self.frames = []
        self.screens = []
        self.keyrefs = []
        for descendant in element.findall(".//frame"):
            frame = Frame(dataset, self, descendant)
            self.frames.append(frame)
            self.screens.extend(frame.screens)
            self.keyrefs.extend(frame.keyrefs)

    def __repr__(self):
        return "Video %s" % self.dirname

class Document(object):
    """ This class represents a document. """
    def __init__(self, dataset, parent, element):
        """Constructs the object representation of a document.

        Parameters:
            dataset The dataset to which the document belongs.
            parent  The parent Video object.
            element The XML element that represents the document.
        """
        self.dataset = dataset
        self.video = parent

        # Set own attributes.
        self.filename = "%s/%s" % (parent.dirname, element.attrib["filename"])

        # Process descendant elements.
        self.pages = []
        for descendant in element.findall(".//page"):
            page = Page(dataset, self, descendant)
            self.pages.append(page)

    def __repr__(self):
        return "Document %s" % self.filename

class Page(object):
    """ This class represents a page in a document. """
    def __init__(self, dataset, parent, element):
        """Constructs the object representation of a page in a document.

        Parameters:
            dataset The dataset to which the page belongs.
            parent  The parent Document object.
            element The XML element that represents the page.
        """
        self.dataset = dataset
        self.document = parent
        self.video = self.document.video

        # Set own attributes.
        self.filename = "%s/%s" % (self.video.dirname, element.attrib["filename"])
        self.key = element.attrib["key"]
        self.number = int(element.attrib["number"])
        self.vgg256 = json.loads(element.attrib["vgg256"])

    def __repr__(self):
        return "Page %s" % self.filename

class Frame(object):
    """ This class represents a video frame. """
    def __init__(self, dataset, parent, element):
        """Constructs the object representation of a video frame.

        Parameters:
            dataset The dataset to which the video frame belongs.
            parent  The parent Video object.
            element The XML element that represents the video frame.
        """
        self.dataset = dataset
        self.video = parent

        # Set own attributes.
        self.filename = "%s/%s" % (self.video.dirname, element.attrib["filename"])
        self.number = int(element.attrib["number"])
        self.vgg256 = json.loads(element.attrib["vgg256"])

        # Process descendant elements.
        self.screens = []
        self.keyrefs = []
        for descendant in element.findall(".//screen"):
            screen = Screen(dataset, self, descendant)
            self.screens.append(screen)
            self.keyrefs.extend(screen.keyrefs)

    def __repr__(self):
        return "Frame %s" % self.filename

# This class specifies a point in the 2D projection space of a video frame.
Coordinate = namedtuple("Coordinate", ['x', 'y'])

# This class specifies a bounding quadrilinear of a screen on a video frame.
BoundingQuadrilinear = namedtuple("BoundingQuadrilinear",
                                  ["top_left", "top_right", "bottom_left", "bottom_right"])

class Screen(object):
    """ This class represents a screen on a video frame. """
    def __init__(self, dataset, parent, element):
        """Constructs the object representation of a screen on a video frame.

        Parameters:
            dataset The dataset to which the screen belongs.
            parent  The parent Frame object.
            element The XML element that represents the screen.
        """
        self.dataset = dataset
        self.frame = parent
        self.video = self.frame.video

        # Set own attributes.
        self.condition = element.attrib["condition"]
        self.vgg256 = json.loads(element.attrib["vgg256"])
        top_left = Coordinate(int(element.attrib["x0"]), int(element.attrib["y0"]))
        top_right = Coordinate(int(element.attrib["x1"]), int(element.attrib["y1"]))
        bottom_left = Coordinate(int(element.attrib["x2"]), int(element.attrib["y2"]))
        bottom_right = Coordinate(int(element.attrib["x3"]), int(element.attrib["y3"]))
        self.bounds = BoundingQuadrilinear(top_left, top_right, bottom_left, bottom_right)
        self.is_beyond_bounds = self.bounds.top_left.x < 0 \
                or self.bounds.bottom_left.x < 0 \
                or self.bounds.top_right.x >= self.video.width \
                or self.bounds.bottom_right.x >= self.video.width \
                or self.bounds.top_left.y < 0 \
                or self.bounds.top_right.y < 0 \
                or self.bounds.bottom_left.y >= self.video.height \
                or self.bounds.bottom_right.y >= self.video.height

        # Process descendant elements.
        self.keyrefs = []
        for descendant in element.findall(".//keyref"):
            keyref = KeyRef(self, dataset, descendant)
            self.keyrefs.append(keyref)
        self.matching_pages = set([keyref.page for keyref in self.keyrefs \
                                   if keyref.similarity == "full"])
        if not self.matching_pages: # If there is no fully matching page, accept any matching page.
            self.matching_pages = set((keyref.page for keyref in self.keyrefs))

    def is_outlier(self, windowed=True, obstacle=True, beyond_bounds=True, incremental=True,
                   no_match=True):
        """
            Returns whether the screen is an outlier.

            Parameters
                windowed        If the screen displays windowed content, it is considered an
                                outlier.
                obstacle        If the screen is obscured by an obstacle, it is considered an
                                outlier.
                beyond_bounds   If the screen goes beyond the bounds of the video, it is considered
                                an outlier.
                incremental     If the screen has no corresponding fully matching document page, it
                                is considered an outlier.
                no_match        If the screen has no corresponding matching document page, it is
                                considered an outlier.

            The above parameters fully characterize an outlier.
        """
        if windowed and self.condition == "windowed":
            return True
        if obstacle and self.condition == "obstacle":
            return True
        if beyond_bounds and self.is_beyond_bounds:
            return True
        if incremental and not [keyref for keyref in self.keyrefs if keyref.similarity == "full"]:
            return True
        if no_match and not self.keyrefs:
            return True
        return False

    def __repr__(self):
        return "%s, screen #%d" % (self.frame, self.frame.screens.index(self)+1)

class KeyRef(object):
    """
        This class represents a is-displayed-on relation between a document page and a screen on a
        video frame.
    """
    def __init__(self, parent, dataset, element):
        """Constructs the object representation of a relation between a document page and a screen.

        Parameters:
            dataset The dataset to which the screen belongs.
            parent  The parent Screen object.
            element The XML element that represents the relation.
        """
        self.dataset = dataset
        self.video = parent.video
        self.page = self.video.page_dict[element.text]

        # Set own attributes.
        self.similarity = element.attrib["similarity"]

    def __repr__(self):
        return "KeyRef: %s <-> %s" % (self.video, self.page)
