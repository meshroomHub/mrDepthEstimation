__version__ = "1.0"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

# models paths
MOGE_MODEL_PATH = os.getenv('MOGE_MODEL_PATH')
VDA_MODEL_PATH = os.getenv('VDA_MODEL_PATH')

class DepthEstimationNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        extension = extension_param.value
        input_path = input_path_param.value
        image_paths = get_image_paths_list(input_path, extension)

        return(max(1, len(image_paths)))


class DepthEstimationBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        # case when block size is 0, only one block is used
        else:
            return size, size, 1


class DepthEstimation(desc.Node):
    category = "Depth Estimation"
    documentation = """This node generates an estimated depth map from a monocular image sequence."""
    
    gpu = desc.Level.INTENSIVE

    size = DepthEstimationNodeSize(['imagesFolder', 'inputExtension'])
    parallelization = DepthEstimationBlockSize()

    inputs = [
        desc.File(
            name="imagesFolder",
            label="Images Folder",
            description="Input images to estimate the depth from.",
            value="",
        ),
        desc.ChoiceParam(
            name="inputExtension",
            label="Input Extension",
            description="Extension of the input images. This will be used to determine which images are to be used if \n"
                        "a directory is provided as the input. If \"\" is selected, the provided input will be used as such.",
            values=["jpg", "jpeg", "png", "exr"],
            value="exr",
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="model",
            label="Model",
            description="Model used during inference.",
            values=["MoGe", "Video-Depth-Anything"],
            value="MoGe",
        ),
        desc.BoolParam(
            name="automaticFOVEstimation",
            label="Automatic FOV Estimation",
            description="If this option is enabled, the MoGe model will estimate the field of view.",
            value=True,
            enabled=lambda node: node.model.value == "MoGe"
        ),
        desc.FloatParam(
            name="horizontalFov",
            label="Horizontal FOV",
            value=50.0,
            description="If camera parameters are known, set the horizontal field of view in degrees.",
            range=(0.0, 360.0, 1.0),
            enabled=lambda node: not node.automaticFOVEstimation.value,
        ),
        
        desc.BoolParam(
            name="saveMesh",
            label="Save Mesh",
            description="If this option is enabled, a ply file will be saved with the estimated mesh. The color will be saved as vertex colors.",
            value=False,
            enabled=lambda node: node.model.value == "MoGe"
        ),
        desc.IntParam(
            name="blockSize",
            label="Block Size",
            value=50,
            description="Sets the number of images to process in one chunk. If set to 0, all images are processed at once.",
            range=(0, 1000, 1),
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the estimated depth maps.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="depthMapVis",
            label="Depth Map Visualization",
            description="Color mapped output depth maps for visualization purpose",
            semantic="imageList",
            value="{nodeCacheFolder}/depth_vis/*_depth_vis.png",
            group="",
        ),
        desc.File(
            name="depthMap",
            label="Depth Map Output",
            description="Output depth maps",
            semantic="imageList",
            value="{nodeCacheFolder}/depth/*_depth.exr",
            group="",
        )
    ]

    def preprocess(self, node):
        extension = node.inputExtension.value
        input_path = node.imagesFolder.value

        image_paths = get_image_paths_list(input_path, extension)

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        self.image_paths = image_paths

    def processChunk(self, chunk):

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.imagesFolder.value:
                chunk.logger.warning('No input folder given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            # inference
            chunk.logger.info(f'Starting inference on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)} with {chunk.node.model.value} model...')
            if chunk.node.model.value == 'MoGe':
                from moge_utils.moge_inference import moge_inference

                fov =  None if chunk.node.automaticFOVEstimation else chunk.node.horizontalFov.value

                moge_inference(
                        input_image_paths=chunk_image_paths,
                        fov_x_= fov,
                        output_path = chunk.node.output.value,
                        pretrained_model = MOGE_MODEL_PATH,
                        threshold = 0.03,
                        ply = chunk.node.saveMesh.value)

            elif chunk.node.model.value == 'Video-Depth-Anything':
                from vda_utils.vda_inference import vda_inference

                vda_inference(
                    input_image_paths=chunk_image_paths,
                    output_path = chunk.node.output.value,
                    pretrained_model= VDA_MODEL_PATH
                )
            
            chunk.logger.info('Publish end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pathlib import Path
    import itertools

    include_suffices = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        raise ValueError(f"Input path '{input_path}' is not a directory.")
    return image_paths
