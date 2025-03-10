__version__ = "1.0"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

# models paths
MOGE_MODEL_PATH = os.getenv('MOGE_MODEL_PATH')
# VDA_MODEL_PATH = os.getenv('VDA_MODEL_PATH')

class DepthEstimation(desc.Node):
    category = "Depth Estimation"
    documentation = """This node generates an estimated depth map from a monocular image sequence."""
    
    gpu = desc.Level.INTENSIVE

    inputs = [
        desc.File(
            name="imagesFolder",
            label="Images Folder",
            description="Input images to estimate the depth from",
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
            values=["MoGe"], #, "Video-Depth-Anything"],
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
            value="{nodeCacheFolder}/*/depth_vis.png",
            group="",
        ),
        desc.File(
            name="depthMap",
            label="Depth Map Output",
            description="Output depth maps",
            semantic="imageList",
            value="{nodeCacheFolder}/*/depth.exr",
            group="",
        )
    ]

    def processChunk(self, chunk):
        from moge_utils.moge_inference import moge_inference

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.imagesFolder.value:
                chunk.logger.warning('No input folder given.')

            # inference
            if chunk.node.model.value == 'MoGe':

                fov =  None if chunk.node.automaticFOVEstimation else chunk.node.horizontalFov.value

                moge_inference(
                        input_path=chunk.node.imagesFolder.value,
                        fov_x_= fov,
                        output_path = chunk.node.output.value,
                        pretrained_model = MOGE_MODEL_PATH,
                        threshold = 0.03,
                        extension = chunk.node.inputExtension.value,
                        ply = chunk.node.saveMesh.value)
            elif chunk.node.model.value == 'Video-Depth-Anything':
                chunk.logger.warning('Model not implemented yet')
            
            chunk.logger.info('Publish end')
        finally:
            chunk.logManager.end()


