from typing import Any, Dict, List

from gym import spaces
from transformers import AutoModel, AutoTokenizer
import torch

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    text_uuid: str = "text",
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    if (
        instruction_sensor_uuid not in observations[0]
        or instruction_sensor_uuid == "pointgoal_with_gps_compass"
    ):
        return observations
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and text_uuid in observations[i][instruction_sensor_uuid]
        ):
            instruction = observations[i][
                instruction_sensor_uuid
            ][text_uuid]
            tokenized_inputs = tokenizer(instruction)
            observations[i][instruction_sensor_uuid] = tokenized_inputs['input_ids']
        else:
            break
    return observations


def single_frame_box_shape(box: spaces.Box) -> spaces.Box:
    """removes the frame stack dimension of a Box space shape if it exists."""
    if len(box.shape) < 4:
        return box

    return spaces.Box(
        low=box.low.min(),
        high=box.high.max(),
        shape=box.shape[1:],
        dtype=box.high.dtype,
    )
