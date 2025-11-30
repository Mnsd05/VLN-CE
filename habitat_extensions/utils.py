import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import habitat_sim
import numpy as np
import quaternion
import torch
from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_to_list,
)
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from numpy import ndarray
from torch import Tensor

from habitat_extensions import maps

cv2 = try_cv2_import()


def observations_to_image(
    observation: Dict[str, Any], info: Dict[str, Any]
) -> ndarray:
    """Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    frame = egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        map_k = "top_down_map"

    if map_k is not None:
        td_map = info[map_k]["map"]

        td_map = maps.colorize_topdown_map(
            td_map,
            info[map_k]["fog_of_war_mask"],
            fog_of_war_desat_amount=0.75,
        )
        td_map = habitat_maps.draw_agent(
            image=td_map,
            agent_center_coord=info[map_k]["agent_map_coord"],
            agent_rotation=info[map_k]["agent_angle"],
            agent_radius_px=min(td_map.shape[0:2]) // 24,
        )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, td_map), axis=1)
    return frame


def add_id_on_img(img: ndarray, txt_id: str) -> ndarray:
    img_height = img.shape[0]
    img_width = img.shape[1]
    white = np.ones((10, img.shape[1], 3)) * 255
    img = np.concatenate((img, white), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    thickness = 2
    text_width = cv2.getTextSize(txt_id, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        txt_id,
        (start_width, img_height),
        font,
        font_size,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def add_instruction_on_img(img: ndarray, text: str) -> None:
    font_size = 1.1
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(
        text, width=int((img.shape[1] - 15) / char_size[0])
    )
    if len(wrapped_text) < 8:
        wrapped_text.insert(0, "")

    y = 0
    start_x = 15
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + 25
        cv2.putText(
            img,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

def add_stop_prob_on_img(img: ndarray, stop: float, selected: bool) -> ndarray:
    img_width = img.shape[1]
    txt = "stop: " + str(round(stop, 2))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    thickness = 2 if selected else 1
    text_width = cv2.getTextSize(txt, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        txt,
        (start_width, 20),
        font,
        font_size,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def navigator_video_frame(
    observations,
    info,
    start_pos,
    start_heading,
    action=None,
    map_k="top_down_map_vlnce",
    frame_width=2048,
):
    def _rtheta_to_global_coordinates(
        r, theta, current_position, current_heading
    ):
        phi = (current_heading + theta) % (2 * np.pi)
        x = current_position[0] - r * np.sin(phi)
        z = current_position[-1] - r * np.cos(phi)
        return [x, z]

    rgb = {k: v for k, v in observations.items() if k.startswith("rgb")}
    rgb["rgb_0"] = rgb["rgb"]
    del rgb["rgb"]
    rgb = [
        f[1]
        for f in sorted(rgb.items(), key=lambda f: int(f[0].split("_")[1]))
    ]

    rgb = [
        add_id_on_img(rgb[i][:, 80 : (rgb[i].shape[1] - 80), :], str(i))
        for i in range(len(rgb))
    ][::-1]
    rgb = np.concatenate(rgb[6:] + rgb[:6], axis=1).astype(np.uint8)
    new_height = int((frame_width / rgb.shape[1]) * rgb.shape[0])
    rgb = cv2.resize(
        rgb,
        (frame_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )

    top_down_map = deepcopy(info[map_k]["map"])

    if action is not None and "action_args" in action:
        maps.draw_waypoint_prediction(
            top_down_map,
            _rtheta_to_global_coordinates(
                action["action_args"]["r"],
                action["action_args"]["theta"],
                start_pos,
                heading_from_quaternion(start_heading),
            ),
            info[map_k]["meters_per_px"],
            info[map_k]["bounds"],
        )

    top_down_map = maps.colorize_topdown_map(
        top_down_map,
        info[map_k]["fog_of_war_mask"],
        fog_of_war_desat_amount=0.75,
    )
    map_agent_pos = info[map_k]["agent_map_coord"]
    top_down_map = habitat_maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info[map_k]["agent_angle"],
        agent_radius_px=int(0.45 / info[map_k]["meters_per_px"]),
    )
    if top_down_map.shape[1] < top_down_map.shape[0]:
        top_down_map = np.rot90(top_down_map, 1)

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map
    old_h, old_w, _ = top_down_map.shape
    top_down_height = rgb.shape[0]
    top_down_width = int(old_w * (top_down_height / old_h))
    top_down_map = cv2.resize(
        top_down_map,
        (int(top_down_width), top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    inst_white = (
        np.ones(
            (top_down_map.shape[0], rgb.shape[1] - top_down_map.shape[1], 3)
        )
        * 255
    )
    add_instruction_on_img(inst_white, observations["instruction"]["text"])
    map_and_inst = np.concatenate((inst_white, top_down_map), axis=1)
    horizontal_white = np.ones((50, rgb.shape[1], 3)) * 255
    return np.concatenate(
        (rgb, horizontal_white, map_and_inst), axis=0
    ).astype(np.uint8)


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[ndarray],
    episode_id: Union[str, int],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    """Generate video according to specified information. Using a custom
    verion instead of Habitat's that passes FPS to video maker.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def compute_heading_to(
    pos_from: Union[List[float], ndarray], pos_to: Union[List[float], ndarray]
) -> Tuple[List[float], float]:
    """Compute the heading that points from position `pos_from` to position `pos_to`
    in the global XZ coordinate frame.

    Args:
        pos_from: [x,y,z] or [x,z]
        pos_to: [x,y,z] or [x,z]

    Returns:
        heading quaternion as [x, y, z, w]
        heading scalar angle
    """
    delta_x = pos_to[0] - pos_from[0]
    delta_z = pos_to[-1] - pos_from[-1]
    xz_angle = np.arctan2(delta_x, delta_z)
    xz_angle = (xz_angle + np.pi) % (2 * np.pi)
    quat = quaternion_to_list(
        quaternion.from_euler_angles([0.0, xz_angle, 0.0])
    )
    return quat, xz_angle


def heading_from_quaternion(quat: quaternion.quaternion) -> float:
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    heading_vector = quaternion_rotate_vector(
        quat.inverse(), np.array([0, 0, -1])
    )
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)


def rtheta_to_global_coordinates(
    sim: Simulator,
    r: float,
    theta: float,
    y_delta: float = 0.0,
    dimensionality: int = 2,
) -> List[float]:
    """Maps relative polar coordinates from an agent position to an updated
    agent position. The returned position is not validated for navigability.
    """
    assert dimensionality in [2, 3]
    scene_node = sim.get_agent(0).scene_node
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    agent_state = sim.get_agent_state()
    rotation = habitat_sim.utils.quat_from_angle_axis(
        theta, habitat_sim.geo.UP
    )
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)
    position = agent_state.position + (move_ax * r)
    position[1] += y_delta

    if dimensionality == 2:
        return [position[0], position[2]]
    return position
