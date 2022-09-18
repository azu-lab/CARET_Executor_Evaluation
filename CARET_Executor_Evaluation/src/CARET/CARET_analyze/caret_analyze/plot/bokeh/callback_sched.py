# Copyright 2021 Research Institute of Systems Planning, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import abstractmethod
from importlib.resources import path

from logging import getLogger
from typing import Dict, List, Optional, Sequence, Tuple, Union

from bokeh.colors import Color, RGB
from bokeh.io import save, show
from bokeh.models import Arrow, HoverTool, NormalHead,CrosshairTool, Label
from bokeh.plotting import ColumnDataSource, figure
from bokeh.resources import CDN

from caret_analyze.runtime.callback import TimerCallback

import colorcet as cc

import pandas as pd

from .util import apply_x_axis_offset, get_callback_param_desc, RectValues
from ...common import ClockConverter, Util
from ...exceptions import InvalidArgumentError
from ...record import Clip
from ...runtime import CallbackBase, CallbackGroup, Executor, Node


logger = getLogger(__name__)


def callback_sched(
    target: Union[Node, CallbackGroup, Executor],
    lstrip_s: float = 0,
    rstrip_s: float = 0,
    coloring_rule='callback',
    use_sim_time: bool = False,
    export_path: Optional[str] = None
):
    assert coloring_rule in ['callback', 'callback_group', 'node']

    cbgs, target_name = get_cbg_and_name(target)
    callbacks = Util.flatten([cbg.callbacks for cbg in cbgs])
    frame_min, frame_max = get_range(callbacks)
    clip_min = int(frame_min + lstrip_s*1.0e9)
    clip_max = int(frame_max - rstrip_s*1.0e9)
    clip = Clip(clip_min, clip_max)

    color_selector = ColorSelector.create_instance(coloring_rule)
    sched_plot_cbg(target_name, cbgs, color_selector, clip, use_sim_time, export_path)


def get_cbg_and_name(
    target: Union[Node, CallbackGroup, Executor]
) -> Tuple[Sequence[CallbackGroup], str]:
    if isinstance(target, Node):
        if target.callback_groups is None:
            raise InvalidArgumentError('target.callback_groups is None')

        return target.callback_groups, target.node_name

    elif isinstance(target, Executor):
        return target.callback_groups, target.executor_name

    else:
        return [target], target.callback_group_name


def get_range(callbacks: Sequence[CallbackBase]) -> Tuple[int, int]:
    callbacks_valid = [cb for cb in callbacks if len(cb.to_records()) > 0]

    if len(callbacks_valid) == 0:
        logger.warning('Failed to found Callback measurement results.')
        return 0, 1

    cb_dfs = [cb.to_dataframe() for cb in callbacks]
    cb_dfs_valid = [cb_df for cb_df in cb_dfs if len(cb_df) > 0]
    cb_min = min(min(df.min()) for df in cb_dfs_valid)
    cb_max = max(max(df.max()) for df in cb_dfs_valid)

    return cb_min, cb_max


def sched_plot_cbg(
    target_name: str,
    cbgs: Sequence[CallbackGroup],
    color_selector: ColorSelector,
    clipper: Clip,
    use_sim_time: bool,
    export_path: Optional[str] = None
):

    p = figure(
               x_axis_label='Time [s]',
               y_axis_label='',
               title=f'Time-line of callbacks in CallbackGroupExecutor',
               width=1200,
               height=600,
               tools=['xwheel_zoom', 'xpan', 'save', 'reset'],
               active_scroll='xwheel_zoom',
                )
    p.sizing_mode = 'stretch_width'  # automatically adjust plot width to the screen size
    x_range_name = 'x_plot_axis'
    converter: Optional[ClockConverter] = None
    if use_sim_time:
        cbs: List[CallbackBase] = Util.flatten(
            cbg.callbacks for cbg in cbgs if len(cbg.callbacks) > 0)
        converter = cbs[0]._provider.get_sim_time_converter()  # TODO(hsgwa): refactor
        frame_min = converter.convert(clipper.min_ns)
        frame_max = converter.convert(clipper.max_ns)
    else:
        frame_min = clipper.min_ns
        frame_max = clipper.max_ns
    apply_x_axis_offset(p, x_range_name, frame_min, frame_max)

    rect_y = 0.0
    rect_height = 0.3
    rect_y_step = -1.5
    callback_idx = 0
    dely_times=0
    for callback_group in cbgs:
        for callback in callback_group.callbacks:
        #  if isinstance(callback, TimerCallback):
            i=0
            callback_idx += 1
            rect_source = get_callback_rects(callback, clipper, rect_y, rect_height, converter)
            bar_source = get_callback_bar(callback, rect_y, frame_max, frame_min)
            color = color_selector.get_color(
                callback.node_name,
                callback_group.callback_group_name,
                callback.callback_name)
            plot1 = p.rect(
                   'x',
                   'y',
                   'width',
                   'height',
                   source=rect_source,
                   color=color,
                   alpha=1.0,
                   legend_label=f'{callback.callback_name}',
                   # Since setting callback name to legend will narrow the graph,
                   # sequential numbering is used here.
                   hover_fill_color=color,
                   hover_alpha=1.0,
                   x_range_name=x_range_name)

            plot2 = p.rect(
                   'x',
                   'y',
                   'width',
                   'height',
                   source=bar_source,
                   fill_color=color,
                   legend_label=f'{callback.callback_name}',
                   hover_fill_color=color,
                   hover_alpha=0.1,
                   fill_alpha=0.1,
                   x_range_name=x_range_name)

            Hover1 = HoverTool(
                     renderers=[plot1],
                     tooltips="""
                     <div style="width:400px; word-wrap: break-word;">
                     <br>
                     callback_start = @x_min [ns] <br>
                     callback_end = @x_max [ns] <br>
                     latency = @latency [ms] <br>
                     """,
                     toggleable=False,
                     attachment='above')

            Hover2 = HoverTool(
                     renderers=[plot2],
                     tooltips="""
                     <div style="width:400px; word-wrap: break-word;">
                     <br>
                     node_name = @node_name <br>
                     callback_name = @callback_name <br>
                     callback_type = @callback_type <br>
                     @callback_param <br>
                     symbol = @symbol
                     </div>
                     """,
                     toggleable=False,
                     point_policy='follow_mouse',
                     attachment='below')
            p.add_tools(Hover1)
            p.add_tools(Hover2)
            p.output_backend = 'svg'
            if isinstance(callback, TimerCallback):
                y_start = rect_source.data['y'][1]+0.9
                y_end = rect_source.data['y'][1]+rect_height
                timer = callback.timer
                df = timer.to_dataframe()
                for item in df.itertuples():
                    timerstamp = item._1
                    callback_start = item._2
                    # callback_end = item._3
                    res = callback_start-timerstamp
                    delayed_th = 500000
                    # The callback is considered delayed if this value is exceeded.
                    if not pd.isna(res):
                        if res > delayed_th:
                            i=i+1
                            p.add_layout(Arrow(end=NormalHead(
                                fill_color='red',
                                line_width=1,
                                size=10
                                ),
                                       x_start=(timerstamp-frame_min)*1.0e-9, y_start=y_start,
                                       x_end=(timerstamp-frame_min)*1.0e-9, y_end=y_end))
                        else:
                            p.add_layout(Arrow(end=NormalHead(
                                fill_color='white',
                                line_width=1,
                                size=10),
                                       x_start=(timerstamp-frame_min)*1.0e-9, y_start=y_start,
                                       x_end=(timerstamp-frame_min)*1.0e-9, y_end=y_end))
            rect_y += rect_y_step
            dely_times=i+dely_times
    p.ygrid.grid_line_alpha = 0
    p.yaxis.visible = False
    p.legend.location = 'bottom_left'
    p.legend.click_policy = 'hide'
    p.legend.orientation = 'vertical'
    # p.title=f'Time-line of callbacks in CallbackGroupExecutor. Timer callback delayed times:{dely_times}'
    p.title=f'Time-line of callbacks in SingleThreadedExecutor.'
    p.add_layout(p.legend[0], 'right')
    p.legend.label_text_font_size = '15pt'
    p.title.text_font_size = '12pt'
    linked_crosshair = CrosshairTool(dimensions="both")
    p.add_tools(linked_crosshair)
    if export_path is None:
        show(p)
    else:
        save(p, export_path, title='callback execution timing-chart', resources=CDN)


def get_callback_rects(
    callback: CallbackBase,
    clip: Clip,
    y,
    height,
    converter: Optional[ClockConverter]
) -> ColumnDataSource:
    y_min = y - height
    y_max = y + height

    rect_source = ColumnDataSource(data={
        'x': [],
        'y': [],
        'x_min': [],
        'x_max': [],
        'width': [],
        'latency': [],
        'height': []
    })

    df = callback.to_dataframe(shaper=clip)
    for item in df.itertuples():
        callback_start = item._1
        callback_end = item._2
        if converter:
            callback_start = converter.convert(callback_start)
            callback_end = converter.convert(callback_end)

        rect = RectValues(callback_start, callback_end, y_min, y_max)
        new_data = {
            'x': [rect.x],
            'y': [rect.y],
            'x_min': [callback_start],
            'x_max': [callback_end],
            'width': [rect.width],
            'latency': [(callback_end-callback_start)*1.0e-6],
            'height': [rect.height]
        }
        rect_source.stream(new_data)
    return rect_source


def get_callback_bar(
    callback: CallbackBase,
    y,
    frame_max,
    frame_min
) -> ColumnDataSource:
    y_min = y - 0.6
    y_max = y + 0.5

    rect_source = ColumnDataSource(data={
        'x': [],
        'y': [],
        'width': [],
        'height': [],
        'node_name': [],
        'callback_name': [],
        'callback_type': [],
        'callback_param': [],
        'symbol': []

    })

    callback_param = get_callback_param_desc(callback)
    bar_start = frame_min - 10000000000
    bar_end = frame_max + 10000000000
    rect = RectValues(bar_start, bar_end, y_min, y_max)
    rect_source = ColumnDataSource(data={
            'x': [rect.x],
            'y': [rect.y],
            'width': [rect.width],
            'height': [rect.height],
            'node_name': [callback.node_name],
            'callback_name': [callback.callback_name],
            'symbol': [callback.symbol],
            'callback_param': [callback_param],
            'callback_type': [f'{callback.callback_type}']
        })

    return rect_source


class ColorSelector:

    @staticmethod
    def create_instance(coloring_rule: str):
        if coloring_rule == 'callback':
            return ColorSelectorCallback()

        if coloring_rule == 'callback_group':
            return ColorSelectorCbg()

        if coloring_rule == 'node':
            return ColorSelectorNode()

    def __init__(self) -> None:
        self._palette: Sequence[Color] = [self._from_rgb(*rgb) for rgb in cc.glasbey_bw_minc_20]
        self._color_map: Dict[str, Color] = {}

    def get_color(self, node_name: str, cbg_name: str, callback_name: str) -> Color:
        color_hash = self._get_color_hash(node_name, cbg_name, callback_name)

        if color_hash not in self._color_map:
            color_index = len(self._color_map) % len(self._palette)
            self._color_map[color_hash] = self._palette[color_index]

        return self._color_map[color_hash]

    @abstractmethod
    def _get_color_hash(self, node_name: str, cbg_name: str, callback_name: str) -> Color:
        return

    @staticmethod
    def _from_rgb(r: float, g: float, b: float) -> Color:
        r_ = int(r*255)
        g_ = int(g*255)
        b_ = int(b*255)
        return RGB(r_, g_, b_)


class ColorSelectorCallback(ColorSelector):

    def _get_color_hash(self, node_name: str, cbg_name: str, callback_name: str) -> Color:
        return callback_name


class ColorSelectorCbg(ColorSelector):

    def _get_color_hash(self, node_name: str, cbg_name: str, callback_name: str) -> Color:
        return cbg_name


class ColorSelectorNode(ColorSelector):

    def _get_color_hash(self, node_name: str, cbg_name: str, callback_name: str) -> Color:
        return node_name
