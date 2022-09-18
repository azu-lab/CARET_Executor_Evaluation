
from __future__ import annotations

from abc import abstractmethod

from logging import getLogger
from operator import index
from platform import node
from typing import Dict, List, Optional, Sequence, Tuple, Union

from IPython.display import display
from ipywidgets import interact
from bokeh.colors import Color, RGB
from bokeh.io import show
from bokeh.models import Arrow, NormalHead, HoverTool, CustomJS,CustomJSHover
from bokeh.plotting import ColumnDataSource, figure
from caret_analyze import Application
from bokeh.models import Span
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.layouts import gridplot,layout
from bokeh.layouts import row
from bokeh.models import CustomJS
from bokeh.models import Span, CrosshairTool, HoverTool, ResetTool, PanTool, WheelZoomTool
from bokeh.models.widgets import Tabs, Panel

import numpy as np 
import pandas as pd
from bokeh.transform import dodge
from bokeh.core.properties import value
from bokeh.layouts import Column, Row

from ...runtime import CallbackBase, Path
from ...runtime import CallbackBase, CallbackGroup, Executor, Node
from bokeh.io import curdoc
import yaml
from ...exceptions import InvalidYamlFormatError
from bokeh.io import push_notebook, show, output_notebook

def deadlinemiss(app:Application,  target: Union[Node, Path],loc:str):
    arch=deadline(loc)
    deadline_dict=get_path_deadline_dict(arch)
    check_end_to_end_path(arch,app,target,deadline_dict)
    return 0


def deadline(loc:str):
    arch=read_yaml(loc)
    return arch

def read_yaml(file_path: str):
        with open(file_path, 'r') as f:
            yaml_str = f.read()
        arch = yaml.safe_load(yaml_str)

        if arch is None:
            raise InvalidYamlFormatError('Failed to parse yaml.')
        else:
            return arch


def get_node_deadline_dict(arch, target_name):
    node_dict = get_node_dict(arch, target_name)
    node_name=[]
    deadline=[]
    node_name.append(get_value(node_dict, 'node_name'))
    deadline.append(get_value(node_dict, 'deadline'))
    deadline_dict = dict(zip(node_name, deadline))
    return deadline_dict

def check_end_to_end_path(arch,app:Application,path:Path,deadline_dick):
    path_deadline_miss_times=0
    path_deadline=deadline_dick[path.path_name]
    treat_drop_as_delay=True
    lstrip_s=0
    rstrip_s=0
    latencys=get_end_to_end_latency(path, treat_drop_as_delay, lstrip_s, rstrip_s)
    p1,p2,p3=show_stack(arch,path,app,path_deadline,latencys,percent=False)
    show_mix(p1,p2,p3)
    return 0

def get_end_to_end_latency(
    path: Path,
    treat_drop_as_delay: bool,
    lstrip_s: float,
    rstrip_s: float
) :
    node_paths = path.node_paths

    for node_path in [node_paths[0], node_paths[-1]]:
        node_name = node_path.node_name
        label = node_name
        if len(node_path.column_names) != 0:
            _, latency = node_path.to_timeseries(
                remove_dropped=True,
                treat_drop_as_delay=treat_drop_as_delay,
                lstrip_s=lstrip_s,
                rstrip_s=rstrip_s,
            )

    _, latency = path.to_timeseries(
        remove_dropped=True,
        lstrip_s=lstrip_s,
        rstrip_s=rstrip_s,
    )
    return latency* 1.0e-6

def get_value(obj: Dict, key_name: str):
        try:
            v = obj[key_name]
            if v is None:
                raise InvalidYamlFormatError(f"'{key_name}' value is None. obj: {obj}")
            return v
        except (KeyError, TypeError) as e:
            msg = f'Failed to parse yaml file. {key_name} not found. obj: {obj}'
            raise InvalidYamlFormatError(msg) from e

def get_node_dict(
        arch,
        node_name: str
    ) -> Dict:
        node_values = get_value(arch, 'nodes')
        nodes = list(filter(lambda x: get_value(
            x, 'node_name') == node_name, node_values))

        if len(nodes) == 0:
            message = f'Failed to find node by node_name. target node name = {node_name}'
            raise InvalidYamlFormatError(message)

        if len(nodes) >= 2:
            message = (
                'Failed to specify node by node_name. same node_name are exist.'
                f'target node name = {node_name}')
            raise InvalidYamlFormatError(message)

        return nodes[0]

def get_path_deadline_dict(arch):
        aliases_info = get_value(arch, 'named_paths')
        path_name = []
        deadline_path = []
        for alias in aliases_info:
            path_name.append(get_value(alias, 'path_name'))
            deadline_path.append(get_value(alias, 'deadline'))
        deadline_dict_path = dict(zip(path_name, deadline_path))
            
        return deadline_dict_path


def deadline_dataframe_node_path(
    arch,
    path: Path,
    treat_drop_as_delay=False,
    lstrip_s: float = 0,
    rstrip_s: float = 0):
    data=[]
    columns1 = ["node_name", "min_latency [ms]","avg_latency [ms]","max_latency [ms]","node_deadline [ms]","node_deadline_miss_rate [%]"]
    for node_path in path.node_paths:
        deadline_miss_times=0
        node_name = node_path.node_name
        node_deadline_dict=get_node_deadline_dict(arch,node_name)
        node_deadline=node_deadline_dict[node_name]
        if node_path.column_names != []:
            _, latency = node_path.to_timeseries(
                remove_dropped=True,
                treat_drop_as_delay=treat_drop_as_delay,
                lstrip_s=lstrip_s,
                rstrip_s=rstrip_s,
            )
            for node_latency in latency* 1.0e-6:
                if node_latency > node_deadline:
                    deadline_miss_times+=1
            node_deadline_miss_rate=(deadline_miss_times/len(latency))* 100
            min_latency, avg_latency, max_latency=latency_manage(latency)
            data.append([node_name,min_latency, avg_latency, max_latency,node_deadline,node_deadline_miss_rate])
    node_latency=pd.DataFrame(data=data, columns=columns1)
    return node_latency

def show_stack(
    arch,
    path: Path,
    app: Application,
    path_deadline,
    latency: np.array,
    percent:bool,
    treat_drop_as_delay=False,
    lstrip_s: float = 0,
    rstrip_s: float = 0,):
    node_names=[node_path.node_name for node_path in path.node_paths]
    global df 
    df= pd.DataFrame(columns=node_names)
    p1 = figure(plot_width=1000,
           plot_height=350, 
           )
    p1.yaxis.axis_label = 'Latency [ms]'
    p2 = figure(plot_width=1000,
           plot_height=350, 
           x_range=p1.x_range
           )
    df=get_dataframe(app,path,arch)
    colors = ["#c9d9d3", "#718dbf", "#e84d60","yellow","blue","green","#1f77b4","#ff7f0e","#9467bd","#8c564b"]
    index=list(df.index)
    df.drop([len(df)-1],inplace=True)
    df1=pd.DataFrame(columns=node_names)
    df2=pd.DataFrame(columns=node_names)
    df1=df
    idx = node_names
    df2[idx]=df[idx].divide((df.sum(axis=1)), axis=0)  
    p2.yaxis.axis_label = 'Percentage of End-to-end Latency'
    p2.yaxis.formatter = NumeralTickFormatter(format='0 %')
    source1 = ColumnDataSource(df1)
    source2 = ColumnDataSource(df2)
    renderers=p1.vbar_stack(
            node_names,
            x='index',
            width=0.5,
            source=source1,
            color=colors,
            legend_label=node_names,
            )
    renderers2=p2.vbar_stack(
            node_names,
            x='index',
            width=0.5,
            source=source2,
            color=colors,
            legend_label=node_names,
            )
    for r in renderers:
        node_name = r.name
        hover = HoverTool(tooltips=[
        ("%s :" % node_name,"@$name"),
        ("index", "$index"),
        ], renderers=[r])
        p1.add_tools(hover)
    for r in renderers2:
        node_name = r.name
        hover = HoverTool(tooltips=[
        ("%s :" % node_name,"@$name"),
        ("index", "$index"),
        ], renderers=[r])
        p2.add_tools(hover)
    span=Span(location=path_deadline,
                            dimension='width', line_color='red',
                            line_dash='dashed', line_width=5)
    p1.add_layout(span)
    p2.add_layout(span)
    min_latency,avg_latency,max_latency=latency_manage(latency)
    p1.xaxis.axis_label = 'Index'
    p2.xaxis.axis_label = 'Index'
    p1.y_range.start = 0
    p1.x_range.range_padding = 0.5
    p1.xgrid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
    p1.add_layout(p1.legend[0], 'right')
    p1.legend.click_policy = 'hide'
    p2.y_range.start = 0
    p2.x_range.range_padding = 0.5
    p2.xgrid.grid_line_color = None
    p2.axis.minor_tick_line_color = None
    p2.outline_line_color = None
    p2.add_layout(p2.legend[0], 'right')
    p2.legend.click_policy = 'hide'
    linked_crosshair = CrosshairTool(dimensions="both")
    p1.output_backend = 'svg'
    p2.output_backend = 'svg'
    p1.add_tools(linked_crosshair)
    p2.add_tools(linked_crosshair)
    p2.title=f'Callback-group-level Executor. End-to-end Latency [ms]: avg:{avg_latency:.1f} min:{min_latency:.1f} max:{max_latency:.1f}'
    p1.legend.label_text_font_size = '15pt'
    p1.title.text_font_size = '12pt'
    p2.legend.label_text_font_size = '13pt'
    p2.title.text_font_size = '12pt'
    p = gridplot([[p1], [p2]])
    return p1,p2,p

def show_mix(fig1,fig2,fig3):
    
    # Create two panels, one for each conference
    panel1 = Panel(child=fig1, title='normal')
    panel2 = Panel(child=fig2, title='percent')
    panel3 = Panel(child=fig3, title='compare')
    # Assign the panels to Tabs
    tabs = Tabs(tabs=[panel1, panel2,panel3])
    # Show the tabbed layout
    show(tabs)

def get_dataframe(app:Application,path:Path, arch):
    deadline=0
    treat_drop_as_delay=False
    lstrip_s: float = 0
    rstrip_s: float = 0
    for node_path in path.node_paths:
        if node_path.column_names != []:
            _, latency = node_path.to_timeseries(
                remove_dropped=True,
                treat_drop_as_delay=treat_drop_as_delay,
                lstrip_s=lstrip_s,
                rstrip_s=rstrip_s,
            )
            df[node_path.node_name]=pd.Series(latency* 1.0e-6)
        deadline_dict=get_node_deadline_dict(arch,node_path.node_name)
        node_deadline=deadline_dict[node_path.node_name]
        deadline=deadline+node_deadline
    return df

def latency_manage(latency: np.array):
    min_latency=np.min(latency )
    avg_latency=np.average(latency )
    max_latency=np.max(latency )
    return min_latency, avg_latency, max_latency

def get_callback_execution_time(callback:CallbackBase):
    df=callback.to_dataframe()
    callback_latency=[]
    for item in df.itertuples():
        callback_start = item._1
        callback_end = item._2
        execution_time=(callback_end-callback_start)*1.0e-6
        callback_latency.append(execution_time)
    return callback_latency