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


from numpy import single
from caret_analyze.record.record_factory import RecordFactory
from platform import node
from caret_analyze.architecture.architecture import Architecture
from caret_analyze.infra.lttng import Lttng, RecordsProviderLttng
from caret_analyze.infra.lttng.lttng import LttngEventFilter
from caret_analyze.plot.bokeh.callback_info_factory import Plot
from caret_analyze.plot.bokeh.callback_sched import callback_sched
from caret_analyze.plot.bokeh.message_flow import message_flow
from caret_analyze.plot.bokeh.deadlinemiss import deadlinemiss, deadline_miss_plot
from caret_analyze.plot.graphviz.chain_latency import chain_latency
from caret_analyze.runtime.application import Application
from caret_analyze.value_objects import CallbackGroupType, TimerStructValue
from bokeh.plotting import output_notebook, figure, show
from caret_analyze import Architecture, Application, Lttng, LttngEventFilter


import pandas as pd

from caret_analyze.value_objects.callback import TimerCallbackStructValue

class TestRecordFactory:

    # def test_is_cpp_impl_valid(self):
    #     is_valid = RecordFactory.is_cpp_impl_valid()
    #     assert is_valid, 'Failed to find record_cpp_impl. Skip Cpp package tests.'

    def test_callback_sched(self):
        # lttng = Lttng('./test/trace-reference-system-20220710003817_muti_1core')
        # arch = Architecture('yaml',
        #                       './test/arch_reference_system_single.yaml')
        # arch = Architecture('yaml',
        #                       './test/arch_reference_system_single_end_to_end.yaml')
        lttng = Lttng('./test/end_to_end_sample')
        arch = Architecture('yaml',
                              './test/arch_e2e_sample_use_latest_message.yaml')
        # arch = Architecture('yaml',
        #                       './test/arch_e2e_sample_sim.yaml')
        app = Application(arch, lttng)
        path = app.get_path('target')
        # node = app.get_node('/ObjectCollisionEstimator')
        deadlinemiss(app,path)
        # chain_latency(path,granularity='end-to-end')
        # executor = app.get_executor('executor_0')
        # node = app.get_node('/sensing/lidar/concatenate_data_front')
        # callback_sched(executor)
        single_miss = []
        # single_miss.append(deadline_miss_rates)
        # callback_sched(executor)
        # # single_miss=[0.3, 0.2, 0.1, 0.1]
        # muti_miss=[0.15, 0.2, 0.2, 0.3]
        # picas_miss=[0.1, 0.05, 0.05, 0.02]
        # cbg_miss=[0.1, 0.05, 0.05, 0.02]
        # deadline_miss_plot(single_miss,muti_miss,picas_miss,cbg_miss)
        # plot = Plot.create_callback_frequency_plot(executor)
        # plot.show('system_time', ywheel_zoom=False) 
        # plot.show('sim_time', ywheel_zoom=False)
        # plot.show('index', ywheel_zoom=False)   
        # deadline_miss(executor)