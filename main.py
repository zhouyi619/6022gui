import math
import time
from PyHT6022.LibUsbScope import Oscilloscope
import dearpygui.dearpygui as dpg
import threading
import queue


class Data(object):
    def __init__(self, time, sample_rate):
        self.point = time * sample_rate
        self.sample_rate = int(sample_rate)
        self.data_ch1 = []
        self.data_ch2 = []
        self.ch1_total_Q = 0
        self.ch2_total_Q = 0
        self.sampling_resistor = 100

    def add(self, ch1, ch2):
        if len(self.data_ch1) + len(ch1) > self.point:
            del self.data_ch1[0 : int(len(self.data_ch1) + len(ch1) - self.point)]
            del self.data_ch2[0 : int(len(self.data_ch2) + len(ch2) - self.point)]
        self.data_ch1 += ch1
        self.data_ch2 += ch2
        self.ch1_total_Q += sum(map(lambda x: abs(x)  / self.sampling_resistor, ch1)) / len(ch1) * (1 / self.sample_rate * len(ch1))
        self.ch2_total_Q += sum(map(lambda x: abs(x) / self.sampling_resistor, ch2)) / len(ch2) * (1 / self.sample_rate * len(ch2))

    def get(self):
        return self.data_ch1, self.data_ch2

    def set_time(self, time):
        self.point = time * self.sample_rate
        print(f"data: set time to {time} , point to {self.point}")

    def get_time(self):
        return self.point / self.sample_rate

    def get_point(self):
        return self.point

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        self.point = self.point / self.sample_rate * sample_rate
        self.sample_rate = sample_rate
        print(f"data: set sample rate to {sample_rate}, point to {self.point}")

    def get_total_Q(self):
        return self.ch1_total_Q, self.ch2_total_Q
    
    def reset_total_Q(self):
        self.ch1_total_Q = 0
        self.ch2_total_Q = 0
        print(f"data: reset total Q")

    def set_sampling_resistor(self, sampling_resistor):
        self.sampling_resistor = sampling_resistor
        print(f"data: set sampling resistor to {sampling_resistor}")
        self.reset_total_Q()


def dataThread(control_queue, data_queue, sample_rate=20):
    try:
        print(f"dataThread {threading.current_thread().ident} start")
        ch1gain = 1
        ch2gain = 1

        def init(sample_rate, ch1gain, ch2gain):
            print("dataThread: init")

            scope = Oscilloscope()

            scope.setup()

            scope.open_handle()

            if not scope.is_device_firmware_present:
                scope.flash_firmware()

            scope.set_calibration_frequency(400)

            calibration = scope.get_calibration_values()

            scope.set_interface(0)
            scope.set_num_channels(2)
            valid_sample_rates = (20, 32, 50, 64, 100, 128, 200)
            sample_rate = int(sample_rate)

            if sample_rate not in valid_sample_rates:
                raise Exception("Invalid sample rate")
            else:
                sample_rate = sample_rate * 1000

            if sample_rate < 1e6:
                sample_id = int(round(100 + sample_rate / 10e3))
            else:
                sample_id = int(round(sample_rate / 1e6))
            scope.set_sample_rate(sample_id)

            scope.set_ch1_voltage_range(ch1gain)
            scope.set_ch2_voltage_range(ch2gain)
            global skip1st
            skip1st = True
            return scope

        def callback(ch1, ch2):
            global skip1st

            ch1_scaled = scope.scale_read_data(ch1, ch1gain, channel=1)
            ch2_scaled = scope.scale_read_data(ch2, ch2gain, channel=2)
            size = len(ch1)
            if size == 0:
                return
            if skip1st:
                skip1st = False
                return
            data_queue.put([ch1_scaled, ch2_scaled])

        capture_running = False
        inited = False
        print("dataThread: start loop")
        while True:
            try:
                s = control_queue.get(False)
                if s[0] == "stop":
                    print("scopeThread: get signal stop")
                    scope.stop_capture()
                    shutdown_event.set()
                    time.sleep(0.1)
                    scope.close_handle()
                    capture_running = False
                    break
                if s[0] == "start":
                    print("scopeThread: get signal start")
                    if not inited:
                        scope = init(sample_rate, ch1gain, ch2gain)
                        inited = True
                    print("dataThread: start capture")
                    scope.start_capture()
                    shutdown_event = scope.read_async(
                        callback, scope.packetsize, outstanding_transfers=10, raw=True
                    )
                    capture_running = True
            except queue.Empty:
                if capture_running:
                    scope.poll()
        print(f"dataThread {threading.current_thread().ident} end")
    except Exception as e:
        print(e)
        scope.close_handle()
        exit(1)


def guiThread(control_queue, data_queue):
    global data
    data = Data(1, 20000)
    global control_signal
    control_signal = False

    def control():
        print("guiThread: start/stop button clicked")
        global control_signal, scopeThread
        control_signal = not control_signal
        if control_signal:
            dpg.configure_item("control", label="关闭采样")
            scopeThread = threading.Thread(
                target=dataThread,
                args=(control_queue, data_queue, dpg.get_value("sample_rate")),
            )
            scopeThread.start()
            print("guiThread: start dataThread")
            control_queue.put(["start"])
            print("guiThread: send signal start")
        else:
            dpg.configure_item("control", label="开始采样")
            control_queue.put(["stop"])
            print("guiThread: send signal stop")
            scopeThread.join()

    def stopCallBack():
        global scopeThread
        print("guiThread: stop button clicked")
        control_queue.put(["stop"])
        print("guiThread: send signal stop")
        try:
            if scopeThread.is_alive():
                scopeThread.join()
        except NameError:
            pass
        dpg.stop_dearpygui()

    def set_sample_rate():
        global scopeThread
        print("guiThread: set sample rate to %s kS/s" % dpg.get_value("sample_rate"))
        data.set_sample_rate(int(dpg.get_value("sample_rate")) * 1000)
        dpg.set_axis_limits("x_axis", 0, data.get_point())
        dpg.set_value(
            "time_base_recommend",
            f"推荐时基: {1/int(dpg.get_value('sample_rate'))*1000} ms",
        )
        try:
            if scopeThread.is_alive():
                control_queue.put(["stop"])
                print("guiThread: send signal stop")
                scopeThread.join()
        except NameError:
            pass
        if control_signal:
            scopeThread = threading.Thread(
                target=dataThread,
                args=(control_queue, data_queue, dpg.get_value("sample_rate")),
            )
            scopeThread.start()
            print("guiThread: start dataThread")
            control_queue.put(["start"])
            print("guiThread: send signal start")

    global plot_ch1_display, plot_ch2_display
    plot_ch1_display = False
    plot_ch2_display = False

    def plot_ch1():
        global plot_ch1_display
        plot_ch1_display = not plot_ch1_display
        if plot_ch1_display:
            dpg.configure_item("plot_ch1_buttom", label="隐藏Ch1")
            dpg.configure_item("plot_ch1", show=True)
        else:
            dpg.configure_item("plot_ch1_buttom", label="显示Ch1")
            dpg.configure_item("plot_ch1", show=False)

    def plot_ch2():
        global plot_ch2_display
        plot_ch2_display = not plot_ch2_display
        if plot_ch2_display:
            dpg.configure_item("plot_ch2_buttom", label="隐藏Ch2")
            dpg.configure_item("plot_ch2", show=True)
        else:
            dpg.configure_item("plot_ch2_buttom", label="显示Ch2")
            dpg.configure_item("plot_ch2", show=False)

    global triggr_point, enabel_trigger, trigger_ch
    triggr_point = 0
    enabel_trigger = False
    trigger_ch = "ch1"

    def set_trigger_raising_point():
        global triggr_point
        triggr_point = dpg.get_value("trigger_point")

    def set_trigger_static_value_point():
        global triggr_point
        triggr_point = dpg.get_value("trigger_static_value")
        print("guiThread: set trigger point to ", triggr_point)

    def enable_trigger():
        global enabel_trigger
        enabel_trigger = not enabel_trigger
        if enabel_trigger:
            dpg.configure_item("enable_trigger", label="关闭触发")
        else:
            dpg.configure_item("enable_trigger", label="启动触发")

    def set_trigger_raising_ch():
        global trigger_ch
        trigger_ch = dpg.get_value("trigger_ch")

    def set_trigger_static_value_ch():
        global trigger_ch
        trigger_ch = dpg.get_value("trigger_static_value_ch")
        print("guiThread: set trigger channel to ", trigger_ch)

    def update_time():
        global data
        data.set_time(float(dpg.get_value("time_base")) / 1000)
        print("guiThread: set time to ", data.get_time())
        dpg.set_axis_limits("x_axis", 0, data.get_point())

    def set_plot():
        dpg.configure_item(
            "plot_ch1",
            max_scale=dpg.get_value("ch1_plot_max"),
            min_scale=dpg.get_value("ch1_plot_min"),
        )
        dpg.configure_item(
            "plot_ch2",
            max_scale=dpg.get_value("ch2_plot_max"),
            min_scale=dpg.get_value("ch2_plot_min"),
        )

    def set_sampling_resistor():
        global data
        data.sampling_resistor = dpg.get_value("Sampling_resistor")
        print("guiThread: set sampling resistor to ", data.sampling_resistor)

    global trigger_method, trigger_static_value_signal
    trigger_method = "rising"
    trigger_static_value_signal = False

    def set_trigger_method():
        global trigger_method, trigger_static_value_signal
        trigger_method = dpg.get_value("trigger_method")
        match trigger_method:
            case "上升沿":
                trigger_method = "rising"
            case "静态值":
                trigger_method = "static_value"
        if trigger_method == "rising":
            dpg.configure_item("trigger_raising_group", show=True)
            dpg.configure_item("trigger_static_value_group", show=False)
            """
            dpg.configure_item("trigger_point", enabled=True)
            dpg.configure_item("trigger_ch", enabled=True)
            dpg.configure_item("trigger_static_value", enabled=False)
            dpg.configure_item("trigger_static_value_ch", enabled=False)
            dpg.configure_item("trigger_static_value_signal", enabled=False) """
        elif trigger_method == "static_value":
            dpg.configure_item("trigger_static_value_group", show=True)
            dpg.configure_item("trigger_raising_group", show=False)
            """
            dpg.configure_item("trigger_point", enabled=False)
            dpg.configure_item("trigger_ch", enabled=False)
            dpg.configure_item("trigger_static_value", enabled=True)
            dpg.configure_item("trigger_static_value_ch", enabled=True)
            dpg.configure_item("trigger_static_value_signal", enabled=True)"""
            trigger_static_value_signal = False
        print("guiThread: set trigger method to ", trigger_method)

    def reset_trigger_static_value_signal():
        global trigger_static_value_signal
        trigger_static_value_signal = False
        print("guiThread: reset trigger static value signal")

    dpg.create_context()
    with dpg.font_registry():
        with dpg.font("sarasa-bold.ttc", 20) as font1:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)

            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)

    with dpg.window() as main_window:
        dpg.add_button(label="stop", callback=stopCallBack, tag="stop_buttom")
        with dpg.group(horizontal=True):
            dpg.add_button(label="开始采样", callback=control, tag="control")

            dpg.add_combo(
                label="kS/s采样率",
                items=[20, 32, 50, 64, 100, 128, 200],
                default_value=20,
                tag="sample_rate",
                callback=set_sample_rate,
                width=100,
            )
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="时基",
                default_value=50,
                tag="time_base",
                callback=update_time,
                width=200,
                step=1,
            )
            dpg.add_text("推荐时基:50ms", tag="time_base_recommend")
        with dpg.group(horizontal=True):
            dpg.add_button(label="启动触发", callback=enable_trigger, tag="enable_trigger")
            dpg.add_combo(
                label="触发方式",
                items=["上升沿", "静态值"],
                default_value="上升沿",
                tag="trigger_method",
                callback=set_trigger_method,
                width=200,
            )
        
        with dpg.collapsing_header(label="配置静态值触发" , tag="trigger_static_value_group"):
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="静态值触发值",
                    default_value=1,
                    tag="trigger_static_value",
                    callback=set_trigger_static_value_point,
                    width=200,
                )
                dpg.add_combo(
                    label="触发通道",
                    items=["ch1", "ch2"],
                    default_value="ch1",
                    tag="trigger_static_value_ch",
                    callback=set_trigger_static_value_ch,
                    width=150,
                )
                dpg.add_button(
                    label="重置触发",
                    callback=reset_trigger_static_value_signal,
                    tag="trigger_static_value_signal",
            )
        with dpg.collapsing_header(label="配置上升沿触发" , tag="trigger_raising_group"):
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="触发点",
                    default_value=1,
                    tag="trigger_point",
                    callback=set_trigger_raising_point,
                    width=200,
                )
                dpg.add_combo(
                    label="触发通道",
                    items=["ch1", "ch2"],
                    default_value="ch1",
                    tag="trigger_ch",
                    callback=set_trigger_raising_ch,
                    width=150,
                )

        with dpg.group(horizontal=True):
            dpg.add_text("ch1: ", tag="ch1_current")
            dpg.add_text("ch1_Vmin: ", tag="ch1_Vmin")
            dpg.add_text("ch1_Vmax: ", tag="ch1_Vmax")
            dpg.add_text("ch1_Vavg: ", tag="ch1_Vavg")
            dpg.add_text("ch1_Vrms: ", tag="ch1_Vrms")

        with dpg.group(horizontal=True):
            dpg.add_text("ch2: ", tag="ch2_current")
            dpg.add_text("ch2_Vmin: ", tag="ch2_Vmin")
            dpg.add_text("ch2_Vmax: ", tag="ch2_Vmax")
            dpg.add_text("ch2_Vavg: ", tag="ch2_Vavg")
            dpg.add_text("ch2_Vrms: ", tag="ch2_Vrms")

        with dpg.collapsing_header(label="统计Q" , tag="total_Q_group"):
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="采样电阻大小", default_value=100, tag="Sampling_resistor",callback=set_sampling_resistor, width=200)
                dpg.add_text("ch1_total_Q: ", tag="ch1_total_Q")
                dpg.add_text("ch2_total_Q: ", tag="ch2_total_Q")
                dpg.add_button(label="重置", callback=data.reset_total_Q)

        with dpg.plot(label="plot_ch1", height=400, width=800, tag="plot_ch1_"):
            dpg.add_plot_legend()
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis", label="time(ms)")
            dpg.set_axis_limits(xaxis, 0, 2000)
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis", label="voltage(V)")
            dpg.add_line_series([0], [0], tag="plot_ch1", parent=yaxis, label="ch1")
            dpg.add_line_series([0], [0], tag="plot_ch2", parent=yaxis, label="ch2")

        dpg.bind_font(font1)

    dpg.create_viewport(width=900, height=600, title="Updating plot data")
    dpg.setup_dearpygui()
    dpg.set_primary_window(main_window, True)
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        if control_signal:
            if not data_queue.empty():
                while not data_queue.empty():
                    data.add(*data_queue.get())

            tem_data = data.get()

            def round_plus(x, n):
                if x >= 0:
                    return "+{:>{width}.4f}".format(x, width=n)
                else:
                    return "-{:>{width}.4f}".format(-x, width=n).replace("-", "\u2009-")

            if len(tem_data[0]) > 0:
                ch1_Vmin = min(tem_data[0])
                ch1_Vmax = max(tem_data[0])
                ch1_Vavg = sum(tem_data[0]) / len(tem_data[0])
                ch1_Vrms = math.sqrt(
                    sum([i**2 for i in tem_data[0]]) / len(tem_data[0])
                )
                ch2_Vmin = min(tem_data[1])
                ch2_Vmax = max(tem_data[1])
                ch2_Vavg = sum(tem_data[1]) / len(tem_data[1])
                ch2_Vrms = math.sqrt(
                    sum([i**2 for i in tem_data[1]]) / len(tem_data[1])
                )
                dpg.set_value("ch1_current", f"ch1: {round_plus(tem_data[0][-1],3)}")
                dpg.set_value("ch1_Vmin", f"ch1_Vmin: {round_plus(ch1_Vmin,3)}")
                dpg.set_value("ch1_Vmax", f"ch1_Vmax: {round_plus(ch1_Vmax,3)}")
                dpg.set_value("ch1_Vavg", f"ch1_Vavg: {round_plus(ch1_Vavg,3)}")
                dpg.set_value("ch1_Vrms", f"ch1_Vrms: {round_plus(ch1_Vrms,3)}")
                dpg.set_value("ch2_current", f"ch2: {round_plus(tem_data[1][-1],3)}")
                dpg.set_value("ch2_Vmin", f"ch2_Vmin: {round_plus(ch2_Vmin,3)}")
                dpg.set_value("ch2_Vmax", f"ch2_Vmax: {round_plus(ch2_Vmax,3)}")
                dpg.set_value("ch2_Vavg", f"ch2_Vavg: {round_plus(ch2_Vavg,3)}")
                dpg.set_value("ch2_Vrms", f"ch2_Vrms: {round_plus(ch2_Vrms,3)}")
                if not trigger_static_value_signal:
                    ch1_total_Q, ch2_total_Q = data.get_total_Q()
                    dpg.set_value("ch1_total_Q", f"ch1_total_Q: {round_plus(ch1_total_Q,5)}")
                    dpg.set_value("ch2_total_Q", f"ch2_total_Q: {round_plus(ch2_total_Q,5)}")
            if enabel_trigger:
                if trigger_method == "rising":
                    if trigger_ch == "ch1":
                        for i in range(len(tem_data[0]) - 1):
                            trigger_index = 0
                            if (
                                tem_data[0][i] <= triggr_point
                                and tem_data[0][i + 1] >= triggr_point
                            ):
                                trigger_index = i
                                break
                        dpg.set_value(
                            "plot_ch1",
                            [
                                [i for i in range(len(tem_data[0][trigger_index:]))],
                                tem_data[0][trigger_index:],
                            ],
                        )
                        dpg.set_value(
                            "plot_ch2",
                            [
                                [i for i in range(len(tem_data[1][trigger_index:]))],
                                tem_data[1][trigger_index:],
                            ],
                        )
                    elif trigger_ch == "ch2":
                        for i in range(len(tem_data[1]) - 1):
                            trigger_index = 0
                            if (
                                tem_data[1][i] <= triggr_point
                                and tem_data[1][i + 1] >= triggr_point
                            ):
                                trigger_index = i
                                break
                        dpg.set_value(
                            "plot_ch1",
                            [
                                [i for i in range(len(tem_data[0][trigger_index:]))],
                                tem_data[0][trigger_index:],
                            ],
                        )
                        dpg.set_value(
                            "plot_ch2",
                            [
                                [i for i in range(len(tem_data[1][trigger_index:]))],
                                tem_data[1][trigger_index:],
                            ],
                        )
                elif (
                    trigger_method == "static_value" and not trigger_static_value_signal
                ):
                    if trigger_ch == "ch1":
                        for i in range(len(tem_data[0]) - 1):
                            if (
                                tem_data[0][i] <= triggr_point
                                and tem_data[0][i + 1] > triggr_point
                            ):
                                dpg.set_value(
                                    "plot_ch1",
                                    [[i for i in range(len(tem_data[0]))], tem_data[0]],
                                )
                                dpg.set_value(
                                    "plot_ch2",
                                    [[i for i in range(len(tem_data[1]))], tem_data[1]],
                                )
                                trigger_static_value_signal = True
                                print("guiThread:static value trigged")
                                break
                    elif trigger_ch == "ch2":
                        for i in range(len(tem_data[1]) - 1):
                            if (
                                tem_data[1][i] <= triggr_point
                                and tem_data[1][i + 1] > triggr_point
                            ):
                                dpg.set_value(
                                    "plot_ch1",
                                    [[i for i in range(len(tem_data[0]))], tem_data[0]],
                                )
                                dpg.set_value(
                                    "plot_ch2",
                                    [[i for i in range(len(tem_data[1]))], tem_data[1]],
                                )
                                trigger_static_value_signal = True
                                print("guiThread:static value trigged")
                                break
            else:
                dpg.set_value(
                    "plot_ch1", [[i for i in range(len(tem_data[0]))], tem_data[0]]
                )
                dpg.set_value(
                    "plot_ch2", [[i for i in range(len(tem_data[1]))], tem_data[1]]
                )
        dpg.render_dearpygui_frame()

    time.sleep(0.1)
    dpg.destroy_context()


if __name__ == "__main__":
    control_queue = queue.Queue()
    data_queue = queue.Queue()
    Guithread = threading.Thread(target=guiThread, args=(control_queue, data_queue))
    Guithread.start()
    Guithread.join()

    print("done")
