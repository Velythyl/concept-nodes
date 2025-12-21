# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys

isMacOS = (platform.system() == "Darwin")



def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window(
        "Open3D", 1024, 768)
    w = window  # to make the code more concise

    # 3D widget
    _scene = gui.SceneWidget()
    _scene.scene = rendering.Open3DScene(w.renderer)

    # ---- Settings panel ----
    em = w.theme.font_size
    separation_height = int(round(0.5 * em))

    _settings_panel = gui.Vert(
        0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

    #view_ctrls = gui.CollapsableVert("Rise Composer Overview", 0.25 * em,
    #                      gui.Margins(em, 0, 0, 0))
    _settings_panel.add_child(gui.Label("Mouse controls"))

    _settings_panel.add_fixed(separation_height)

    def _on_layout(layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = window.content_rect
        _scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            _settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        _settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    w.set_on_layout(_on_layout)
    w.add_child(_scene)
    w.add_child(_settings_panel)

    return _scene, gui.Application.instance
    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()