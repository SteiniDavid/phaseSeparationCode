import os
import ovito
from ovito.anim import *
from ovito.data import *
from ovito.io import import_file
from ovito.modifiers import *
from ovito.vis import *

import sys

#infile = str(sys.argv[1])
#bin_value = int(sys.argv[2])                            # theory prediction

infile = 'pa90_pb150_xa80LF.gsd'
bin_value = 1

outfile, file_extension = os.path.splitext(infile)      # get base name

node = import_file(infile, multiple_frames = True)      # load file as trajectory
node.add_to_scene()

a = 0
final = node.source.num_frames - 1                      # index of last frame

# both phase sep
if bin_value == 3:                      # should be yellow
    r_col = 1.0
    g_col = 1.0
    b_col = 0.0
# both gas
elif bin_value == 2:                    # should be white
    r_col = 1.0
    g_col = 1.0
    b_col = 1.0
# theory phase sep
elif bin_value == 1:                    # should be green
    r_col = 0.0
    g_col = 1.0
    b_col = 0.0
# sim phase sep
else:                                   # should be cyan
    r_col = 0.2
    g_col = 1.0
    b_col = 0.9

# made this a while loop so that I could handle exceptions
while a == 0:
    rs = RenderSettings(
        filename = "final_tstep_" + outfile + ".png",
        size = (2000,2000),
        #generate_alpha = True,
        #background_color = (r_col, g_col, b_col),
        renderer = OpenGLRenderer()
    )                                                       # settings for render

    #part_a = node.output.attributes['SelectParticleType.num_selected']
    ovito.dataset.anim.current_frame = final                # grab final snapshot
    vp = ovito.dataset.viewports.active_vp

    #ptp = node.source.particle_properties.particle_type
    #print(ptp.array)

    node.modifiers.append(SelectParticleTypeModifier(property='Particle Type',
                                                     types={0}))
    node.modifiers.append(AssignColorModifier(color=(1.0, 0.3, 0.3)))
    
    # catch any corrupted data errors
    try:
        node.compute()                                          # set color of type A
        a = 1
    except Exception:
        final -= 1
        a = 0

    node.modifiers.append(SelectParticleTypeModifier(property='Particle Type',
                                                     types={1}))
    node.modifiers.append(AssignColorModifier(color=(0.3, 0.3, 1.0)))

    # this block will only run if we haven't thrown an exception
    if a == 1:
        node.compute()                                          # set color of type B
        vp.type = Viewport.Type.TOP                             # top view
        vp.zoom_all()                                           # zoom to fit
        vp.render(rs)                                           # render image
#    else:
#        print("Looping back through!")
