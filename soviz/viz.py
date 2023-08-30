

from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
import xml.sax
from xml.dom.minidom import parse
import xml.dom.minidom
import pydot
import os
import xml.etree.ElementTree as ET
from random import uniform
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


# In[3]:


graph = {}

path = dir_path+ '/softwares/bb'
path2 = dir_path
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    DOMtree =xml.dom.minidom.parse(fullname)
    collection = DOMtree.documentElement
    compound_name = collection.getElementsByTagName('compoundname')
    nodes= collection.getElementsByTagName('label')
    temp_lst =[]
    for node in nodes:
     temp_lst.append(str(node.childNodes[0].data))
    graph[compound_name[0].childNodes[0].data] = temp_lst
#print(graph)
#print("-------------------------------------------------------------------------------------------------\n")




for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    DOMtree =xml.dom.minidom.parse(fullname)
    collection = DOMtree.documentElement
    compound_name = collection.getElementsByTagName('compoundname')
    includes = collection.getElementsByTagName('includes')
    temp_lst =[]
    for include in includes:
     temp_lst.append(str(include.childNodes[0].data))
    graph[compound_name[0].childNodes[0].data] = temp_lst
#print(graph)







g = pydot.Dot(graph_type = "graph" )

for node,neighbours in graph.items():
    gnode = pydot.Node(str(node),style = "filled",fillcolor = "yellow")
    g.add_node(gnode)
    for neighbour in neighbours:
        gedge = pydot.Edge(str(neighbour),gnode)
        g.add_edge(gedge)
im = g.write_png('viz1.png',prog = 'dot')

#imgplot = plt.imshow(im, aspect='equal')
#plt.show(block=False)
#print("-------------------------------------------------------------------------------\n")
graph = {}


# In[8]:




path = '/home/shunya/setool/xmls'
path2 = './'

graph = {}


comdef = []

for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    DOMtree =xml.dom.minidom.parse(fullname)
    collection = DOMtree.documentElement
    compound_name = collection.getElementsByTagName('compoundname')
    comdef.append(compound_name)
    includes = collection.getElementsByTagName('includes')
    temp_lst =[]
    for include in includes:
         temp_lst.append(str(include.childNodes[0].data))
    graph[compound_name[0].childNodes[0].data] = temp_lst
nodelist = []
connections2 = []
for node, neighbours in graph.items():
    for neighbour in neighbours:
        connections2.append((node,neighbour))

nodecoord = {}
coord2 = []
for node1,node2 in connections2:
    if node1 not in nodelist:
        nodelist.append(node1)
        nodecoord[node1] = (uniform(-180,180),uniform(-180,180))
        coord2.append(nodecoord[node1])
    if node2 not in nodelist:
        nodelist.append(node2)
        nodecoord[node2] = (uniform(-180,180),uniform(-180,180))
        coord2.append(nodecoord[node2])
e_nodelist = {}
for i,node in enumerate(nodelist):
    e_nodelist[node] = i


connections3 = []
for connection in connections2:
    connections3.append((e_nodelist[connection[0]],e_nodelist[connection[1]]))







#  Cities = e_nodelist
#  connections = connections3
#  coords =  coord2
#print(e_nodelist)
#print(connections2)
#print('\n________________________________________________________\n')
#print(nodecoord)


# In[9]:


from mayavi import mlab
mlab.figure(1, bgcolor=(0.48, 0.48, 0.48), fgcolor=(0, 0, 0),
               size=(400, 400))
mlab.clf()

###############################################################################
# Display points at random positions
import numpy as np
coord2 = np.array(coord2)
# First we have to convert latitude/longitude information to 3D
# positioning.
lat, long = coord2.T * np.pi / 180
x = np.cos(long) * np.cos(lat)
y = np.cos(long) * np.sin(lat)
z = np.sin(long)

points = mlab.points3d(x, y, z,
                     scale_mode='none',
                     scale_factor=0.03,
                     color=(0, 0, 1))

connections3 = np.array(connections3)
# We add lines between the points that we have previously created by

points.mlab_source.dataset.lines = connections3
points.mlab_source.reset()
# To represent the lines, we use the surface module. Using a wireframe
# representation allows to control the line-width.
mlab.pipeline.surface(points, color=(1, 1, 1),
                              representation='wireframe',
                              line_width=1,
                              name='Connections')

# Display city names
#for city, index in e_nodelist.items():
    #label = mlab.text(x[index], y[index], city, z=z[index],
                      #width=0.016 * len(city), name=city)
    #label.property.shadow = True


###############################################################################
# Display continents outline, using the VTK Builtin surface 'Earth'
from mayavi.sources.builtin_surface import BuiltinSurface
continents_src = BuiltinSurface(source='earth', name='Continents')
# The on_ratio of the Earth source controls the level of detail of the
# continents outline.
continents_src.data_source.on_ratio = 2
continents = mlab.pipeline.surface(continents_src, color=(0, 0, 0))

###############################################################################



###############################################################################
# Display a semi-transparent sphere, for the surface of the Earth

# We use a sphere Glyph, throught the points3d mlab function, rather than
# building the mesh ourselves, because it gives a better transparent
# rendering.
sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                                scale_factor=0.2,
                                color=(0.67, 0.77, 0.93),
                                resolution=50,
                                opacity=0.7,
                                name='Earth')

# These parameters, as well as the color, where tweaked through the GUI,
# with the record mode to produce lines of code usable in a script.
sphere.actor.property.specular = 0.45
sphere.actor.property.specular_power = 5
# Backface culling is necessary for more a beautiful transparent
# rendering.
sphere.actor.property.backface_culling = True

###############################################################################

# Plot the equator and the tropiques
theta = np.linspace(0, 2 * np.pi, 100)
for angle in (- np.pi / 6, 0, np.pi / 6):
    x = np.cos(theta) * np.cos(angle)
    y = np.sin(theta) * np.cos(angle)
    z = np.ones_like(theta) * np.sin(angle)

    mlab.plot3d(x, y, z, color=(1, 1, 1),
                        opacity=0.2, tube_radius=None)

mlab.view(63.4, 73.8, 4, [-0.05, 0, 0])
mlab.show()


# In[6]:


path = 
fil_met = {}
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    DOMtree =xml.dom.minidom.parse(fullname)
    tree = ET.parse(fullname)
    root = tree.getroot()
    compound_name2 = root.findall('compounddef')
    #print(compound_name2[0].find('compoundname').text)
    members2 = root.iter('memberdef')
    count_d = 0
    count_f = 0
    for member in members2:
        if member.get('kind') == 'define':
            count_d += 1
            #print('d : ',member.find('name').text)

        else:
            count_f += 1
            #print('f : ',member.find('name').text)
    fil_met[compound_name2[0].find('compoundname').text] = [count_d,count_f]
    collection = DOMtree.documentElement
    compound_name = collection.getElementsByTagName('coumpundname')
    members= collection.getElementsByTagName('memberdef')
    temp_lst =[]

print(fil_met)


# In[7]:


import numpy as np

# Create data with x and y random in the [-2, 2] segment, and z a
# Gaussian function of x and y.
np.random.seed(12345)
x = 4 * (np.random.random(500) - 0.5)
y = 4 * (np.random.random(500) - 0.5)

x1 = list()
x2 = list()
for a,b in fil_met.items():
    x1.append(np.random.choice([-1,1])*sum(b))
    x2.append(np.random.choice([-1,1])*len(graph[a]))
print(x[0:5],y[0:5])
print(x1,x2)
x1 = np.array(x1)
x2 = np.array(x2)
x1 = np.divide(x1,max(np.absolute(x1)))
x2 = np.divide(x2,max(np.absolute(x2)))
x1 = 4*x1
x2 = 4*x2
print(x1,x2)
def f(x, y):
    return np.exp(-(x ** 2 + y ** 2))

z = f(x, y)
z2 = f(x1,x2)
from mayavi import mlab

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

pts2 = mlab.points3d(x1,x2,z2,z2,scale_mode = 'none',scale_factor = 0.2)
mesh2 = mlab.pipeline.delaunay2d(pts2)
surf = mlab.pipeline.surface(mesh2)
mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
mlab.show()

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

# Visualize the points
pts = mlab.points3d(x, y, z, z, scale_mode='none', scale_factor=0.2)
# Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)

surf = mlab.pipeline.surface(mesh)

mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
mlab.show()
